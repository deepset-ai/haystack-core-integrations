# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

logger = logging.getLogger(__name__)

# Number of documents sent per UNWIND batch to avoid memory spikes.
_WRITE_BATCH_SIZE = 100

# Haystack filter operators that map to Cypher comparison operators.
_COMPARISON_OPS: dict[str, str] = {
    "==": "=",
    "!=": "<>",
    ">": ">",
    ">=": ">=",
    "<": "<",
    "<=": "<=",
}


class SimilarityFunction(str, Enum):
    """
    Similarity functions supported by FalkorDB's vector index.

    FalkorDB currently supports cosine and euclidean (L2) distance.
    """

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


class FalkorDBDocumentStore:
    """
    A Haystack DocumentStore backed by FalkorDB — a high-performance graph database
    optimised for GraphRAG workloads.

    Documents are stored as ``(:Document)`` nodes in a named FalkorDB graph.
    Vector search is performed via FalkorDB's native vector index (no APOC required).
    All bulk writes use ``UNWIND`` + ``MERGE`` for safe, idiomatic OpenCypher upserts.

    Usage example:
    ```python
    from haystack_integrations.document_stores.falkor_db import FalkorDBDocumentStore
    from haystack.dataclasses import Document

    store = FalkorDBDocumentStore(host="localhost", port=6379, graph_name="haystack")
    store.write_documents([Document(content="Hello, GraphRAG!")])
    print(store.count_documents())  # 1
    ```
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6379,
        graph_name: str = "haystack",
        username: str | None = None,
        password: Secret | None = None,
        embedding_dim: int = 768,
        similarity: SimilarityFunction | str = SimilarityFunction.COSINE,
        recreate_index: bool = False,
    ) -> None:
        """
        Create a new FalkorDBDocumentStore.

        :param host: Hostname of the FalkorDB server. Defaults to ``"localhost"``.
        :param port: Port the FalkorDB server listens on. Defaults to ``6379``.
        :param graph_name: Name of the FalkorDB graph to use. Each graph is an isolated
            namespace. Defaults to ``"haystack"``.
        :param username: Optional username for FalkorDB authentication.
        :param password: Optional :class:`haystack.utils.Secret` holding the FalkorDB
            password. The secret value is resolved lazily on first connection.
        :param embedding_dim: Dimensionality of the vector embeddings stored in this
            graph. Used when creating the vector index. Defaults to ``768``.
        :param similarity: Similarity / distance function used by the vector index.
            Must be a :class:`SimilarityFunction` value or its string equivalent.
            Defaults to ``"cosine"``.
        :param recreate_index: When ``True`` the existing graph (and all its data) is
            dropped and recreated on initialisation. Useful for testing. Defaults to
            ``False``.
        """
        self._host = host
        self._port = port
        self._graph_name = graph_name
        self._username = username
        self._password = password
        self._embedding_dim = embedding_dim
        self._similarity = (
            SimilarityFunction(similarity) if isinstance(similarity, str) else similarity
        )
        self._recreate_index = recreate_index

        # Lazy — populated on first use via _ensure_connected().
        self._client: Any | None = None
        self._graph: Any | None = None
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Internal connection helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """
        Lazily open the FalkorDB connection and set up the graph schema.

        This is called at the start of every public method so that the store
        remains serialisable without an active connection.
        """
        if self._initialized:
            return

        import falkordb  # noqa: PLC0415 — intentional lazy import

        password_value = self._password.resolve_value() if self._password is not None else None

        self._client = falkordb.FalkorDB(
            host=self._host,
            port=self._port,
            username=self._username,
            password=password_value,
        )

        if self._recreate_index:
            try:
                self._client.delete(self._graph_name)
            except Exception:  # noqa: BLE001 — graph may not exist yet
                pass

        self._graph = self._client.select_graph(self._graph_name)
        self._ensure_schema()
        self._initialized = True

    def _ensure_schema(self) -> None:
        """
        Create the node index and vector index if they do not already exist.

        Uses only standard OpenCypher / FalkorDB-native syntax — no APOC.
        """
        # Property index on :Document(id) for fast MERGE lookups.
        try:
            self._graph.query("CREATE INDEX FOR (d:Document) ON (d.id)")
        except Exception:  # noqa: BLE001 — index may already exist
            pass

        # FalkorDB vector index: CALL db.idx.vector.createNodeIndex(label, property, dim, metric)
        try:
            self._graph.query(
                "CALL db.idx.vector.createNodeIndex($label, $prop, $dim, $metric)",
                {
                    "label": "Document",
                    "prop": "embedding",
                    "dim": self._embedding_dim,
                    "metric": self._similarity.value,
                },
            )
        except Exception:  # noqa: BLE001 — index may already exist
            pass

    # ------------------------------------------------------------------
    # Haystack DocumentStore protocol
    # ------------------------------------------------------------------

    def count_documents(self) -> int:
        """
        Return the number of documents currently stored in the graph.

        :returns: Integer count of ``(:Document)`` nodes.
        """
        self._ensure_connected()
        result = self._graph.query("MATCH (d:Document) RETURN count(d) AS n")
        rows = result.result_set
        return int(rows[0][0]) if rows else 0

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Retrieve all documents that match the provided Haystack filters.

        :param filters: Optional Haystack filter dict. When ``None`` all documents are
            returned. For filter syntax see
            https://docs.haystack.deepset.ai/docs/metadata-filtering
        :returns: List of matching :class:`haystack.dataclasses.Document` objects.
        :raises ValueError: If the filter dict is malformed.
        """
        self._ensure_connected()

        if filters is not None and "operator" not in filters and "conditions" not in filters:
            msg = "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering"
            raise ValueError(msg)

        if filters:
            where_clause, params = _convert_filters(filters)
            cypher = f"MATCH (d:Document) WHERE {where_clause} RETURN d"
        else:
            cypher = "MATCH (d:Document) RETURN d"
            params = {}

        result = self._graph.query(cypher, params)
        return [_node_to_document(row[0]) for row in result.result_set]

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Write documents to the FalkorDB graph using ``UNWIND`` + ``MERGE`` for batching.

        Batch size is capped at :data:`_WRITE_BATCH_SIZE` (100) per query to avoid
        memory pressure.  No APOC is used — all upsert logic is expressed in standard
        OpenCypher ``ON CREATE SET`` / ``ON MATCH SET`` clauses.

        :param documents: List of :class:`haystack.dataclasses.Document` objects to write.
        :param policy: How to handle documents whose ``id`` already exists in the store.
            Defaults to :attr:`DuplicatePolicy.NONE` (treated as FAIL).
        :raises ValueError: If ``documents`` is not a list of ``Document`` objects.
        :raises DuplicateDocumentError: If ``policy`` is FAIL / NONE and a duplicate ID
            is encountered.
        :raises DocumentStoreError: If any other error occurs during the write.
        :returns: Number of documents written (or updated).
        """
        self._ensure_connected()

        if not documents:
            return 0

        if not isinstance(documents[0], Document):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        written = 0
        for batch_start in range(0, len(documents), _WRITE_BATCH_SIZE):
            batch = documents[batch_start : batch_start + _WRITE_BATCH_SIZE]
            written += self._write_batch(batch, policy)

        return written

    def _write_batch(self, documents: list[Document], policy: DuplicatePolicy) -> int:
        """
        Write a single batch of documents using UNWIND.

        :param documents: Batch of documents (≤ _WRITE_BATCH_SIZE).
        :param policy: Duplicate handling policy.
        :returns: Number of documents successfully written / updated.
        """
        docs_data = [_document_to_node_props(doc) for doc in documents]

        if policy == DuplicatePolicy.OVERWRITE:
            cypher = """
UNWIND $docs AS doc
MERGE (d:Document {id: doc.id})
ON CREATE SET d += doc
ON MATCH SET d += doc
RETURN count(d) AS n
"""
        elif policy == DuplicatePolicy.SKIP:
            cypher = """
UNWIND $docs AS doc
MERGE (d:Document {id: doc.id})
ON CREATE SET d += doc
RETURN count(d) AS n
"""
        else:
            # FAIL policy: detect pre-existing nodes before writing.
            ids = [d["id"] for d in docs_data]
            existing = self._graph.query(
                "UNWIND $ids AS id MATCH (d:Document {id: id}) RETURN d.id",
                {"ids": ids},
            )
            existing_ids = {row[0] for row in existing.result_set}
            if existing_ids:
                msg = f"IDs {sorted(existing_ids)!r} already exist in the document store."
                raise DuplicateDocumentError(msg)

            cypher = """
UNWIND $docs AS doc
MERGE (d:Document {id: doc.id})
ON CREATE SET d += doc
RETURN count(d) AS n
"""

        try:
            result = self._graph.query(cypher, {"docs": docs_data})
            rows = result.result_set
            return int(rows[0][0]) if rows else len(documents)
        except DuplicateDocumentError:
            raise
        except Exception as exc:
            msg = f"Failed to write documents to FalkorDB: {exc}"
            raise DocumentStoreError(msg) from exc

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Delete documents by their IDs using a single ``UNWIND``-based query.

        :param document_ids: List of document IDs to remove from the graph.
        """
        self._ensure_connected()
        if not document_ids:
            return
        self._graph.query(
            "UNWIND $ids AS id MATCH (d:Document {id: id}) DETACH DELETE d",
            {"ids": document_ids},
        )

    # ------------------------------------------------------------------
    # Internal retrieval helpers (called by retriever components)
    # ------------------------------------------------------------------

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieve documents by vector similarity using FalkorDB's native vector index.

        The query uses ``CALL db.idx.vector.queryNodes`` — FalkorDB's OpenCypher
        extension for ANN search.  No APOC is required.

        :param query_embedding: Query vector as a plain Python list of floats.
        :param top_k: Maximum number of results to return.
        :param filters: Optional Haystack filters applied as a ``WHERE`` predicate
            inside the vector search result set.
        :returns: List of :class:`Document` objects ordered by similarity (best first).
        """
        self._ensure_connected()

        if filters:
            where_clause, filter_params = _convert_filters(filters)
            cypher = f"""
CALL db.idx.vector.queryNodes('Document', 'embedding', $top_k, vecf32($query_embedding))
YIELD node AS d, score
WHERE {where_clause}
RETURN d, score
ORDER BY score ASC
"""
            params: dict[str, Any] = {
                "top_k": top_k,
                "query_embedding": query_embedding,
                **filter_params,
            }
        else:
            cypher = """
CALL db.idx.vector.queryNodes('Document', 'embedding', $top_k, vecf32($query_embedding))
YIELD node AS d, score
RETURN d, score
ORDER BY score ASC
"""
            params = {"top_k": top_k, "query_embedding": query_embedding}

        result = self._graph.query(cypher, params)
        documents = []
        for row in result.result_set:
            node, score = row[0], row[1]
            doc = _node_to_document(node)
            doc.score = float(score)
            documents.append(doc)
        return documents

    def _cypher_retrieval(
        self,
        cypher_query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Execute an arbitrary OpenCypher query and map the results to Documents.

        Each result row must contain at least one node or map that can be coerced
        into a :class:`Document`.  The first element of each row is used.

        :param cypher_query: A valid OpenCypher query string.
        :param parameters: Optional query parameters (``$param`` placeholders).
        :returns: List of :class:`Document` objects built from the query results.
        :raises DocumentStoreError: If the query fails.
        """
        self._ensure_connected()
        try:
            result = self._graph.query(cypher_query, parameters or {})
            return [_node_to_document(row[0]) for row in result.result_set]
        except Exception as exc:
            msg = f"Cypher query failed: {exc}"
            raise DocumentStoreError(msg) from exc

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the document store to a dictionary suitable for pipeline YAML.

        :returns: Dictionary representation of this store's configuration.
        """
        return default_to_dict(
            self,
            host=self._host,
            port=self._port,
            graph_name=self._graph_name,
            username=self._username,
            password=self._password.to_dict() if self._password is not None else None,
            embedding_dim=self._embedding_dim,
            similarity=self._similarity.value,
            recreate_index=self._recreate_index,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FalkorDBDocumentStore":
        """
        Deserialise a FalkorDBDocumentStore from a dictionary.

        :param data: Dictionary previously produced by :meth:`to_dict`.
        :returns: A new :class:`FalkorDBDocumentStore` instance.
        """
        init_params = data.get("init_parameters", data)
        if (pwd := init_params.get("password")) is not None and isinstance(pwd, dict):
            init_params["password"] = Secret.from_dict(pwd)
        return default_from_dict(cls, data)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _document_to_node_props(doc: Document) -> dict[str, Any]:
    """
    Convert a Haystack Document to a flat dict suitable for storing as node properties.

    FalkorDB stores float arrays natively, so embeddings are passed as-is.
    Metadata is flattened into top-level properties with a ``meta_`` prefix to
    avoid collisions with reserved property names (``id``, ``content``, ``score``).

    :param doc: The document to convert.
    :returns: Flat dictionary of node properties.
    """
    props: dict[str, Any] = {"id": doc.id}

    if doc.content is not None:
        props["content"] = doc.content

    if doc.embedding is not None:
        props["embedding"] = doc.embedding

    if doc.meta:
        for key, value in doc.meta.items():
            props[f"meta_{key}"] = value

    return props


def _node_to_document(node: Any) -> Document:
    """
    Convert a FalkorDB graph node (or property map) back to a Haystack Document.

    :param node: A FalkorDB ``Node`` object or a ``dict``-like mapping.
    :returns: Reconstructed :class:`haystack.dataclasses.Document`.
    """
    # FalkorDB Node objects expose their properties via .properties dict.
    if hasattr(node, "properties"):
        props: dict[str, Any] = dict(node.properties)
    elif isinstance(node, dict):
        props = node
    else:
        props = {}

    doc_id: str = props.pop("id", "")
    content: str | None = props.pop("content", None)
    embedding: list[float] | None = props.pop("embedding", None)
    score: float | None = props.pop("score", None)

    # Re-assemble metadata from the ``meta_`` prefixed properties.
    meta: dict[str, Any] = {}
    remaining = dict(props)
    for key in list(remaining):
        if key.startswith("meta_"):
            meta[key[5:]] = remaining.pop(key)

    return Document(
        id=doc_id,
        content=content,
        embedding=list(embedding) if embedding is not None else None,
        score=score,
        meta=meta,
    )


def _convert_filters(filters: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Translate a Haystack filter dict into an OpenCypher ``WHERE`` sub-expression.

    Supports the full Haystack filter DSL:
    - Logical operators: ``AND``, ``OR``, ``NOT``
    - Comparison operators: ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``
    - Membership operators: ``in``, ``not in``

    All values are passed as named query parameters to prevent injection.

    :param filters: A Haystack filter dictionary.
    :returns: Tuple of ``(where_clause_string, params_dict)``.
    :raises ValueError: If an unsupported operator or malformed filter is found.
    """
    params: dict[str, Any] = {}
    clause = _build_clause(filters, params, counter=[0])
    return clause, params


def _build_clause(node: dict[str, Any], params: dict[str, Any], counter: list[int]) -> str:
    """
    Recursively build a Cypher WHERE sub-expression from a Haystack filter node.

    :param node: A filter node (logical group or comparison leaf).
    :param params: Accumulating query parameter dict (mutated in place).
    :param counter: Single-element list used as a mutable integer for unique param names.
    :returns: Cypher expression string.
    """
    operator = node.get("operator", "")

    # ------------------------------------------------------------------
    # Logical / grouping operators
    # ------------------------------------------------------------------
    if operator.upper() in ("AND", "OR"):
        sub_clauses = [_build_clause(c, params, counter) for c in node["conditions"]]
        joiner = f" {operator.upper()} "
        return f"({joiner.join(sub_clauses)})"

    if operator.upper() == "NOT":
        inner = _build_clause(node["conditions"][0], params, counter)
        return f"NOT ({inner})"

    # ------------------------------------------------------------------
    # Leaf (comparison / membership) operators
    # ------------------------------------------------------------------
    field: str = node.get("field", "")
    value: Any = node.get("value")

    # Map Haystack field names to Cypher property access.
    # Top-level fields (content, id) are accessed directly; metadata fields
    # are stored with a ``meta_`` prefix (see _document_to_node_props).
    if field in ("id", "content", "embedding"):
        cypher_field = f"d.{field}"
    else:
        cypher_field = f"d.meta_{field}"

    param_name = f"p{counter[0]}"
    counter[0] += 1

    if operator in _COMPARISON_OPS:
        params[param_name] = value
        return f"{cypher_field} {_COMPARISON_OPS[operator]} ${param_name}"

    if operator == "in":
        params[param_name] = list(value)
        return f"{cypher_field} IN ${param_name}"

    if operator == "not in":
        params[param_name] = list(value)
        return f"NOT ({cypher_field} IN ${param_name})"

    msg = f"Unsupported filter operator: '{operator}'"
    raise ValueError(msg)
