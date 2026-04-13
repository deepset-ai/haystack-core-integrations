# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Literal

import numpy as np
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

logger = logging.getLogger(__name__)

# Haystack filter operators → Cypher comparison operators.
_COMPARISON_OPS: dict[str, str] = {
    "==": "=",
    "!=": "<>",
    ">": ">",
    ">=": ">=",
    "<": "<",
    "<=": "<=",
}

SimilarityFunction = Literal["cosine", "euclidean"]


class FalkorDBDocumentStore:
    """
    A Haystack DocumentStore backed by FalkorDB — a high-performance graph database.

    Optimised for GraphRAG workloads.

    Documents are stored as graph nodes (labelled ``Document`` by default) in a named
    FalkorDB graph.  Document properties, including ``meta`` fields, are stored
    **flat** at the same level as ``id`` and ``content`` — exactly the same layout as
    the ``neo4j-haystack`` reference integration.

    Vector search is performed via FalkorDB's native vector index —
    **no APOC is required**.  All bulk writes use ``UNWIND`` + ``MERGE`` for safe,
    idiomatic OpenCypher upserts.

    Usage example:

    ```python
    from haystack_integrations.document_stores.falkor_db import FalkorDBDocumentStore
    from haystack.dataclasses import Document

    store = FalkorDBDocumentStore(host="localhost", port=6379)
    store.write_documents([
        Document(content="Hello, GraphRAG!", meta={"year": 2024}),
    ])
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
        node_label: str = "Document",
        embedding_dim: int = 768,
        embedding_field: str = "embedding",
        similarity: SimilarityFunction = "cosine",
        write_batch_size: int = 100,
        recreate_index: bool = False,
        verify_connectivity: bool = False,
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
        :param node_label: Label used for document nodes in the graph.
            Defaults to ``"Document"``.
        :param embedding_dim: Dimensionality of the vector embeddings. Used when
            creating the vector index. Defaults to ``768``.
        :param embedding_field: Name of the node property that stores the embedding
            vector. Defaults to ``"embedding"``.
        :param similarity: Similarity function for the vector index.  Accepted values
            are ``"cosine"`` (default) and ``"euclidean"``.
        :param write_batch_size: Number of documents written per ``UNWIND`` batch.
            Defaults to ``100``.
        :param recreate_index: When ``True`` the existing graph (and all its data) is
            dropped and recreated on initialisation. Useful for tests.
            Defaults to ``False``.
        :param verify_connectivity: When ``True`` a connectivity probe is run
            immediately in ``__init__`` — raises if the server is unreachable.
            Defaults to ``False`` (lazy connection).
        :raises ValueError: If ``similarity`` is not ``"cosine"`` or ``"euclidean"``.
        """
        if similarity not in ("cosine", "euclidean"):
            msg = (
                f"Provided similarity '{similarity}' is not supported by FalkorDBDocumentStore. "
                "Please choose one of: 'cosine', 'euclidean'."
            )
            raise ValueError(msg)

        self._host = host
        self._port = port
        self._graph_name = graph_name
        self._username = username
        self._password = password
        self._node_label = node_label
        self._embedding_dim = embedding_dim
        self._embedding_field = embedding_field
        self._similarity: SimilarityFunction = similarity
        self._write_batch_size = write_batch_size
        self._recreate_index = recreate_index
        self._verify_connectivity = verify_connectivity

        # Lazy — populated on first use via _ensure_connected().
        self._client: Any | None = None
        self._graph: Any | None = None
        self._initialized: bool = False

        if verify_connectivity:
            self._ensure_connected()

    # ------------------------------------------------------------------
    # Internal connection helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """
        Lazily open the FalkorDB connection and set up the graph schema.

        Called at the start of every public method so the store remains
        serialisable without an active database connection.
        """
        if self._initialized:
            return

        import falkordb  # type: ignore[import-untyped,import-not-found]  # noqa: PLC0415

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
            except Exception:
                logger.debug("Graph '%s' could not be deleted (may not exist yet).", self._graph_name)

        self._graph = self._client.select_graph(self._graph_name)
        self._ensure_schema()
        self._initialized = True

    @property
    def _g(self) -> Any:
        """Return the active FalkorDB Graph object, guaranteed non-None after _ensure_connected."""
        assert self._graph is not None, "_ensure_connected() must be called first"  # noqa: S101
        return self._graph

    def _ensure_schema(self) -> None:
        """
        Create the property index and vector index if they do not already exist.

        Uses only standard OpenCypher / FalkorDB-native syntax — **no APOC**.
        """
        # Property index on (:node_label {id}) for fast MERGE lookups.
        try:
            self._g.query(f"CREATE INDEX FOR (d:{self._node_label}) ON (d.id)")
        except Exception:
            logger.debug("Property index on %s(id) already exists — skipping creation.", self._node_label)

        # FalkorDB-native vector index syntax (different from Neo4j)
        try:
            cypher = (
                f"CREATE VECTOR INDEX FOR (d:{self._node_label}) "
                f"ON (d.{self._embedding_field}) "
                f"OPTIONS {{dimension: {self._embedding_dim}, similarityFunction: '{self._similarity}'}}"
            )
            self._g.query(cypher)
        except Exception:
            logger.debug(
                "Vector index on %s(%s) already exists — skipping creation.",
                self._node_label,
                self._embedding_field,
            )

    # ------------------------------------------------------------------
    # Haystack DocumentStore protocol
    # ------------------------------------------------------------------

    def count_documents(self) -> int:
        """
        Return the number of documents currently stored in the graph.

        :returns: Integer count of document nodes.
        """
        self._ensure_connected()
        result = self._g.query(f"MATCH (d:{self._node_label}) RETURN count(d) AS n")
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
            cypher = f"MATCH (d:{self._node_label}) WHERE {where_clause} RETURN d"
        else:
            cypher = f"MATCH (d:{self._node_label}) RETURN d"
            params = {}

        result = self._g.query(cypher, params)
        return [_node_to_document(row[0]) for row in result.result_set]

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Write documents to the FalkorDB graph using ``UNWIND`` + ``MERGE`` for batching.

        Document ``meta`` fields are stored **flat** at the same level as ``id`` and
        ``content`` — no prefix is added.  This matches the layout used by the
        ``neo4j-haystack`` reference integration.

        :param documents: List of :class:`haystack.dataclasses.Document` objects.
        :param policy: How to handle documents whose ``id`` already exists.
            Defaults to :attr:`DuplicatePolicy.NONE` (treated as FAIL).
        :raises ValueError: If ``documents`` contains non-Document elements.
        :raises DuplicateDocumentError: If ``policy`` is FAIL / NONE and a duplicate
            ID is encountered.
        :raises DocumentStoreError: If any other DB error occurs.
        :returns: Number of documents written or updated.
        """
        self._ensure_connected()

        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"write_documents() expects a list of Documents but got an element of type {type(doc)}."
                raise ValueError(msg)

        if not documents:
            logger.warning("Calling FalkorDBDocumentStore.write_documents() with an empty list.")
            return 0

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        document_objects = self._handle_duplicate_documents(documents, policy)

        written = 0
        for batch_start in range(0, len(document_objects), self._write_batch_size):
            batch = document_objects[batch_start : batch_start + self._write_batch_size]
            written += self._write_batch(batch, policy)

        return written

    def _handle_duplicate_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy,
    ) -> list[Document]:
        """
        Mirrors neo4j-haystack: drops intra-batch duplicates and for SKIP/FAIL checks existing IDs.

        Checks for IDs that already exist in the database.

        :param documents: All documents to write.
        :param policy: Duplicate handling policy.
        :returns: Filtered list ready for batch writing.
        :raises DuplicateDocumentError: When ``policy`` is FAIL and existing IDs found.
        """
        if policy in (DuplicatePolicy.SKIP, DuplicatePolicy.FAIL):
            # Step 1: deduplicate within the incoming list itself.
            documents = self._drop_duplicate_documents(documents)

            # Step 2: find which IDs already exist in the DB.
            ids = [doc.id for doc in documents]
            existing = self._g.query(
                f"UNWIND $ids AS id MATCH (d:{self._node_label} {{id: id}}) RETURN d.id",
                {"ids": ids},
            )
            ids_exist_in_db: list[str] = [row[0] for row in existing.result_set]

            if ids_exist_in_db and policy == DuplicatePolicy.FAIL:
                msg = f"Document with ids '{', '.join(ids_exist_in_db)}' already exists in graph '{self._graph_name}'."
                raise DuplicateDocumentError(msg)

            # For SKIP: remove those that already exist.
            if ids_exist_in_db:
                existing_set = set(ids_exist_in_db)
                documents = [d for d in documents if d.id not in existing_set]

        return documents

    def _drop_duplicate_documents(self, documents: list[Document]) -> list[Document]:
        """
        Drop duplicate documents (by ID) within the provided list.

        :param documents: Input list — may contain repeated IDs.
        :returns: Deduplicated list preserving first-occurrence order.
        """
        seen_ids: set[str] = set()
        unique: list[Document] = []
        for doc in documents:
            if doc.id in seen_ids:
                logger.info(
                    "Duplicate Documents: Document with id '%s' already present in the batch — skipping.",
                    doc.id,
                )
                continue
            unique.append(doc)
            seen_ids.add(doc.id)
        return unique

    def _write_batch(self, documents: list[Document], policy: DuplicatePolicy) -> int:
        """
        Write a single batch of documents using a single UNWIND query.

        By the time this is called, duplicate handling has already been performed by
        :meth:`_handle_duplicate_documents`.

        :param documents: Batch of Documents (≤ ``write_batch_size``).
        :param policy: Duplicate policy — only OVERWRITE needs a different Cypher template.
        :returns: Number of nodes created or updated.
        """
        records = [_document_to_falkordb_record(doc) for doc in documents]

        if policy == DuplicatePolicy.OVERWRITE:
            # ON MATCH SET applies the full map (including updated fields).
            cypher = f"""
UNWIND $docs AS doc
MERGE (d:{self._node_label} {{id: doc.id}})
ON CREATE SET d += doc
ON MATCH SET d += doc
RETURN count(d) AS n
"""
        else:
            # FAIL already filtered duplicates above; SKIP excluded them.
            # In both remaining cases we only write truly-new nodes.
            cypher = f"""
UNWIND $docs AS doc
MERGE (d:{self._node_label} {{id: doc.id}})
ON CREATE SET d += doc
RETURN count(d) AS n
"""

        try:
            result = self._g.query(cypher, {"docs": records})
            rows = result.result_set
            return int(rows[0][0]) if rows else len(documents)
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
        self._g.query(
            f"UNWIND $ids AS id MATCH (d:{self._node_label} {{id: id}}) DETACH DELETE d",
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
        scale_score: bool = True,
    ) -> list[Document]:
        """
        Retrieve documents by vector similarity using FalkorDB's native vector index.

        Uses ``CALL db.idx.vector.queryNodes`` — FalkorDB's OpenCypher extension for
        ANN search.  **No APOC is required.**

        Cosine scores are returned in ``[-1, 1]``; when ``scale_score=True`` they are
        scaled to ``[0, 1]`` using the same formula as neo4j-haystack:
        ``(score + 1) / 2``.  Euclidean scores are transformed with a sigmoid.

        :param query_embedding: Query vector as a plain Python list of floats.
        :param top_k: Maximum number of results to return.
        :param filters: Optional Haystack filters applied as a ``WHERE`` predicate
            on the vector search result set (post-filter).
        :param scale_score: Whether to scale the raw similarity score to ``[0, 1]``.
        :returns: List of :class:`Document` objects ordered by similarity (best first).
        """
        self._ensure_connected()

        if filters:
            where_clause, filter_params = _convert_filters(filters)
            cypher = f"""
CALL db.idx.vector.queryNodes('{self._node_label}', '{self._embedding_field}', $top_k, vecf32($query_embedding))
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
            cypher = f"""
CALL db.idx.vector.queryNodes('{self._node_label}', '{self._embedding_field}', $top_k, vecf32($query_embedding))
YIELD node AS d, score
RETURN d, score
ORDER BY score ASC
"""
            params = {"top_k": top_k, "query_embedding": query_embedding}

        result = self._g.query(cypher, params)
        documents = []
        for row in result.result_set:
            node, score = row[0], row[1]
            doc = _node_to_document(node)
            final_score = self._scale_to_unit_interval(float(score)) if scale_score else float(score)
            doc = replace(doc, score=final_score)
            documents.append(doc)
        return documents

    def _cypher_retrieval(
        self,
        cypher_query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Execute an arbitrary OpenCypher query and map the results to Documents.

        The first element of each result row is converted to a
        :class:`haystack.dataclasses.Document`.

        :param cypher_query: A valid OpenCypher query string.
        :param parameters: Optional query parameters (``$param`` placeholders).
        :returns: List of :class:`Document` objects built from the query results.
        :raises DocumentStoreError: If the query fails.
        """
        self._ensure_connected()
        try:
            result = self._g.query(cypher_query, parameters or {})
            return [_node_to_document(row[0]) for row in result.result_set]
        except Exception as exc:
            msg = f"Cypher query failed: {exc}"
            raise DocumentStoreError(msg) from exc

    def _scale_to_unit_interval(self, score: float) -> float:
        """
        Scale a raw similarity score to the unit interval ``[0, 1]``.

        Mirrors the scaling formula used in neo4j-haystack:
        - Cosine: ``(score + 1) / 2``
        - Euclidean: sigmoid ``1 / (1 + exp(-score / 100))``

        :param score: Raw score returned by the vector index.
        :returns: Scaled score in ``[0, 1]``.
        """
        if self._similarity == "cosine":
            return (score + 1) / 2
        return float(1 / (1 + np.exp(-score / 100)))

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
            node_label=self._node_label,
            embedding_dim=self._embedding_dim,
            embedding_field=self._embedding_field,
            similarity=self._similarity,
            write_batch_size=self._write_batch_size,
            recreate_index=self._recreate_index,
            verify_connectivity=self._verify_connectivity,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FalkorDBDocumentStore:
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


def _document_to_falkordb_record(doc: Document) -> dict[str, Any]:
    """
    Convert a Haystack Document to a flat dict for storage as FalkorDB node properties.

    Mirrors the ``_DefaultDocumentMarshaller`` from neo4j-haystack:

    - ``meta`` fields are stored **at the same level** as ``id`` and ``content``
      (i.e. **no** ``meta_`` prefix).
    - The ``score`` field is excluded — it is a retrieval-time value, not stored data.
    - ``Document.to_dict(flatten=True)`` is used so nested ``meta`` dicts are
      also flattened (e.g. ``{"meta": {"a": {"b": 1}}}`` → ``{"a.b": 1}``).

    :param doc: The document to convert.
    :returns: Flat dictionary of node properties.
    """
    doc_dict = doc.to_dict(flatten=True)

    # Remove runtime-only / non-serialisable fields.
    doc_dict.pop("score", None)
    doc_dict.pop("sparse_embedding", None)

    # Filter out None values — FalkorDB nodes don't need null properties stored.
    return {k: v for k, v in doc_dict.items() if v is not None}


def _node_to_document(node: Any) -> Document:
    """
    Convert a FalkorDB graph node back to a Haystack Document.

    Because properties are stored flat (see :func:`_document_to_falkordb_record`),
    reconstruction uses ``Document.from_dict`` directly — the same approach taken
    by neo4j-haystack's ``_neo4j_record_to_document``.

    :param node: A FalkorDB ``Node`` object or a plain ``dict``.
    :returns: Reconstructed :class:`haystack.dataclasses.Document`.
    """
    if hasattr(node, "properties"):
        record: dict[str, Any] = dict(node.properties)
    elif isinstance(node, dict):
        record = node
    else:
        record = {}

    return Document.from_dict(record)


def _convert_filters(filters: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Translate a Haystack filter dict into an OpenCypher ``WHERE`` sub-expression.

    Supports the full Haystack filter DSL:

    - Logical: ``AND``, ``OR``, ``NOT``
    - Comparison: ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``
    - Membership: ``in``, ``not in``

    All values are passed as named query parameters to prevent injection.

    :param filters: A Haystack filter dictionary.
    :returns: Tuple of ``(where_clause_string, params_dict)``.
    :raises ValueError: If an unsupported operator or malformed filter is provided.
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

    # Because meta fields are stored flat (no prefix), all fields map to d.<field>.
    # Top-level Document fields (id, content, embedding) are among them too.
    cypher_field = f"d.{field}"

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
