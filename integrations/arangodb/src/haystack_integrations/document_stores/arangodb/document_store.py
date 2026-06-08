# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any, Literal, cast

from arango import ArangoClient
from arango.collection import StandardCollection
from arango.cursor import Cursor
from arango.database import StandardDatabase
from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from .filters import _convert_filters

logger = logging.getLogger(__name__)

# Maps the configured similarity function to the AQL vector function, its sort order,
# and the metric name used when creating the vector index.
_SIMILARITY_AQL: dict[str, tuple[str, str, str]] = {
    "cosine": ("APPROX_NEAR_COSINE", "DESC", "cosine"),
    "dot_product": ("APPROX_NEAR_INNER_PRODUCT", "DESC", "innerProduct"),
    "l2": ("APPROX_NEAR_L2", "ASC", "l2"),
}

_VECTOR_INDEX_NAME = "haystack_vector_index"


def _doc_to_arango(doc: Document) -> dict[str, Any]:
    d = doc.to_dict(flatten=False)
    d["_key"] = d.pop("id")
    return d


def _arango_to_doc(record: dict[str, Any]) -> Document:
    record = dict(record)
    record["id"] = record.pop("_key")
    record.pop("_id", None)
    record.pop("_rev", None)
    return Document.from_dict(record)


class ArangoDocumentStore:
    """
    A Haystack DocumentStore backed by [ArangoDB](https://www.arangodb.com/).

    Documents are stored in an ArangoDB collection and support vector similarity search
    via AQL vector functions (requires ArangoDB 3.12+).

    Example usage:

    ```python
    from haystack_integrations.document_stores.arangodb import ArangoDocumentStore
    from haystack.utils import Secret

    store = ArangoDocumentStore(
        host="http://localhost:8529",
        database="haystack",
        username=Secret.from_env_var("ARANGO_USERNAME", strict=False),
        password=Secret.from_env_var("ARANGO_PASSWORD"),
        collection_name="documents",
        embedding_dimension=768,
    )
    ```
    """

    def __init__(
        self,
        *,
        host: str = "http://localhost:8529",
        database: str = "haystack",
        username: Secret = Secret.from_env_var("ARANGO_USERNAME", strict=False),
        password: Secret = Secret.from_env_var("ARANGO_PASSWORD"),
        collection_name: str = "haystack_documents",
        embedding_dimension: int = 768,
        recreate_collection: bool = False,
        similarity_function: Literal["cosine", "dot_product", "l2"] = "cosine",
    ) -> None:
        """
        Creates a new ArangoDocumentStore instance.

        :param host: ArangoDB server URL, e.g. `http://localhost:8529`.
        :param database: Name of the ArangoDB database to use. Created if it does not exist.
        :param username: ArangoDB username as a `Secret`. Defaults to `ARANGO_USERNAME` env var,
            falling back to `root` if the variable is not set.
        :param password: ArangoDB password as a `Secret`. Defaults to `ARANGO_PASSWORD` env var.
        :param collection_name: Name of the collection to store documents in.
        :param embedding_dimension: Dimensionality of document embeddings.
        :param recreate_collection: If `True`, drop and recreate the collection on startup.
        :param similarity_function: Vector similarity function to use for embedding retrieval.
            One of `"cosine"` (default), `"dot_product"`, or `"l2"`.
        """
        self.host = host
        self.database = database
        self.username = username
        self.password = password
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.recreate_collection = recreate_collection
        self.similarity_function = similarity_function
        self._db: StandardDatabase | None = None
        self._col: StandardCollection | None = None

    def _ensure_connected(self) -> None:
        if self._db is not None and self._col is not None:
            return
        username = self.username.resolve_value() or "root"
        pwd = self.password.resolve_value() or ""
        client = ArangoClient(hosts=self.host)
        sys_db = client.db("_system", username=username, password=pwd)
        if not sys_db.has_database(self.database):
            sys_db.create_database(self.database)
        db = client.db(self.database, username=username, password=pwd)
        if self.recreate_collection and db.has_collection(self.collection_name):
            db.delete_collection(self.collection_name)
        if not db.has_collection(self.collection_name):
            db.create_collection(self.collection_name)
        self._db = db
        self._col = db.collection(self.collection_name)

    def _ensure_vector_index(self) -> None:
        """
        Lazily creates the FAISS vector index used by the `APPROX_NEAR_*` functions.

        The index is created on first retrieval rather than at write time because ArangoDB
        requires the collection to already contain training documents with non-null embeddings,
        and `nLists` must be `<=` the number of such documents. We use `nLists=1`, which
        probes the single Voronoi cell (a full scan) and therefore returns exact, complete results.
        """
        col = cast(StandardCollection, self._col)
        existing = cast("list[dict[str, Any]]", col.indexes())
        if any(index.get("type") == "vector" for index in existing):
            return
        _, _, metric = _SIMILARITY_AQL[self.similarity_function]
        col.add_index(
            {
                "type": "vector",
                "name": _VECTOR_INDEX_NAME,
                "fields": ["embedding"],
                "params": {
                    "metric": metric,
                    "dimension": self.embedding_dimension,
                    "nLists": 1,
                },
            }
        )

    def count_documents(self) -> int:
        """
        Returns the number of documents in the store.

        :returns: Document count.
        """
        self._ensure_connected()
        return cast(int, cast(StandardCollection, self._col).count())

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns documents matching the provided filters.

        :param filters: Haystack metadata filters. If `None`, all documents are returned.
        :returns: List of matching `Document` objects.
        """
        self._ensure_connected()
        db = cast(StandardDatabase, self._db)

        aql = f"FOR doc IN {self.collection_name}"
        bind_vars: dict[str, Any] = {}

        if filters:
            expr, bind_vars = _convert_filters(filters)
            aql += f" FILTER {expr}"

        aql += " RETURN doc"

        cursor = cast(Cursor, db.aql.execute(aql, bind_vars=bind_vars))
        return [_arango_to_doc(r) for r in cursor]

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents to the store.

        :param documents: Documents to write.
        :param policy: How to handle duplicates — `OVERWRITE`, `SKIP`, or `FAIL` (default).
        :raises ValueError: If `documents` contains non-`Document` objects.
        :raises DuplicateDocumentError: If a duplicate is found and policy is `FAIL`.
        :returns: Number of documents written.
        """
        if not documents:
            return 0
        if not isinstance(documents[0], Document):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        self._ensure_connected()
        col = cast(StandardCollection, self._col)

        arango_docs = [_doc_to_arango(doc) for doc in documents]
        overwrite = policy == DuplicatePolicy.OVERWRITE
        raw = col.insert_many(arango_docs, overwrite=overwrite, silent=False)
        results = raw if isinstance(raw, list) else []

        written = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if policy == DuplicatePolicy.FAIL:
                    msg = f"Document with id '{documents[i].id}' already exists."
                    raise DuplicateDocumentError(msg)
                # SKIP: just move on
            else:
                written += 1

        return written

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes documents by their IDs.

        :param document_ids: List of document IDs to delete.
        """
        if not document_ids:
            return
        self._ensure_connected()
        col = cast(StandardCollection, self._col)
        col.delete_many([{"_key": doc_id} for doc_id in document_ids])

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieves documents most similar to the query embedding using cosine similarity.

        This method is used internally by `ArangoEmbeddingRetriever`.

        :param query_embedding: The query vector.
        :param top_k: Number of top results to return.
        :param filters: Optional metadata filters.
        :returns: List of `Document` objects sorted by descending similarity score.
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        self._ensure_connected()
        db = cast(StandardDatabase, self._db)
        col = cast(StandardCollection, self._col)
        doc_count = col.count()
        if doc_count == 0:
            return []

        self._ensure_vector_index()

        aql_func, sort_order, _ = _SIMILARITY_AQL[self.similarity_function]
        bind_vars: dict[str, Any] = {"query_vec": query_embedding, "top_k": top_k}

        # ArangoDB only uses the vector index when the `APPROX_NEAR_*` call is followed
        # directly by `SORT` + `LIMIT` with no preceding `FILTER`. Metadata filters are
        # therefore applied *after* the vector search.
        if filters:
            expr, fvars = _convert_filters(filters)
            bind_vars.update(fvars)
            bind_vars["candidates"] = doc_count
            aql = f"""
            FOR doc IN (
                FOR d IN {self.collection_name}
                    LET score = {aql_func}(d.embedding, @query_vec)
                    SORT score {sort_order}
                    LIMIT @candidates
                    RETURN MERGE(d, {{score: score}})
            )
                FILTER {expr}
                LIMIT @top_k
                RETURN doc
            """
        else:
            aql = f"""
            FOR doc IN {self.collection_name}
                LET score = {aql_func}(doc.embedding, @query_vec)
                SORT score {sort_order}
                LIMIT @top_k
                RETURN MERGE(doc, {{score: score}})
            """

        cursor = cast(Cursor, db.aql.execute(aql, bind_vars=bind_vars))
        docs = []
        for record in cursor:
            score = record.pop("score", None)
            d = _arango_to_doc(record)
            if score is not None:
                d = dataclasses.replace(d, score=score)
            docs.append(d)
        return docs

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            host=self.host,
            database=self.database,
            username=self.username.to_dict(),
            password=self.password.to_dict(),
            collection_name=self.collection_name,
            embedding_dimension=self.embedding_dimension,
            recreate_collection=self.recreate_collection,
            similarity_function=self.similarity_function,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArangoDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], ["password", "username"])
        return default_from_dict(cls, data)
