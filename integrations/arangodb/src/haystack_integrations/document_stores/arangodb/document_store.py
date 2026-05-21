# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any, cast

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


def _doc_to_arango(doc: Document) -> dict[str, Any]:
    d = doc.to_dict()
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
    via AQL's `COSINE_SIMILARITY` function (requires ArangoDB 3.12+).

    Example usage:

    ```python
    from haystack_integrations.document_stores.arangodb import ArangoDocumentStore
    from haystack.utils import Secret

    store = ArangoDocumentStore(
        host="http://localhost:8529",
        database="haystack",
        username="root",
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
        username: str = "root",
        password: Secret = Secret.from_env_var("ARANGO_PASSWORD"),
        collection_name: str = "haystack_documents",
        embedding_dimension: int = 768,
        recreate_collection: bool = False,
    ) -> None:
        """
        Creates a new ArangoDocumentStore instance.

        :param host: ArangoDB server URL, e.g. `http://localhost:8529`.
        :param database: Name of the ArangoDB database to use. Created if it does not exist.
        :param username: ArangoDB username.
        :param password: ArangoDB password as a `Secret`. Defaults to `ARANGO_PASSWORD` env var.
        :param collection_name: Name of the collection to store documents in.
        :param embedding_dimension: Dimensionality of document embeddings.
        :param recreate_collection: If `True`, drop and recreate the collection on startup.
        """
        self.host = host
        self.database = database
        self.username = username
        self.password = password
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.recreate_collection = recreate_collection
        self._db: StandardDatabase | None = None
        self._col: StandardCollection | None = None

    def _ensure_connected(self) -> None:
        if self._db is not None and self._col is not None:
            return
        pwd = self.password.resolve_value() or ""
        client = ArangoClient(hosts=self.host)
        sys_db = client.db("_system", username=self.username, password=pwd)
        if not sys_db.has_database(self.database):
            sys_db.create_database(self.database)
        db = client.db(self.database, username=self.username, password=pwd)
        if self.recreate_collection and db.has_collection(self.collection_name):
            db.delete_collection(self.collection_name)
        if not db.has_collection(self.collection_name):
            db.create_collection(self.collection_name)
        self._db = db
        self._col = db.collection(self.collection_name)

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

        written = 0
        for doc in documents:
            arango_doc = _doc_to_arango(doc)
            key = arango_doc["_key"]
            exists = col.has(key)

            if exists:
                if policy == DuplicatePolicy.FAIL:
                    msg = f"Document with id '{doc.id}' already exists."
                    raise DuplicateDocumentError(msg)
                if policy == DuplicatePolicy.SKIP:
                    continue
                if policy == DuplicatePolicy.OVERWRITE:
                    col.replace(arango_doc)
                    written += 1
                    continue
            else:
                col.insert(arango_doc)
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
        for doc_id in document_ids:
            if col.has(doc_id):
                col.delete(doc_id)

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

        filter_clause = ""
        bind_vars: dict[str, Any] = {"query_vec": query_embedding, "top_k": top_k}

        if filters:
            expr, fvars = _convert_filters(filters)
            filter_clause = f"FILTER {expr} "
            bind_vars.update(fvars)

        aql = f"""
        FOR doc IN {self.collection_name}
            FILTER doc.embedding != null
            {filter_clause}
            LET score = COSINE_SIMILARITY(doc.embedding, @query_vec)
            SORT score DESC
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
            username=self.username,
            password=self.password.to_dict(),
            collection_name=self.collection_name,
            embedding_dimension=self.embedding_dimension,
            recreate_collection=self.recreate_collection,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArangoDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], ["password"])
        return default_from_dict(cls, data)
