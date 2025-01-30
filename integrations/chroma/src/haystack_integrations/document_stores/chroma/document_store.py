# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Literal, Optional

import chromadb
from chromadb.api.types import GetResult, QueryResult
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy

from .filters import _convert_filters
from .utils import get_embedding_function

logger = logging.getLogger(__name__)


VALID_DISTANCE_FUNCTIONS = "l2", "cosine", "ip"
SUPPORTED_TYPES_FOR_METADATA_VALUES = str, int, float, bool


class ChromaDocumentStore:
    """
    A document store using [Chroma](https://docs.trychroma.com/) as the backend.

    We use the `collection.get` API to implement the document store protocol,
    the `collection.search` API will be used in the retriever instead.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_function: str = "default",
        persist_path: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        distance_function: Literal["l2", "cosine", "ip"] = "l2",
        metadata: Optional[dict] = None,
        **embedding_function_params,
    ):
        """
        Creates a new ChromaDocumentStore instance.
        It is meant to be connected to a Chroma collection.

        Note: for the component to be part of a serializable pipeline, the __init__
        parameters must be serializable, reason why we use a registry to configure the
        embedding function passing a string.

        :param collection_name: the name of the collection to use in the database.
        :param embedding_function: the name of the embedding function to use to embed the query
        :param persist_path: Path for local persistent storage. Cannot be used in combination with `host` and `port`.
            If none of `persist_path`, `host`, and `port` is specified, the database will be `in-memory`.
        :param host: The host address for the remote Chroma HTTP client connection. Cannot be used with `persist_path`.
        :param port: The port number for the remote Chroma HTTP client connection. Cannot be used with `persist_path`.
        :param distance_function: The distance metric for the embedding space.
            - `"l2"` computes the Euclidean (straight-line) distance between vectors,
            where smaller scores indicate more similarity.
            - `"cosine"` computes the cosine similarity between vectors,
            with higher scores indicating greater similarity.
            - `"ip"` stands for inner product, where higher scores indicate greater similarity between vectors.
            **Note**: `distance_function` can only be set during the creation of a collection.
            To change the distance metric of an existing collection, consider cloning the collection.
        :param metadata: a dictionary of chromadb collection parameters passed directly to chromadb's client
            method `create_collection`. If it contains the key `"hnsw:space"`, the value will take precedence over the
            `distance_function` parameter above.
        :param embedding_function_params: additional parameters to pass to the embedding function.
        """

        if distance_function not in VALID_DISTANCE_FUNCTIONS:
            error_message = (
                f"Invalid distance_function: '{distance_function}' for the collection. "
                f"Valid options are: {VALID_DISTANCE_FUNCTIONS}."
            )
            raise ValueError(error_message)

        # Store the params for marshalling
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._embedding_func = get_embedding_function(embedding_function, **embedding_function_params)
        self._embedding_function_params = embedding_function_params
        self._distance_function = distance_function
        self._metadata = metadata
        self._collection = None

        self._persist_path = persist_path
        self._host = host
        self._port = port

        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            # Create the client instance
            if self._persist_path and (self._host or self._port is not None):
                error_message = (
                    "You must specify `persist_path` for local persistent storage or, "
                    "alternatively, `host` and `port` for remote HTTP client connection. "
                    "You cannot specify both options."
                )
                raise ValueError(error_message)
            if self._host and self._port is not None:
                # Remote connection via HTTP client
                client = chromadb.HttpClient(
                    host=self._host,
                    port=self._port,
                )
            elif self._persist_path is None:
                # In-memory storage
                client = chromadb.Client()
            else:
                # Local persistent storage
                client = chromadb.PersistentClient(path=self._persist_path)

            self._metadata = self._metadata or {}
            if "hnsw:space" not in self._metadata:
                self._metadata["hnsw:space"] = self._distance_function

            if self._collection_name in client.list_collections():
                self._collection = client.get_collection(self._collection_name, embedding_function=self._embedding_func)

                if self._metadata != self._collection.metadata:
                    logger.warning(
                        "Collection already exists. "
                        "The `distance_function` and `metadata` parameters will be ignored."
                    )
            else:
                self._collection = client.create_collection(
                    name=self._collection_name,
                    metadata=self._metadata,
                    embedding_function=self._embedding_func,
                )

            self._initialized = True

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns: how many documents are present in the document store.
        """
        self._ensure_initialized()
        assert self._collection is not None
        return self._collection.count()

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
         Returns the documents that match the filters provided.

         Filters can be provided as a dictionary supporting filtering by ids, metadata, and document content.
         Metadata filters should use the `"meta.<metadata_key>"` syntax, while content-based filters
         use the `"content"` field directly.
         Content filters support the `contains` and `not contains` operators,
         while id filters only support the `==` operator.

         Due to Chroma's distinction between metadata filters and document filters, filters with `"field": "content"`
        (i.e., document content filters) and metadata fields must be supplied separately. For details on chroma filters,
        see the [Chroma documentation](https://docs.trychroma.com/guides).

         Example:

         ```python
         filter_1 = {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.name", "operator": "==", "value": "name_0"},
                    {"field": "meta.number", "operator": "not in", "value": [2, 9]},
                ],
            }
         filter_2 = {
                "operator": "AND",
                "conditions": [
                    {"field": "content", "operator": "contains", "value": "FOO"},
                    {"field": "content", "operator": "not contains", "value": "BAR"},
                ],
            }
         ```

        If you need to apply the same logical operator (e.g., "AND", "OR") to multiple conditions at the same level,
         you can provide a list of dictionaries as the value for the operator, like in the example below:

         ```python
         filters = {
             "operator": "OR",
             "conditions": [
                 {"field": "meta.author", "operator": "==", "value": "author_1"},
                 {
                     "operator": "AND",
                     "conditions": [
                         {"field": "meta.tag", "operator": "==", "value": "tag_1"},
                         {"field": "meta.page", "operator": ">", "value": 100},
                     ],
                 },
                 {
                     "operator": "AND",
                     "conditions": [
                         {"field": "meta.tag", "operator": "==", "value": "tag_2"},
                         {"field": "meta.page", "operator": ">", "value": 200},
                     ],
                 },
             ],
         }
         ```

         :param filters: the filters to apply to the document list.
         :returns: a list of Documents that match the given filters.
        """
        self._ensure_initialized()
        assert self._collection is not None

        if filters:
            chroma_filter = _convert_filters(filters)
            kwargs: Dict[str, Any] = {"where": chroma_filter.where}

            if chroma_filter.ids:
                kwargs["ids"] = chroma_filter.ids
            if chroma_filter.where_document:
                kwargs["where_document"] = chroma_filter.where_document

            result = self._collection.get(**kwargs)
        else:
            result = self._collection.get()

        return self._get_result_to_documents(result)

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int:
        """
        Writes (or overwrites) documents into the store.

        :param documents:
            A list of documents to write into the document store.
        :param policy:
            Not supported at the moment.

        :raises ValueError:
            When input is not valid.

        :returns:
            The number of documents written
        """
        self._ensure_initialized()
        assert self._collection is not None

        for doc in documents:
            if not isinstance(doc, Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

            if doc.content is None:
                logger.warning(
                    "ChromaDocumentStore cannot store documents with `content=None`. "
                    "`array`, `dataframe` and `blob` are not supported. "
                    "Document with id %s will be skipped.",
                    doc.id,
                )
                continue
            data = {"ids": [doc.id], "documents": [doc.content]}

            if doc.meta:
                valid_meta = {}
                discarded_keys = []

                for k, v in doc.meta.items():
                    if isinstance(v, SUPPORTED_TYPES_FOR_METADATA_VALUES):
                        valid_meta[k] = v
                    else:
                        discarded_keys.append(k)

                if discarded_keys:
                    logger.warning(
                        "Document %s contains `meta` values of unsupported types for the keys: %s. "
                        "These items will be discarded. Supported types are: %s.",
                        doc.id,
                        ", ".join(discarded_keys),
                        ", ".join([t.__name__ for t in SUPPORTED_TYPES_FOR_METADATA_VALUES]),
                    )

                if valid_meta:
                    data["metadatas"] = [valid_meta]

            if doc.embedding is not None:
                data["embeddings"] = [doc.embedding]

            if hasattr(doc, "sparse_embedding") and doc.sparse_embedding is not None:
                logger.warning(
                    "Document %s has the `sparse_embedding` field set,"
                    "but storing sparse embeddings in Chroma is not currently supported."
                    "The `sparse_embedding` field will be ignored.",
                    doc.id,
                )

            self._collection.add(**data)

        return len(documents)

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.

        :param document_ids: the document ids to delete
        """
        self._ensure_initialized()
        assert self._collection is not None

        self._collection.delete(ids=document_ids)

    def search(self, queries: List[str], top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[List[Document]]:
        """Search the documents in the store using the provided text queries.

        :param queries: the list of queries to search for.
        :param top_k: top_k documents to return for each query.
        :param filters: a dictionary of filters to apply to the search. Accepts filters in haystack format.
        :returns: matching documents for each query.
        """
        self._ensure_initialized()
        assert self._collection is not None

        if not filters:
            results = self._collection.query(
                query_texts=queries,
                n_results=top_k,
                include=["embeddings", "documents", "metadatas", "distances"],
            )
        else:
            chroma_filters = _convert_filters(filters=filters)
            results = self._collection.query(
                query_texts=queries,
                n_results=top_k,
                where=chroma_filters.where,
                where_document=chroma_filters.where_document,
                include=["embeddings", "documents", "metadatas", "distances"],
            )

        return self._query_result_to_documents(results)

    def search_embeddings(
        self, query_embeddings: List[List[float]], top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[List[Document]]:
        """
        Perform vector search on the stored document, pass the embeddings of the queries instead of their text.

        :param query_embeddings: a list of embeddings to use as queries.
        :param top_k: the maximum number of documents to retrieve.
        :param filters: a dictionary of filters to apply to the search. Accepts filters in haystack format.

        :returns: a list of lists of documents that match the given filters.

        """
        self._ensure_initialized()
        assert self._collection is not None

        if not filters:
            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                include=["embeddings", "documents", "metadatas", "distances"],
            )
        else:
            chroma_filters = _convert_filters(filters=filters)
            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                where=chroma_filters.where,
                where_document=chroma_filters.where_document,
                include=["embeddings", "documents", "metadatas", "distances"],
            )

        return self._query_result_to_documents(results)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChromaDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            collection_name=self._collection_name,
            embedding_function=self._embedding_function,
            persist_path=self._persist_path,
            host=self._host,
            port=self._port,
            distance_function=self._distance_function,
            **self._embedding_function_params,
        )

    @staticmethod
    def _get_result_to_documents(result: GetResult) -> List[Document]:
        """
        Helper function to convert Chroma results into Haystack Documents
        """
        retval = []
        for i in range(len(result.get("documents", []))):
            document_dict: Dict[str, Any] = {"id": result["ids"][i]}

            result_documents = result.get("documents")
            if result_documents:
                document_dict["content"] = result_documents[i]

            result_metadata = result.get("metadatas")
            # Ensure metadata[i] is not None or don't add it to the document dict
            if result_metadata and result_metadata[i]:
                document_dict["meta"] = result_metadata[i]

            result_embeddings = result.get("embeddings")
            if result_embeddings:
                document_dict["embedding"] = list(result_embeddings[i])

            retval.append(Document.from_dict(document_dict))

        return retval

    @staticmethod
    def _query_result_to_documents(result: QueryResult) -> List[List[Document]]:
        """
        Helper function to convert Chroma results into Haystack Documents
        """
        retval: List[List[Document]] = []
        documents = result.get("documents")
        if documents is None:
            return retval

        for i, answers in enumerate(documents):
            converted_answers = []
            for j in range(len(answers)):
                document_dict: Dict[str, Any] = {
                    "id": result["ids"][i][j],
                    "content": answers[j],
                }

                # prepare metadata
                metadatas = result.get("metadatas")
                try:
                    if metadatas and metadatas[i][j] is not None:
                        document_dict["meta"] = metadatas[i][j]
                except IndexError:
                    pass

                if embeddings := result.get("embeddings"):
                    document_dict["embedding"] = embeddings[i][j]

                if distances := result.get("distances"):
                    document_dict["score"] = distances[i][j]

                converted_answers.append(Document.from_dict(document_dict))
            retval.append(converted_answers)

        return retval
