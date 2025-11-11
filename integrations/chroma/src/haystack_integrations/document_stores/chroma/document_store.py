# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any, Literal, Optional, cast

import chromadb
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.types import GetResult, QueryResult
from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError
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
        **embedding_function_params: Any,
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

        self._persist_path = persist_path
        self._host = host
        self._port = port

        self._collection: Optional[chromadb.Collection] = None
        self._async_collection: Optional[AsyncCollection] = None

    def _ensure_initialized(self):
        if not self._collection:
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

            self._client = client  # store client for potential future use

            self._metadata = self._metadata or {}
            if "hnsw:space" not in self._metadata:
                self._metadata["hnsw:space"] = self._distance_function

            existing_collection_names = [c.name for c in client.list_collections()]
            if self._collection_name in existing_collection_names:
                self._collection = client.get_collection(
                    self._collection_name,
                    embedding_function=self._embedding_func,
                )

                if self._metadata != self._collection.metadata:
                    logger.warning(
                        "Collection already exists. The `distance_function` and `metadata` parameters will be ignored."
                    )
            else:
                self._collection = client.create_collection(
                    name=self._collection_name,
                    metadata=self._metadata,
                    embedding_function=self._embedding_func,
                )

    async def _ensure_initialized_async(self):
        if not self._async_collection:
            if self._host is None or self._port is None:
                error_message = (
                    "Async support in Chroma is only available for remote connections. "
                    "You must specify `host` and `port` for remote asynchronous HTTP client connection. "
                )
                raise ValueError(error_message)

            client = await chromadb.AsyncHttpClient(
                host=self._host,
                port=self._port,
            )

            self._async_client = client  # store client for potential future use

            self._metadata = self._metadata or {}
            if "hnsw:space" not in self._metadata:
                self._metadata["hnsw:space"] = self._distance_function

            collection = await client.list_collections()
            existing_collection_names = [c.name for c in collection]
            if self._collection_name in existing_collection_names:
                self._async_collection = await client.get_collection(
                    self._collection_name,
                    embedding_function=self._embedding_func,
                )

                if self._metadata != self._async_collection.metadata:
                    logger.warning(
                        "Collection already exists. The `distance_function` and `metadata` parameters will be ignored."
                    )
            else:
                self._async_collection = await client.create_collection(
                    name=self._collection_name,
                    metadata=self._metadata,
                    embedding_function=self._embedding_func,
                )

    def _prepare_get_kwargs(self, filters: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Prepare kwargs for Chroma get operations.
        """
        kwargs: dict[str, Any] = {"include": ["embeddings", "documents", "metadatas"]}

        if filters:
            chroma_filter = _convert_filters(filters)
            kwargs["where"] = chroma_filter.where

            if chroma_filter.ids:
                kwargs["ids"] = chroma_filter.ids
            if chroma_filter.where_document:
                kwargs["where_document"] = chroma_filter.where_document

        return kwargs

    def _prepare_query_kwargs(self, filters: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        Prepare kwargs for Chroma query operations.
        """
        if not filters:
            return {"include": ["embeddings", "documents", "metadatas", "distances"]}

        chroma_filters = _convert_filters(filters=filters)
        return {
            "where": chroma_filters.where,
            "where_document": chroma_filters.where_document,
            "include": ["embeddings", "documents", "metadatas", "distances"],
        }

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns: how many documents are present in the document store.
        """
        self._ensure_initialized()
        assert self._collection is not None
        return self._collection.count()

    async def count_documents_async(self) -> int:
        """
        Asynchronously returns how many documents are present in the document store.

        Asynchronous methods are only supported for HTTP connections.

        :returns: how many documents are present in the document store.
        """
        await self._ensure_initialized_async()
        assert self._async_collection is not None
        value = await self._async_collection.count()

        return value

    def filter_documents(self, filters: Optional[dict[str, Any]] = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

        :param filters: the filters to apply to the document list.
        :returns: a list of Documents that match the given filters.
        """
        self._ensure_initialized()
        assert self._collection is not None

        kwargs = self._prepare_get_kwargs(filters)
        result = self._collection.get(**kwargs)

        return self._get_result_to_documents(result)

    async def filter_documents_async(self, filters: Optional[dict[str, Any]] = None) -> list[Document]:
        """
        Asynchronously returns the documents that match the filters provided.

        Asynchronous methods are only supported for HTTP connections.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

        :param filters: the filters to apply to the document list.
        :returns: a list of Documents that match the given filters.
        """
        await self._ensure_initialized_async()
        assert self._async_collection is not None

        kwargs = self._prepare_get_kwargs(filters)
        result = await self._async_collection.get(**kwargs)

        return self._get_result_to_documents(result)

    def _convert_document_to_chroma(self, doc: Document) -> Optional[dict[str, Any]]:
        """
        Converts a Haystack Document to a Chroma document.
        """
        if not isinstance(doc, Document):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

        if doc.content is None:
            logger.warning(
                "ChromaDocumentStore cannot store documents with `content=None`. "
                "Document with id {doc_id} will be skipped.",
                doc_id=doc.id,
            )
            return None
        elif hasattr(doc, "blob") and doc.blob is not None:
            logger.warning(
                "Document with id {doc_id} contains the `blob` field. "
                "ChromaDocumentStore cannot store `blob` fields. "
                "This field will be ignored.",
                doc_id=doc.id,
            )
        data: dict[str, Any] = {"ids": [doc.id], "documents": [doc.content]}

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
                    "Document {doc_id} contains `meta` values of unsupported types for the keys: {keys}. "
                    "These items will be discarded. Supported types are: {types}.",
                    doc_id=doc.id,
                    keys=", ".join(discarded_keys),
                    types=", ".join([t.__name__ for t in SUPPORTED_TYPES_FOR_METADATA_VALUES]),
                )

            if valid_meta:
                data["metadatas"] = [valid_meta]

        if doc.embedding is not None:
            data["embeddings"] = [doc.embedding]

        if hasattr(doc, "sparse_embedding") and doc.sparse_embedding is not None:
            logger.warning(
                "Document {doc_id} has the `sparse_embedding` field set, "
                "but storing sparse embeddings in Chroma is not currently supported. "
                "The `sparse_embedding` field will be ignored.",
                doc_id=doc.id,
            )

        return data

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL,
    ) -> int:
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
            data = self._convert_document_to_chroma(doc)
            if data is not None:
                self._collection.add(**data)

        return len(documents)

    async def write_documents_async(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL,
    ) -> int:
        """
        Asynchronously writes (or overwrites) documents into the store.

        Asynchronous methods are only supported for HTTP connections.

        :param documents:
            A list of documents to write into the document store.
        :param policy:
            Not supported at the moment.

        :raises ValueError:
            When input is not valid.

        :returns:
            The number of documents written
        """
        await self._ensure_initialized_async()
        assert self._async_collection is not None

        for doc in documents:
            data = self._convert_document_to_chroma(doc)
            if data is not None:
                await self._async_collection.add(**data)

        return len(documents)

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.

        :param document_ids: the document ids to delete
        """
        self._ensure_initialized()
        assert self._collection is not None

        self._collection.delete(ids=document_ids)

    async def delete_documents_async(self, document_ids: list[str]) -> None:
        """
        Asynchronously deletes all documents with a matching document_ids from the document store.

        Asynchronous methods are only supported for HTTP connections.

        :param document_ids: the document ids to delete
        """
        await self._ensure_initialized_async()
        assert self._async_collection is not None

        await self._async_collection.delete(ids=document_ids)

    def delete_all_documents(self, *, recreate_index: bool = False) -> None:
        """
        Deletes all documents in the document store.

        A fast way to clear all documents from the document store while preserving any collection settings and mappings.
        :param recreate_index: Whether to recreate the index after deleting all documents.
        """
        self._ensure_initialized()  # _ensure_initialized ensures _client is not None and a collection exists
        assert self._collection is not None

        try:
            if recreate_index:
                # Store existing collection metadata and embedding function
                metadata = self._collection.metadata
                embedding_function = self._collection._embedding_function
                collection_name = self._collection_name

                # Delete the collection
                self._client.delete_collection(name=collection_name)

                # Recreate the collection with previous metadata
                self._collection = self._client.create_collection(
                    name=collection_name,
                    metadata=metadata,
                    embedding_function=embedding_function,
                )

            else:
                collection = self._collection.get()
                ids = collection.get("ids", [])
                self._collection.delete(ids=ids)  # type: ignore
                logger.info(
                    "Deleted all the {n_docs} documents from the collection '{name}'.",
                    name=self._collection_name,
                    n_docs=len(ids),
                )
        except Exception as e:
            msg = f"Failed to delete all documents from ChromaDB: {e!s}"
            raise DocumentStoreError(msg) from e

    async def delete_all_documents_async(self, *, recreate_index: bool = False) -> None:
        """
        Asynchronously deletes all documents in the document store.

        A fast way to clear all documents from the document store while preserving any collection settings and mappings.
        :param recreate_index: Whether to recreate the index after deleting all documents.
        """
        await self._ensure_initialized_async()  # ensures _async_client is not None
        assert self._async_collection is not None

        try:
            if recreate_index:
                # Store existing collection metadata and embedding function
                metadata = self._async_collection.metadata
                embedding_function = self._async_collection._embedding_function
                collection_name = self._collection_name

                # Delete the collection
                await self._async_client.delete_collection(name=collection_name)

                # Recreate the collection with previous metadata
                self._async_collection = await self._async_client.create_collection(
                    name=collection_name,
                    metadata=metadata,
                    embedding_function=embedding_function,
                )
            else:
                collection = await self._async_collection.get()
                ids = collection.get("ids", [])
                await self._async_collection.delete(ids=ids)  # type: ignore
                logger.info(
                    "Deleted all the {n_docs} documents from the collection '{name}'.",
                    name=self._collection_name,
                    n_docs=len(ids),
                )

        except Exception as e:
            msg = f"Failed to delete all documents from ChromaDB: {e!s}"
            raise DocumentStoreError(msg) from e

    def search(
        self,
        queries: list[str],
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[list[Document]]:
        """
        Search the documents in the store using the provided text queries.

        :param queries: the list of queries to search for.
        :param top_k: top_k documents to return for each query.
        :param filters: a dictionary of filters to apply to the search. Accepts filters in haystack format.
        :returns: matching documents for each query.
        """
        self._ensure_initialized()
        assert self._collection is not None

        kwargs = self._prepare_query_kwargs(filters)
        results = self._collection.query(
            query_texts=queries,
            n_results=top_k,
            **kwargs,
        )

        return self._query_result_to_documents(results)

    async def search_async(
        self,
        queries: list[str],
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[list[Document]]:
        """
        Asynchronously search the documents in the store using the provided text queries.

        Asynchronous methods are only supported for HTTP connections.

        :param queries: the list of queries to search for.
        :param top_k: top_k documents to return for each query.
        :param filters: a dictionary of filters to apply to the search. Accepts filters in haystack format.
        :returns: matching documents for each query.
        """
        await self._ensure_initialized_async()
        assert self._async_collection is not None

        kwargs = self._prepare_query_kwargs(filters)
        results = await self._async_collection.query(
            query_texts=queries,
            n_results=top_k,
            **kwargs,
        )

        return self._query_result_to_documents(results)

    def search_embeddings(
        self,
        query_embeddings: list[list[float]],
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[list[Document]]:
        """
        Perform vector search on the stored document, pass the embeddings of the queries instead of their text.

        :param query_embeddings: a list of embeddings to use as queries.
        :param top_k: the maximum number of documents to retrieve.
        :param filters: a dictionary of filters to apply to the search. Accepts filters in haystack format.

        :returns: a list of lists of documents that match the given filters.

        """
        self._ensure_initialized()
        assert self._collection is not None

        kwargs = self._prepare_query_kwargs(filters)
        results = self._collection.query(
            query_embeddings=cast(list[Sequence[float]], query_embeddings),
            n_results=top_k,
            **kwargs,
        )

        return self._query_result_to_documents(results)

    async def search_embeddings_async(
        self,
        query_embeddings: list[list[float]],
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[list[Document]]:
        """
        Asynchronously perform vector search on the stored document, pass the embeddings of the queries instead of
        their text.

        Asynchronous methods are only supported for HTTP connections.

        :param query_embeddings: a list of embeddings to use as queries.
        :param top_k: the maximum number of documents to retrieve.
        :param filters: a dictionary of filters to apply to the search. Accepts filters in haystack format.

        :returns: a list of lists of documents that match the given filters.

        """
        await self._ensure_initialized_async()
        assert self._async_collection is not None

        kwargs = self._prepare_query_kwargs(filters)
        results = await self._async_collection.query(
            query_embeddings=cast(list[Sequence[float]], query_embeddings),
            n_results=top_k,
            **kwargs,
        )

        return self._query_result_to_documents(results)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChromaDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
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
    def _get_result_to_documents(result: GetResult) -> list[Document]:
        """
        Helper function to convert Chroma results into Haystack Documents
        """
        retval = []
        documents = result.get("documents") or []
        for i in range(len(documents)):
            document_dict: dict[str, Any] = {"id": result["ids"][i]}

            result_documents = result.get("documents")
            if result_documents:
                document_dict["content"] = result_documents[i]

            result_metadata = result.get("metadatas")
            # Ensure metadata[i] is not None or don't add it to the document dict
            if result_metadata and result_metadata[i]:
                document_dict["meta"] = result_metadata[i]

            result_embeddings = result.get("embeddings")
            # Chroma can return different types of embeddings
            # https://github.com/chroma-core/chroma/blob/8ca12208aa3d3b33192a4a0f8b37501fd8293bfb/chromadb/api/types.py#L378

            if result_embeddings is not None:
                try:
                    # we first try to call tolist() which is available for numpy arrays
                    document_dict["embedding"] = result_embeddings[i].tolist()  # type: ignore
                except AttributeError:
                    document_dict["embedding"] = result_embeddings[i]

            retval.append(Document.from_dict(document_dict))

        return retval

    @staticmethod
    def _query_result_to_documents(
        result: QueryResult,
    ) -> list[list[Document]]:
        """
        Helper function to convert Chroma results into Haystack Documents
        """
        retval: list[list[Document]] = []
        documents = result.get("documents")
        if documents is None:
            return retval

        for i, answers in enumerate(documents):
            converted_answers = []
            for j in range(len(answers)):
                document_dict: dict[str, Any] = {
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
