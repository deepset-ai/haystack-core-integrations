# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from copy import copy
from typing import Any, Dict, List, Literal, Optional, Union

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from pinecone import Pinecone, PineconeAsyncio, PodSpec, ServerlessSpec
from pinecone.db_data import _Index, _IndexAsyncio

from .filters import _normalize_filters, _validate_filters

logger = logging.getLogger(__name__)

# Pinecone has a limit of 1000 documents that can be returned in a query
# with include_metadata=True or include_data=True
# https://docs.pinecone.io/docs/limits
TOP_K_LIMIT = 1_000


DEFAULT_STARTER_PLAN_SPEC = {"serverless": {"region": "us-east-1", "cloud": "aws"}}
METADATA_SUPPORTED_TYPES = str, int, bool, float  # List[str] is supported and checked separately


class PineconeDocumentStore:
    """
    A Document Store using [Pinecone vector database](https://www.pinecone.io/).
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("PINECONE_API_KEY"),  # noqa: B008
        index: str = "default",
        namespace: str = "default",
        batch_size: int = 100,
        dimension: int = 768,
        spec: Optional[Dict[str, Any]] = None,
        metric: Literal["cosine", "euclidean", "dotproduct"] = "cosine",
    ):
        """
        Creates a new PineconeDocumentStore instance.
        It is meant to be connected to a Pinecone index and namespace.

        :param api_key: The Pinecone API key.
        :param index: The Pinecone index to connect to. If the index does not exist, it will be created.
        :param namespace: The Pinecone namespace to connect to. If the namespace does not exist, it will be created
            at the first write.
        :param batch_size: The number of documents to write in a single batch. When setting this parameter,
            consider [documented Pinecone limits](https://docs.pinecone.io/reference/quotas-and-limits).
        :param dimension: The dimension of the embeddings. This parameter is only used when creating a new index.
        :param spec: The Pinecone spec to use when creating a new index. Allows choosing between serverless and pod
            deployment options and setting additional parameters. Refer to the
            [Pinecone documentation](https://docs.pinecone.io/reference/api/control-plane/create_index) for more
            details.
            If not provided, a default spec with serverless deployment in the `us-east-1` region will be used
            (compatible with the free tier).
        :param metric: The metric to use for similarity search. This parameter is only used when creating a new index.

        """
        self.api_key = api_key
        spec = spec or DEFAULT_STARTER_PLAN_SPEC
        self.namespace = namespace
        self.batch_size = batch_size
        self.metric = metric
        self.spec = spec
        self.dimension = dimension
        self.index_name = index

        self._index: Optional[_Index] = None
        self._async_index: Optional[_IndexAsyncio] = None
        self._dummy_vector = [-10.0] * self.dimension

    def _initialize_index(self):
        if self._index is not None:
            return self._index

        client = Pinecone(api_key=self.api_key.resolve_value(), source_tag="haystack")

        if self.index_name not in client.list_indexes().names():
            logger.info(f"Index {self.index_name} does not exist. Creating a new index.")
            pinecone_spec = self._convert_dict_spec_to_pinecone_object(self.spec)
            client.create_index(name=self.index_name, dimension=self.dimension, spec=pinecone_spec, metric=self.metric)
        else:
            logger.info(
                f"Connecting to existing index {self.index_name}. `dimension`, `spec`, and `metric` will be ignored."
            )

        self._index = client.Index(name=self.index_name)

        actual_dimension = self._index.describe_index_stats().get("dimension")
        if actual_dimension and actual_dimension != self.dimension:
            logger.warning(
                f"Dimension of index {self.index_name} is {actual_dimension}, but {self.dimension} was specified. "
                "The specified dimension will be ignored."
                "If you need an index with a different dimension, please create a new one."
            )
        self.dimension = actual_dimension or self.dimension
        self._dummy_vector = [-10.0] * self.dimension

    async def _initialize_async_index(self):
        if self._async_index is not None:
            return self._async_index

        async_client = PineconeAsyncio(api_key=self.api_key.resolve_value(), source_tag="haystack")

        indexes = await async_client.list_indexes()
        if self.index_name not in indexes.names():
            logger.info(f"Index {self.index_name} does not exist. Creating a new index.")
            pinecone_spec = self._convert_dict_spec_to_pinecone_object(self.spec)
            new_index = await async_client.create_index(
                name=self.index_name, dimension=self.dimension, spec=pinecone_spec, metric=self.metric
            )
            host = new_index["host"]
        else:
            logger.info(
                f"Connecting to existing index {self.index_name}. `dimension`, `spec`, and `metric` will be ignored."
            )
            host = next((index["host"] for index in indexes if index["name"] == self.index_name), None)

        self._async_index = async_client.IndexAsyncio(host=host)

        index_stats = await self._async_index.describe_index_stats()
        actual_dimension = index_stats.get("dimension")
        if actual_dimension and actual_dimension != self.dimension:
            logger.warning(
                f"Dimension of index {self.index_name} is {actual_dimension}, but {self.dimension} was specified. "
                "The specified dimension will be ignored."
                "If you need an index with a different dimension, please create a new one."
            )
        self.dimension = actual_dimension or self.dimension
        self._dummy_vector = [-10.0] * self.dimension

        await async_client.close()

    def close(self):
        """
        Close the associated synchronous resources.
        """
        if self._index:
            self._index.close()
            self._index = None

    async def close_async(self):
        """
        Close the associated asynchronous resources. To be invoked manually when the Document Store is no longer needed.
        """
        if self._async_index:
            await self._async_index.close()
            self._async_index = None

    @staticmethod
    def _convert_dict_spec_to_pinecone_object(spec: Dict[str, Any]) -> Union[ServerlessSpec, PodSpec]:
        """Convert the spec dictionary to a Pinecone spec object"""

        if "serverless" in spec:
            serverless_spec = spec["serverless"]
            return ServerlessSpec(**serverless_spec)
        if "pod" in spec:
            pod_spec = spec["pod"]
            return PodSpec(**pod_spec)

        msg = (
            "Invalid spec. Must contain either `serverless` or `pod` key. "
            "Refer to https://docs.pinecone.io/reference/api/control-plane/create_index for more details."
        )
        raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PineconeDocumentStore":
        """
        Deserializes the component from a dictionary.
        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.
        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            spec=self.spec,
            index=self.index_name,
            dimension=self.dimension,
            namespace=self.namespace,
            batch_size=self.batch_size,
            metric=self.metric,
        )

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        self._initialize_index()
        assert self._index is not None, "Index is not initialized"

        try:
            count = self._index.describe_index_stats()["namespaces"][self.namespace]["vector_count"]
        except KeyError:
            count = 0
        return count

    async def count_documents_async(self) -> int:
        """
        Asynchronously returns how many documents are present in the document store.
        """
        await self._initialize_async_index()
        assert self._async_index is not None, "Index is not initialized"

        try:
            index_stats = await self._async_index.describe_index_stats()
            count = index_stats["namespaces"][self.namespace]["vector_count"]
        except KeyError:
            count = 0
        return count

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes Documents to Pinecone.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
            PineconeDocumentStore only supports `DuplicatePolicy.OVERWRITE`.

        :returns: The number of documents written to the document store.
        """
        self._initialize_index()
        assert self._index is not None, "Index is not initialized"

        documents_for_pinecone = self._prepare_documents_for_writing(documents, policy)

        result = self._index.upsert(
            vectors=documents_for_pinecone, namespace=self.namespace, batch_size=self.batch_size
        )

        written_docs = result["upserted_count"]
        return written_docs

    async def write_documents_async(
        self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """
        Asynchronously writes Documents to Pinecone.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
            PineconeDocumentStore only supports `DuplicatePolicy.OVERWRITE`.

        :returns: The number of documents written to the document store.
        """
        await self._initialize_async_index()
        assert self._async_index is not None, "Index is not initialized"

        documents_for_pinecone = self._prepare_documents_for_writing(documents, policy)

        result = await self._async_index.upsert(
            vectors=documents_for_pinecone, namespace=self.namespace, batch_size=self.batch_size
        )

        written_docs = result["upserted_count"]

        return written_docs

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """

        _validate_filters(filters)

        self._initialize_index()
        assert self._index is not None, "Index is not initialized"

        # Pinecone only performs vector similarity search
        # here we are querying with a dummy vector and the max compatible top_k
        documents = self._embedding_retrieval(query_embedding=self._dummy_vector, filters=filters, top_k=TOP_K_LIMIT)

        # when simply filtering, we don't want to return any scores
        # furthermore, we are querying with a dummy vector, so the scores are meaningless
        for doc in documents:
            doc.score = None

        if len(documents) == TOP_K_LIMIT:
            logger.warning(
                f"PineconeDocumentStore can return at most {TOP_K_LIMIT} documents and the query has hit this limit. "
                f"It is likely that there are more matching documents in the document store. "
            )
        return documents

    async def filter_documents_async(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Asynchronously returns the documents that match the filters provided.

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        _validate_filters(filters)

        await self._initialize_async_index()
        assert self._async_index is not None, "Index is not initialized"

        documents = await self._embedding_retrieval_async(
            query_embedding=self._dummy_vector, filters=filters, top_k=TOP_K_LIMIT
        )

        for doc in documents:
            doc.score = None

        if len(documents) == TOP_K_LIMIT:
            logger.warning(
                f"PineconeDocumentStore can return at most {TOP_K_LIMIT} documents and the query has hit this limit. "
                f"It is likely that there are more matching documents in the document store. "
            )

        return documents

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """
        self._initialize_index()
        assert self._index is not None, "Index is not initialized"
        self._index.delete(ids=document_ids, namespace=self.namespace)

    async def delete_documents_async(self, document_ids: List[str]) -> None:
        """
        Asynchronously deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """
        await self._initialize_async_index()
        assert self._async_index is not None, "Index is not initialized"
        await self._async_index.delete(ids=document_ids, namespace=self.namespace)

    def _embedding_retrieval(
        self,
        query_embedding: List[float],
        *,
        namespace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.

        This method is not mean to be part of the public interface of
        `PineconeDocumentStore` nor called directly.
        `PineconeEmbeddingRetriever` uses this method directly and is the public interface for it.

        :param query_embedding: Embedding of the query.
        :param namespace: Pinecone namespace to query. Defaults the namespace of the document store.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.

        :returns: List of Document that are most similar to `query_embedding`
        """

        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        _validate_filters(filters)
        filters = _normalize_filters(filters) if filters else None
        self._initialize_index()
        assert self._index is not None, "Index is not initialized"

        result = self._index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace or self.namespace,
            filter=filters,
            include_values=True,
            include_metadata=True,
        )

        return self._convert_query_result_to_documents(result)

    async def _embedding_retrieval_async(
        self,
        query_embedding: List[float],
        *,
        namespace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Document]:
        """
        Asynchronously retrieves documents that are similar to the query embedding using a vector similarity metric.

        :param query_embedding: Embedding of the query.
        :param namespace: Pinecone namespace to query. Defaults the namespace of the document store.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.

        :returns: List of Document that are most similar to `query_embedding`
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        _validate_filters(filters)
        filters = _normalize_filters(filters) if filters else None

        await self._initialize_async_index()
        assert self._async_index is not None, "Index is not initialized"

        result = await self._async_index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace or self.namespace,
            filter=filters,
            include_values=True,
            include_metadata=True,
        )

        return self._convert_query_result_to_documents(result)

    @staticmethod
    def _convert_meta_to_int(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pinecone store numeric metadata values as `float`. Some specific metadata are used in Retrievers components and
        are expected to be `int`. This method converts them back to integers.
        """
        values_to_convert = ["split_id", "split_idx_start", "page_number"]

        for value in values_to_convert:
            if value in metadata:
                metadata[value] = int(metadata[value]) if isinstance(metadata[value], float) else metadata[value]

        return metadata

    def _convert_query_result_to_documents(self, query_result: Dict[str, Any]) -> List[Document]:
        pinecone_docs = query_result["matches"]
        documents = []
        for pinecone_doc in pinecone_docs:
            content = pinecone_doc["metadata"].pop("content", None)

            # we always store vectors during writing but we don't want to return them if they are dummy vectors
            embedding = None
            if pinecone_doc["values"] != self._dummy_vector:
                embedding = pinecone_doc["values"]

            doc = Document(
                id=pinecone_doc["id"],
                content=content,
                meta=self._convert_meta_to_int(pinecone_doc["metadata"]),
                embedding=embedding,
                score=pinecone_doc["score"],
            )
            documents.append(doc)

        return documents

    @staticmethod
    def _discard_invalid_meta(document: Document) -> None:
        """
        Remove metadata fields with unsupported types from the document.
        """

        def valid_type(value: Any) -> bool:
            return isinstance(value, METADATA_SUPPORTED_TYPES) or (
                isinstance(value, list) and all(isinstance(i, str) for i in value)
            )

        if document.meta:
            discarded_keys = []
            new_meta = {}
            for key, value in document.meta.items():
                if not valid_type(value):
                    discarded_keys.append(key)
                else:
                    new_meta[key] = value

            if discarded_keys:
                msg = (
                    f"Document {document.id} has metadata fields with unsupported types: {discarded_keys}. "
                    f"Only str, int, bool, and List[str] are supported. The values of these fields will be discarded."
                )
                logger.warning(msg)

            document.meta = new_meta

    def _convert_documents_to_pinecone_format(self, documents: List[Document]) -> List[Dict[str, Any]]:
        documents_for_pinecone = []
        for document in documents:
            embedding = copy(document.embedding)
            if embedding is None:
                logger.warning(
                    f"Document {document.id} has no embedding. Pinecone is a purely vector database. "
                    "A dummy embedding will be used, but this can affect the search results. "
                )
                embedding = self._dummy_vector

            if document.meta:
                self._discard_invalid_meta(document)

            doc_for_pinecone: Dict[str, Any] = {"id": document.id, "values": embedding, "metadata": dict(document.meta)}

            # we save content as metadata
            if document.content is not None:
                doc_for_pinecone["metadata"]["content"] = document.content

            # currently, storing blob in Pinecone is not supported
            if document.blob is not None:
                logger.warning(
                    f"Document {document.id} has the `blob` field set, but storing `ByteStream` "
                    "objects in Pinecone is not supported. "
                    "The content of the `blob` field will be ignored."
                )
            if hasattr(document, "sparse_embedding") and document.sparse_embedding is not None:
                logger.warning(
                    "Document {document_id} has the `sparse_embedding` field set,"
                    "but storing sparse embeddings in Pinecone is not currently supported."
                    "The `sparse_embedding` field will be ignored.",
                    document_id=document.id,
                )

            documents_for_pinecone.append(doc_for_pinecone)
        return documents_for_pinecone

    def _prepare_documents_for_writing(
        self, documents: List[Document], policy: DuplicatePolicy
    ) -> List[Dict[str, Any]]:
        """
        Helper method to prepare documents for writing to Pinecone.
        """
        if len(documents) > 0 and not isinstance(documents[0], Document):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

        if policy not in [DuplicatePolicy.NONE, DuplicatePolicy.OVERWRITE]:
            logger.warning(
                f"PineconeDocumentStore only supports `DuplicatePolicy.OVERWRITE`"
                f"but got {policy}. Overwriting duplicates is enabled by default."
            )

        return self._convert_documents_to_pinecone_format(documents)
