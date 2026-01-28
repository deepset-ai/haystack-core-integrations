# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import copy
from typing import Any, Literal

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from pinecone import Pinecone, PineconeAsyncio, PodSpec, ServerlessSpec
from pinecone.db_data import _Index, _IndexAsyncio
from pinecone.exceptions import NotFoundException

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
        spec: dict[str, Any] | None = None,
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

        self._index: _Index | None = None
        self._async_index: _IndexAsyncio | None = None
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
    def _convert_dict_spec_to_pinecone_object(spec: dict[str, Any]) -> ServerlessSpec | PodSpec:
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
    def from_dict(cls, data: dict[str, Any]) -> "PineconeDocumentStore":
        """
        Deserializes the component from a dictionary.
        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
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

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
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

        # if the operation is successful, result will have the upserted_count attribute
        return result.upserted_count  # type: ignore[union-attr]

    async def write_documents_async(
        self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
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

        # if the operation is successful, result will have the upserted_count attribute
        return result.upserted_count  # type: ignore[union-attr]

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

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

    async def filter_documents_async(self, filters: dict[str, Any] | None = None) -> list[Document]:
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

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """
        self._initialize_index()
        assert self._index is not None, "Index is not initialized"
        self._index.delete(ids=document_ids, namespace=self.namespace)

    async def delete_documents_async(self, document_ids: list[str]) -> None:
        """
        Asynchronously deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """
        await self._initialize_async_index()
        assert self._async_index is not None, "Index is not initialized"
        await self._async_index.delete(ids=document_ids, namespace=self.namespace)

    def delete_all_documents(self) -> None:
        """
        Deletes all documents in the document store.
        """
        self._initialize_index()
        assert self._index is not None, "Index is not initialized"
        try:
            self._index.delete(delete_all=True, namespace=self.namespace)
        except NotFoundException:
            # Namespace doesn't exist (empty collection), which is fine - nothing to delete
            logger.debug("Namespace '{namespace}' not found. Nothing to delete.", namespace=self.namespace or "default")

    async def delete_all_documents_async(self) -> None:
        """
        Asynchronously deletes all documents in the document store.
        """
        await self._initialize_async_index()
        assert self._async_index is not None, "Index is not initialized"
        try:
            await self._async_index.delete(delete_all=True, namespace=self.namespace)
        except NotFoundException:
            # Namespace doesn't exist (empty collection), which is fine - nothing to delete
            logger.debug("Namespace '{namespace}' not found. Nothing to delete.", namespace=self.namespace or "default")

    @staticmethod
    def _update_documents_metadata(documents: list[Document], meta: dict[str, Any]) -> None:
        """
        Updates metadata for a list of documents by merging the provided meta dictionary.

        :param documents: List of documents to update.
        :param meta: Metadata fields to merge into each document's existing metadata.
        """
        for document in documents:
            if document.meta is None:
                document.meta = {}
            document.meta.update(meta)

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Deletes all documents that match the provided filters.

        Pinecone does not support server-side delete by filter, so this method
        first searches for matching documents, then deletes them by ID.

        :param filters: The filters to apply to select documents for deletion.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents deleted.
        """
        _validate_filters(filters)

        self._initialize_index()
        assert self._index is not None, "Index is not initialized"

        documents = self.filter_documents(filters=filters)
        if not documents:
            return 0

        document_ids = [doc.id for doc in documents]

        self.delete_documents(document_ids)

        deleted_count = len(document_ids)
        logger.info(
            "Deleted {n_docs} documents from index '{index}' using filters.",
            n_docs=deleted_count,
            index=self.index_name,
        )

        return deleted_count

    async def delete_by_filter_async(self, filters: dict[str, Any]) -> int:
        """
        Asynchronously deletes all documents that match the provided filters.

        Pinecone does not support server-side delete by filter, so this method
        first searches for matching documents, then deletes them by ID.

        :param filters: The filters to apply to select documents for deletion.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents deleted.
        """
        _validate_filters(filters)

        await self._initialize_async_index()
        assert self._async_index is not None, "Index is not initialized"

        documents = await self.filter_documents_async(filters=filters)
        if not documents:
            return 0

        document_ids = [doc.id for doc in documents]

        await self.delete_documents_async(document_ids)

        deleted_count = len(document_ids)
        logger.info(
            "Deleted {n_docs} documents from index '{index}' using filters.",
            n_docs=deleted_count,
            index=self.index_name,
        )

        return deleted_count

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Updates the metadata of all documents that match the provided filters.

        Pinecone does not support server-side update by filter, so this method
        first searches for matching documents, then updates their metadata and re-writes them.

        :param filters: The filters to apply to select documents for updating.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param meta: The metadata fields to update. This will be merged with existing metadata.
        :returns: The number of documents updated.
        """
        _validate_filters(filters)

        if not isinstance(meta, dict):
            msg = "meta must be a dictionary"
            raise ValueError(msg)

        self._initialize_index()
        assert self._index is not None, "Index is not initialized"

        documents = self.filter_documents(filters=filters)
        if not documents:
            return 0

        self._update_documents_metadata(documents, meta)

        # Re-write documents with updated metadata
        # Using OVERWRITE policy to update existing documents
        self.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)

        updated_count = len(documents)
        logger.info(
            "Updated {n_docs} documents in index '{index}' using filters.",
            n_docs=updated_count,
            index=self.index_name,
        )

        return updated_count

    async def update_by_filter_async(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Asynchronously updates the metadata of all documents that match the provided filters.

        Pinecone does not support server-side update by filter, so this method
        first searches for matching documents, then updates their metadata and re-writes them.

        :param filters: The filters to apply to select documents for updating.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param meta: The metadata fields to update. This will be merged with existing metadata.
        :returns: The number of documents updated.
        """
        _validate_filters(filters)

        if not isinstance(meta, dict):
            msg = "meta must be a dictionary"
            raise ValueError(msg)

        await self._initialize_async_index()
        assert self._async_index is not None, "Index is not initialized"

        documents = await self.filter_documents_async(filters=filters)
        if not documents:
            return 0

        self._update_documents_metadata(documents, meta)

        # Re-write documents with updated metadata
        # Using OVERWRITE policy to update existing documents
        await self.write_documents_async(documents, policy=DuplicatePolicy.OVERWRITE)

        updated_count = len(documents)
        logger.info(
            "Updated {n_docs} documents in index '{index}' using filters.",
            n_docs=updated_count,
            index=self.index_name,
        )

        return updated_count

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
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
        query_embedding: list[float],
        *,
        namespace: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
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
    def _convert_meta_to_int(metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Pinecone store numeric metadata values as `float`. Some specific metadata are used in Retrievers components and
        are expected to be `int`. This method converts them back to integers.
        """
        values_to_convert = ["split_id", "split_idx_start", "page_number"]

        for value in values_to_convert:
            if value in metadata:
                metadata[value] = int(metadata[value]) if isinstance(metadata[value], float) else metadata[value]

        return metadata

    def _convert_query_result_to_documents(self, query_result: Any) -> list[Document]:
        pinecone_docs = query_result.matches
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

    def _convert_documents_to_pinecone_format(
        self, documents: list[Document]
    ) -> list[tuple[str, list[float], dict[str, Any]]]:
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

            metadata = dict(document.meta) if document.meta else {}

            # we save content as metadata
            if document.content is not None:
                metadata["content"] = document.content

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

            documents_for_pinecone.append((document.id, embedding, metadata))
        return documents_for_pinecone

    def _prepare_documents_for_writing(
        self, documents: list[Document], policy: DuplicatePolicy
    ) -> list[tuple[str, list[float], dict[str, Any]]]:
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

    @staticmethod
    def _count_documents_impl(documents: list[Document]) -> int:
        """Helper method to count documents and log warning if at TOP_K_LIMIT."""
        count = len(documents)
        if count == TOP_K_LIMIT:
            logger.warning(
                f"Count reached Pinecone's limit of {TOP_K_LIMIT} documents. "
                f"The actual number of matching documents may be higher."
            )
        return count

    @staticmethod
    def _count_unique_metadata_impl(documents: list[Document], metadata_fields: list[str]) -> dict[str, int]:
        """Helper method to count unique metadata values across specified fields."""
        result = {}
        for field in metadata_fields:
            unique_values = set()
            for doc in documents:
                if doc.meta and field in doc.meta:
                    value = doc.meta[field]
                    # Handle list values
                    if isinstance(value, list):
                        unique_values.update(value)
                    else:
                        unique_values.add(value)
            result[field] = len(unique_values)

        if len(documents) == TOP_K_LIMIT:
            logger.warning(
                f"Analysis limited to {TOP_K_LIMIT} documents due to Pinecone's limits. "
                f"Unique value counts may be incomplete."
            )
        return result

    @staticmethod
    def _get_metadata_fields_info_impl(documents: list[Document]) -> dict[str, dict[str, str]]:
        """Helper method to infer metadata field types from documents."""
        if not documents:
            return {}

        field_types: dict[str, dict[str, str]] = {}

        # Check if any document has content
        if any(doc.content is not None for doc in documents):
            field_types["content"] = {"type": "text"}

        # Collect all field values to infer types accurately
        field_samples: dict[str, set[str]] = {}

        for doc in documents:
            if doc.meta:
                for field, value in doc.meta.items():
                    if field not in field_samples:
                        field_samples[field] = set()

                    # Note: bool check MUST come before int/float because bool is a subclass of int in Python
                    if isinstance(value, bool):
                        field_samples[field].add("boolean")
                    elif isinstance(value, (int, float)):
                        field_samples[field].add("long")
                    elif isinstance(value, str):
                        field_samples[field].add("keyword")
                    elif isinstance(value, list):
                        # For lists, check the type of elements if list is non-empty
                        if value:
                            # Sample first element to determine list type
                            if isinstance(value[0], str):
                                field_samples[field].add("keyword")
                            elif isinstance(value[0], (int, float)):
                                field_samples[field].add("long")
                            elif isinstance(value[0], bool):
                                field_samples[field].add("boolean")
                        else:
                            # Empty list, default to keyword
                            field_samples[field].add("keyword")

        # Assign types based on collected samples
        for field, types_seen in field_samples.items():
            if len(types_seen) == 1:
                # Consistent type across all documents
                field_types[field] = {"type": types_seen.pop()}
            else:
                # Mixed types - default to keyword and log warning
                logger.warning(
                    f"Field '{field}' has mixed types {types_seen} across documents. "
                    f"Defaulting to 'keyword' type. Consider using consistent types for better query performance."
                )
                field_types[field] = {"type": "keyword"}

        if len(documents) == TOP_K_LIMIT:
            logger.info(
                f"Schema inference based on {TOP_K_LIMIT} documents (Pinecone's query limit). "
                f"If you have more documents with different metadata fields, they won't be reflected here."
            )

        return field_types

    @staticmethod
    def _get_metadata_field_min_max_impl(documents: list[Document], metadata_field: str) -> dict[str, Any]:
        """Helper method to get min/max values for a metadata field (supports numeric, boolean, and string types)."""
        values: list[bool | int | float | str] = []
        for doc in documents:
            if doc.meta and metadata_field in doc.meta:
                value = doc.meta[metadata_field]
                # Note: bool check must come before numeric because bool is subclass of int
                if isinstance(value, bool):
                    values.append(value)
                elif isinstance(value, (int, float)):
                    values.append(value)
                elif isinstance(value, str):
                    values.append(value)

        if not values:
            msg = f"No values found for metadata field '{metadata_field}'"
            raise ValueError(msg)

        result = {"min": min(values), "max": max(values)}

        if len(documents) == TOP_K_LIMIT:
            logger.warning(
                f"Min/max calculation limited to {TOP_K_LIMIT} documents. "
                f"Results may not reflect the true min/max across all documents."
            )

        return result

    @staticmethod
    def _get_metadata_field_unique_values_impl(
        documents: list[Document], metadata_field: str, search_term: str | None, from_: int, size: int
    ) -> tuple[list[str], int]:
        """Helper method to get unique values for a metadata field with search and pagination."""
        unique_values: set[str] = set()
        for doc in documents:
            if doc.meta and metadata_field in doc.meta:
                value = doc.meta[metadata_field]
                # Handle list values
                if isinstance(value, list):
                    unique_values.update(str(v) for v in value)
                else:
                    unique_values.add(str(value))

        # Convert to sorted list
        unique_values_list = sorted(unique_values)

        # Apply search term filter if provided
        if search_term:
            search_term_lower = search_term.lower()
            unique_values_list = [v for v in unique_values_list if search_term_lower in v.lower()]

        total_count = len(unique_values_list)

        # Apply pagination
        paginated_values = unique_values_list[from_ : from_ + size]

        if len(documents) == TOP_K_LIMIT:
            logger.warning(f"Unique values extraction limited to {TOP_K_LIMIT} documents. Results may be incomplete.")

        return paginated_values, total_count

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Returns the count of documents that match the provided filters.

        Note: Due to Pinecone's limitations, this method fetches documents and counts them.
        For large result sets, this is subject to Pinecone's TOP_K_LIMIT of 1000 documents.

        :param filters: The filters to apply to the document list.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents that match the filters.
        """
        documents = self.filter_documents(filters=filters)
        return self._count_documents_impl(documents)

    async def count_documents_by_filter_async(self, filters: dict[str, Any]) -> int:
        """
        Asynchronously returns the count of documents that match the provided filters.

        Note: Due to Pinecone's limitations, this method fetches documents and counts them.
        For large result sets, this is subject to Pinecone's TOP_K_LIMIT of 1000 documents.

        :param filters: The filters to apply to the document list.
        :returns: The number of documents that match the filters.
        """
        documents = await self.filter_documents_async(filters=filters)
        return self._count_documents_impl(documents)

    def count_unique_metadata_by_filter(self, filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]:
        """
        Counts unique values for each specified metadata field in documents matching the filters.

        Note: Due to Pinecone's limitations, this method fetches documents and aggregates in Python.
        Subject to Pinecone's TOP_K_LIMIT of 1000 documents.

        :param filters: The filters to apply to select documents.
        :param metadata_fields: List of metadata field names to count unique values for.
        :returns: Dictionary mapping field names to counts of unique values.
        """
        documents = self.filter_documents(filters=filters)
        return self._count_unique_metadata_impl(documents, metadata_fields)

    async def count_unique_metadata_by_filter_async(
        self, filters: dict[str, Any], metadata_fields: list[str]
    ) -> dict[str, int]:
        """
        Asynchronously counts unique values for each specified metadata field in documents matching the filters.

        Note: Due to Pinecone's limitations, this method fetches documents and aggregates in Python.
        Subject to Pinecone's TOP_K_LIMIT of 1000 documents.

        :param filters: The filters to apply to select documents.
        :param metadata_fields: List of metadata field names to count unique values for.
        :returns: Dictionary mapping field names to counts of unique values.
        """
        documents = await self.filter_documents_async(filters=filters)
        return self._count_unique_metadata_impl(documents, metadata_fields)

    def get_metadata_fields_info(self) -> dict[str, dict[str, str]]:
        """
        Returns information about metadata fields and their types by sampling documents.

        Note: Pinecone doesn't provide a schema introspection API, so this method infers field types
        by examining the metadata of documents stored in the index (up to 1000 documents).

        Type mappings:
        - 'text': Document content field
        - 'keyword': String metadata values
        - 'long': Numeric metadata values (int or float)
        - 'boolean': Boolean metadata values

        :returns: Dictionary mapping field names to type information.
            Example:
            ```python
            {
                'content': {'type': 'text'},
                'category': {'type': 'keyword'},
                'status': {'type': 'keyword'},
                'priority': {'type': 'long'},
            }
            ```

        """
        documents = self.filter_documents(filters=None)
        return self._get_metadata_fields_info_impl(documents)

    async def get_metadata_fields_info_async(self) -> dict[str, dict[str, str]]:
        """
        Asynchronously returns information about metadata fields and their types by sampling documents.

        Note: Pinecone doesn't provide a schema introspection API, so this method infers field types
        by examining the metadata of documents stored in the index (up to 1000 documents).

        Type mappings:
        - 'text': Document content field
        - 'keyword': String metadata values
        - 'long': Numeric metadata values (int or float)
        - 'boolean': Boolean metadata values

        :returns: Dictionary mapping field names to type information.
            Example: {'content': {'type': 'text'}, 'category': {'type': 'keyword'}, 'priority': {'type': 'long'}}
        """
        documents = await self.filter_documents_async(filters=None)
        return self._get_metadata_fields_info_impl(documents)

    def get_metadata_field_min_max(self, metadata_field: str) -> dict[str, Any]:
        """
        Returns the minimum and maximum values for a metadata field.

        Supports numeric (int, float), boolean, and string (keyword) types:
        - Numeric: Returns min/max based on numeric value
        - Boolean: Returns False as min, True as max
        - String: Returns min/max based on alphabetical ordering

        Note: This method fetches all documents and computes min/max in Python.
        Subject to Pinecone's TOP_K_LIMIT of 1000 documents.

        :param metadata_field: The metadata field name to analyze.
        :returns: Dictionary with 'min' and 'max' keys.
        :raises ValueError: If the field doesn't exist or has no values.
        """
        documents = self.filter_documents(filters=None)
        return self._get_metadata_field_min_max_impl(documents, metadata_field)

    async def get_metadata_field_min_max_async(self, metadata_field: str) -> dict[str, Any]:
        """
        Asynchronously returns the minimum and maximum values for a metadata field.

        Supports numeric (int, float), boolean, and string (keyword) types:
        - Numeric: Returns min/max based on numeric value
        - Boolean: Returns False as min, True as max
        - String: Returns min/max based on alphabetical ordering

        Note: This method fetches all documents and computes min/max in Python.
        Subject to Pinecone's TOP_K_LIMIT of 1000 documents.

        :param metadata_field: The metadata field name to analyze.
        :returns: Dictionary with 'min' and 'max' keys.
        :raises ValueError: If the field doesn't exist or has no values.
        """
        documents = await self.filter_documents_async(filters=None)
        return self._get_metadata_field_min_max_impl(documents, metadata_field)

    def get_metadata_field_unique_values(
        self, metadata_field: str, search_term: str | None = None, from_: int = 0, size: int = 10
    ) -> tuple[list[str], int]:
        """
        Retrieves unique values for a metadata field with optional search and pagination.

        Note: This method fetches documents and extracts unique values in Python.
        Subject to Pinecone's TOP_K_LIMIT of 1000 documents.

        :param metadata_field: The metadata field name to get unique values for.
        :param search_term: Optional search term to filter values (case-insensitive substring match).
        :param from_: Starting offset for pagination (default: 0).
        :param size: Number of values to return (default: 10).
        :returns: Tuple of (list of unique values, total count of matching values).
        """
        documents = self.filter_documents(filters=None)
        return self._get_metadata_field_unique_values_impl(documents, metadata_field, search_term, from_, size)

    async def get_metadata_field_unique_values_async(
        self, metadata_field: str, search_term: str | None = None, from_: int = 0, size: int = 10
    ) -> tuple[list[str], int]:
        """
        Asynchronously retrieves unique values for a metadata field with optional search and pagination.

        Note: This method fetches documents and extracts unique values in Python.
        Subject to Pinecone's TOP_K_LIMIT of 1000 documents.

        :param metadata_field: The metadata field name to get unique values for.
        :param search_term: Optional search term to filter values (case-insensitive substring match).
        :param from_: Starting offset for pagination (default: 0).
        :param size: Number of values to return (default: 10).
        :returns: Tuple of (list of unique values, total count of matching values).
        """
        documents = await self.filter_documents_async(filters=None)
        return self._get_metadata_field_unique_values_impl(documents, metadata_field, search_term, from_, size)
