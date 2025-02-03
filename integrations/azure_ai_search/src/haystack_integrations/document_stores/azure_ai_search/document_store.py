# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from .errors import AzureAISearchDocumentStoreConfigError
from .filters import _normalize_filters

type_mapping = {
    str: "Edm.String",
    bool: "Edm.Boolean",
    int: "Edm.Int32",
    float: "Edm.Double",
    datetime: "Edm.DateTimeOffset",
}

DEFAULT_VECTOR_SEARCH = VectorSearch(
    profiles=[
        VectorSearchProfile(name="default-vector-config", algorithm_configuration_name="cosine-algorithm-config")
    ],
    algorithms=[
        HnswAlgorithmConfiguration(
            name="cosine-algorithm-config",
            parameters=HnswParameters(
                metric=VectorSearchAlgorithmMetric.COSINE,
            ),
        )
    ],
)

logger = logging.getLogger(__name__)
logging.getLogger("azure").setLevel(logging.ERROR)
logging.getLogger("azure.identity").setLevel(logging.DEBUG)


class AzureAISearchDocumentStore:
    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("AZURE_SEARCH_API_KEY", strict=False),  # noqa: B008
        azure_endpoint: Secret = Secret.from_env_var("AZURE_SEARCH_SERVICE_ENDPOINT", strict=True),  # noqa: B008
        index_name: str = "default",
        embedding_dimension: int = 768,
        metadata_fields: Optional[Dict[str, type]] = None,
        vector_search_configuration: VectorSearch = None,
        **index_creation_kwargs,
    ):
        """
        A document store using [Azure AI Search](https://azure.microsoft.com/products/ai-services/ai-search/)
        as the backend.

        :param azure_endpoint: The URL endpoint of an Azure AI Search service.
        :param api_key: The API key to use for authentication.
        :param index_name: Name of index in Azure AI Search, if it doesn't exist it will be created.
        :param embedding_dimension: Dimension of the embeddings.
        :param metadata_fields: A dictionary of metadata keys and their types to create
            additional fields in index schema. As fields in Azure SearchIndex cannot be dynamic,
            it is necessary to specify the metadata fields in advance.
            (e.g. metadata_fields = {"author": str, "date": datetime})
        :param vector_search_configuration: Configuration option related to vector search.
            Default configuration uses the HNSW algorithm with cosine similarity to handle vector searches.

        :param index_creation_kwargs: Optional keyword parameters to be passed to `SearchIndex` class
            during index creation. Some of the supported parameters:
                - `semantic_search`: Defines semantic configuration of the search index. This parameter is needed
                to enable semantic search capabilities in index.
                - `similarity`: The type of similarity algorithm to be used when scoring and ranking the documents
                matching a search query. The similarity algorithm can only be defined at index creation time and
                cannot be modified on existing indexes.

        For more information on parameters, see the [official Azure AI Search documentation](https://learn.microsoft.com/en-us/azure/search/).
        """

        azure_endpoint = azure_endpoint or os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT") or None
        if not azure_endpoint:
            msg = "Please provide an Azure endpoint or set the environment variable AZURE_SEARCH_SERVICE_ENDPOINT."
            raise ValueError(msg)

        api_key = api_key or os.environ.get("AZURE_SEARCH_API_KEY") or None

        self._client = None
        self._index_client = None
        self._index_fields = []  # type: List[Any]  # stores all fields in the final schema of index
        self._api_key = api_key
        self._azure_endpoint = azure_endpoint
        self._index_name = index_name
        self._embedding_dimension = embedding_dimension
        self._dummy_vector = [-10.0] * self._embedding_dimension
        self._metadata_fields = metadata_fields
        self._vector_search_configuration = vector_search_configuration or DEFAULT_VECTOR_SEARCH
        self._index_creation_kwargs = index_creation_kwargs

    @property
    def client(self) -> SearchClient:

        # resolve secrets for authentication
        resolved_endpoint = (
            self._azure_endpoint.resolve_value() if isinstance(self._azure_endpoint, Secret) else self._azure_endpoint
        )
        resolved_key = self._api_key.resolve_value() if isinstance(self._api_key, Secret) else self._api_key

        credential = AzureKeyCredential(resolved_key) if resolved_key else DefaultAzureCredential()
        try:
            if not self._index_client:
                self._index_client = SearchIndexClient(
                    resolved_endpoint,
                    credential,
                )
            if not self._index_exists(self._index_name):
                # Create a new index if it does not exist
                logger.debug(
                    "The index '%s' does not exist. A new index will be created.",
                    self._index_name,
                )
                self._create_index(self._index_name)
        except (HttpResponseError, ClientAuthenticationError) as error:
            msg = f"Failed to authenticate with Azure Search: {error}"
            raise AzureAISearchDocumentStoreConfigError(msg) from error

        if self._index_client:
            # Get the search client, if index client is initialized
            index_fields = self._index_client.get_index(self._index_name).fields
            self._index_fields = [field.name for field in index_fields]
            self._client = self._index_client.get_search_client(self._index_name)
        else:
            msg = "Search Index Client is not initialized."
            raise AzureAISearchDocumentStoreConfigError(msg)

        return self._client

    def _create_index(self, index_name: str) -> None:
        """
        Creates a new search index.
        :param index_name: Name of the index to create. If None, the index name from the constructor is used.
        :param kwargs: Optional keyword parameters.
        """

        # default fields to create index based on Haystack Document (id, content, embedding)
        default_fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                hidden=False,
                vector_search_dimensions=self._embedding_dimension,
                vector_search_profile_name="default-vector-config",
            ),
        ]

        if not index_name:
            index_name = self._index_name
        if self._metadata_fields:
            default_fields.extend(self._create_metadata_index_fields(self._metadata_fields))
        index = SearchIndex(
            name=index_name,
            fields=default_fields,
            vector_search=self._vector_search_configuration,
            **self._index_creation_kwargs,
        )
        if self._index_client:
            self._index_client.create_index(index)

    def to_dict(self) -> Dict[str, Any]:
        # This is not the best solution to serialise this class but is the fastest to implement.
        # Not all kwargs types can be serialised to text so this can fail. We must serialise each
        # type explicitly to handle this properly.
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            azure_endpoint=self._azure_endpoint.to_dict() if self._azure_endpoint else None,
            api_key=self._api_key.to_dict() if self._api_key else None,
            index_name=self._index_name,
            embedding_dimension=self._embedding_dimension,
            metadata_fields=self._metadata_fields,
            vector_search_configuration=self._vector_search_configuration.as_dict(),
            **self._index_creation_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzureAISearchDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.

        :returns:
            Deserialized component.
        """

        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key", "azure_endpoint"])
        if (vector_search_configuration := data["init_parameters"].get("vector_search_configuration")) is not None:
            data["init_parameters"]["vector_search_configuration"] = VectorSearch.from_dict(vector_search_configuration)
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the search index.

        :returns: list of retrieved documents.
        """
        return self.client.get_document_count()

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes the provided documents to search index.

        :param documents: documents to write to the index.
        :param policy: Policy to determine how duplicates are handled.
        :raises ValueError: If the documents are not of type Document.
        :raises TypeError: If the document ids are not strings.
        :return: the number of documents added to index.
        """

        def _convert_input_document(documents: Document):
            document_dict = asdict(documents)
            if not isinstance(document_dict["id"], str):
                msg = f"Document id {document_dict['id']} is not a string, "
                raise TypeError(msg)
            index_document = self._convert_haystack_documents_to_azure(document_dict)

            return index_document

        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        if policy not in [DuplicatePolicy.NONE, DuplicatePolicy.OVERWRITE]:
            logger.warning(
                f"AzureAISearchDocumentStore only supports `DuplicatePolicy.OVERWRITE`"
                f"but got {policy}. Overwriting duplicates is enabled by default."
            )
        client = self.client
        documents_to_write = [(_convert_input_document(doc)) for doc in documents]

        if documents_to_write != []:
            client.upload_documents(documents_to_write)
        return len(documents_to_write)

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the search index.

        :param document_ids: ids of the documents to be deleted.
        """
        if self.count_documents() == 0:
            return
        documents = self._get_raw_documents_by_id(document_ids)
        if documents:
            self.client.delete_documents(documents)

    def get_documents_by_id(self, document_ids: List[str]) -> List[Document]:
        return self._convert_search_result_to_documents(self._get_raw_documents_by_id(document_ids))

    def search_documents(self, search_text: str = "*", top_k: int = 10) -> List[Document]:
        """
        Returns all documents that match the provided search_text.
        If search_text is None, returns all documents.
        :param search_text: the text to search for in the Document list.
        :param top_k: Maximum number of documents to return.
        :returns: A list of Documents that match the given search_text.
        """
        result = self.client.search(search_text=search_text, top=top_k)
        return self._convert_search_result_to_documents(list(result))

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the provided filters.
        Filters should be given as a dictionary supporting filtering by metadata. For details on
        filters, see the [metadata filtering documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

        :param filters: the filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        if filters:
            normalized_filters = _normalize_filters(filters)
            result = self.client.search(filter=normalized_filters)
            return self._convert_search_result_to_documents(result)
        else:
            return self.search_documents()

    def _convert_search_result_to_documents(self, azure_docs: List[Dict[str, Any]]) -> List[Document]:
        """
        Converts Azure search results to Haystack Documents.
        """
        documents = []

        for azure_doc in azure_docs:
            embedding = azure_doc.get("embedding")
            if embedding == self._dummy_vector:
                embedding = None

            # Anything besides default fields (id, content, and embedding) is considered metadata
            meta = {
                key: value
                for key, value in azure_doc.items()
                if key not in ["id", "content", "embedding"] and key in self._index_fields and value is not None
            }

            # Create the document with meta only if it's non-empty
            doc = Document(
                id=azure_doc["id"], content=azure_doc["content"], embedding=embedding, meta=meta if meta else {}
            )

            documents.append(doc)
        return documents

    def _index_exists(self, index_name: Optional[str]) -> bool:
        """
        Check if the index exists in the Azure AI Search service.

        :param index_name: The name of the index to check.
        :returns bool: whether the index exists.
        """

        if self._index_client and index_name:
            return index_name in self._index_client.list_index_names()
        else:
            msg = "Index name is required to check if the index exists."
            raise ValueError(msg)

    def _get_raw_documents_by_id(self, document_ids: List[str]):
        """
        Retrieves all Azure documents with a matching document_ids from the document store.

        :param document_ids: ids of the documents to be retrieved.
        :returns: list of retrieved Azure documents.
        """
        azure_documents = []
        for doc_id in document_ids:
            try:
                document = self.client.get_document(doc_id)
                azure_documents.append(document)
            except ResourceNotFoundError:
                logger.warning(f"Document with ID {doc_id} not found.")
        return azure_documents

    def _convert_haystack_documents_to_azure(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Map the document keys to fields of search index"""

        # Because Azure Search does not allow dynamic fields, we only include fields that are part of the schema
        index_document = {k: v for k, v in {**document, **document.get("meta", {})}.items() if k in self._index_fields}
        if index_document["embedding"] is None:
            index_document["embedding"] = self._dummy_vector

        return index_document

    def _create_metadata_index_fields(self, metadata: Dict[str, Any]) -> List[SimpleField]:
        """Create a list of index fields for storing metadata values."""

        index_fields = []
        metadata_field_mapping = self._map_metadata_field_types(metadata)

        for key, field_type in metadata_field_mapping.items():
            index_fields.append(SimpleField(name=key, type=field_type, filterable=True))

        return index_fields

    def _map_metadata_field_types(self, metadata: Dict[str, type]) -> Dict[str, str]:
        """Map metadata field types to Azure Search field types."""

        metadata_field_mapping = {}

        for key, value_type in metadata.items():

            if not key[0].isalpha():
                msg = (
                    f"Azure Search index only allows field names starting with letters. "
                    f"Invalid key: {key} will be dropped."
                )
                logger.warning(msg)
                continue

            field_type = type_mapping.get(value_type)
            if not field_type:
                error_message = f"Unsupported field type for key '{key}': {value_type}"
                raise ValueError(error_message)
            metadata_field_mapping[key] = field_type

        return metadata_field_mapping

    def _embedding_retrieval(
        self,
        query_embedding: List[float],
        *,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.
        It uses the vector configuration specified in the document store. By default, it uses the HNSW algorithm
        with cosine similarity.

        This method is not meant to be part of the public interface of
        `AzureAISearchDocumentStore` nor called directly.
        `AzureAISearchEmbeddingRetriever` uses this method directly and is the public interface for it.

        :param query_embedding: Embedding of the query.
        :param top_k: Maximum number of Documents to return.
        :param filters: Filters applied to the retrieved Documents.
        :param kwargs: Optional keyword arguments to pass to the Azure AI's search endpoint.

        :raises ValueError: If `query_embedding` is an empty list.
        :returns: List of Document that are most similar to `query_embedding`.
        """

        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=top_k, fields="embedding")
        result = self.client.search(vector_queries=[vector_query], filter=filters, **kwargs)
        azure_docs = list(result)
        return self._convert_search_result_to_documents(azure_docs)

    def _bm25_retrieval(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to `query`, using the BM25 algorithm.

        This method is not meant to be part of the public interface of
        `AzureAISearchDocumentStore` nor called directly.
        `AzureAISearchBM25Retriever` uses this method directly and is the public interface for it.

        :param query: Text of the query.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :param kwargs: Optional keyword arguments to pass to the Azure AI's search endpoint.


        :raises ValueError: If `query` is an empty string.
        :returns: List of Document that are most similar to `query`.
        """

        if query is None:
            msg = "query must not be None"
            raise ValueError(msg)

        result = self.client.search(search_text=query, filter=filters, top=top_k, **kwargs)
        azure_docs = list(result)
        return self._convert_search_result_to_documents(azure_docs)

    def _hybrid_retrieval(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Document]:
        """
        Retrieves documents similar to query using the vector configuration in the document store and
        the BM25 algorithm. This method combines vector similarity and BM25 for improved retrieval.

        This method is not meant to be part of the public interface of
        `AzureAISearchDocumentStore` nor called directly.
        `AzureAISearchHybridRetriever` uses this method directly and is the public interface for it.

        :param query: Text of the query.
        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :param kwargs: Optional keyword arguments to pass to the Azure AI's search endpoint.

        :raises ValueError: If `query` or `query_embedding` is empty.
        :returns: List of Document that are most similar to `query`.
        """

        if query is None:
            msg = "query must not be None"
            raise ValueError(msg)
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=top_k, fields="embedding")
        result = self.client.search(
            search_text=query,
            vector_queries=[vector_query],
            filter=filters,
            top=top_k,
            **kwargs,
        )
        azure_docs = list(result)
        return self._convert_search_result_to_documents(azure_docs)
