# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging as python_logging
from datetime import datetime
from typing import Any, Optional, Union

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
)
from azure.core.pipeline.policies import UserAgentPolicy
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    CharFilter,
    CorsOptions,
    HnswAlgorithmConfiguration,
    HnswParameters,
    LexicalAnalyzer,
    LexicalTokenizer,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchResourceEncryptionKey,
    SearchSuggester,
    SimilarityAlgorithm,
    SimpleField,
    TokenFilter,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery
from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from .errors import AzureAISearchDocumentStoreConfigError
from .filters import _normalize_filters

USER_AGENT = "haystack-integrations/azure-ai-search"

type_mapping = {
    str: "Edm.String",
    bool: "Edm.Boolean",
    int: "Edm.Int32",
    float: "Edm.Double",
    datetime: "Edm.DateTimeOffset",
}

# Map of expected field names to their corresponding classes
AZURE_CLASS_MAPPING: dict[str, Any] = {
    "suggesters": SearchSuggester,
    "analyzers": LexicalAnalyzer,
    "tokenizers": LexicalTokenizer,
    "token_filters": TokenFilter,
    "char_filters": CharFilter,
    "cors_options": CorsOptions,
    "similarity_algorithm": SimilarityAlgorithm,
    "encryption_key": SearchResourceEncryptionKey,
    "scoring_profiles": ScoringProfile,
}

DEFAULT_VECTOR_SEARCH = VectorSearch(
    profiles=[
        VectorSearchProfile(
            name="default-vector-config",
            algorithm_configuration_name="cosine-algorithm-config",
        )
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
python_logging.getLogger("azure").setLevel(python_logging.ERROR)
python_logging.getLogger("azure.identity").setLevel(python_logging.DEBUG)


class AzureAISearchDocumentStore:
    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("AZURE_AI_SEARCH_API_KEY", strict=False),  # noqa: B008
        azure_endpoint: Secret = Secret.from_env_var("AZURE_AI_SEARCH_ENDPOINT", strict=True),  # noqa: B008
        index_name: str = "default",
        embedding_dimension: int = 768,
        metadata_fields: Optional[dict[str, Union[SearchField, type]]] = None,
        vector_search_configuration: Optional[VectorSearch] = None,
        include_search_metadata: bool = False,
        **index_creation_kwargs: Any,
    ):
        """
        A document store using [Azure AI Search](https://azure.microsoft.com/products/ai-services/ai-search/)
        as the backend.

        :param azure_endpoint: The URL endpoint of an Azure AI Search service.
        :param api_key: The API key to use for authentication.
        :param index_name: Name of index in Azure AI Search, if it doesn't exist it will be created.
        :param embedding_dimension: Dimension of the embeddings.
        :param metadata_fields: A dictionary mapping metadata field names to their corresponding field definitions.
            Each field can be defined either as:
            - A SearchField object to specify detailed field configuration like type, searchability, and filterability
            - A Python type (`str`, `bool`, `int`, `float`, or `datetime`) to create a simple filterable field

            These fields are automatically added when creating the search index.
            Example:
            ```python
            metadata_fields={
                "Title": SearchField(
                    name="Title",
                    type="Edm.String",
                    searchable=True,
                    filterable=True
                ),
                "Pages": int
            }
            ```
        :param vector_search_configuration: Configuration option related to vector search.
            Default configuration uses the HNSW algorithm with cosine similarity to handle vector searches.

        :param include_search_metadata: Whether to include Azure AI Search metadata fields
            in the returned documents. When set to True, the `meta` field of the returned
            documents will contain the @search.score, @search.reranker_score, @search.highlights,
            @search.captions, and other fields returned by Azure AI Search.
        :param index_creation_kwargs: Optional keyword parameters to be passed to `SearchIndex` class
            during index creation. Some of the supported parameters:
                - `semantic_search`: Defines semantic configuration of the search index. This parameter is needed
                to enable semantic search capabilities in index.
                - `similarity`: The type of similarity algorithm to be used when scoring and ranking the documents
                matching a search query. The similarity algorithm can only be defined at index creation time and
                cannot be modified on existing indexes.

        For more information on parameters, see the [official Azure AI Search documentation](https://learn.microsoft.com/en-us/azure/search/).
        """
        self._client: Optional[SearchClient] = None
        self._index_client: Optional[SearchIndexClient] = None
        self._index_fields = []  # type: list[Any]  # stores all fields in the final schema of index
        self._api_key = api_key
        self._azure_endpoint = azure_endpoint
        self._index_name = index_name
        self._embedding_dimension = embedding_dimension
        self._dummy_vector = [-10.0] * self._embedding_dimension
        self._metadata_fields = self._normalize_metadata_index_fields(metadata_fields)
        self._vector_search_configuration = vector_search_configuration or DEFAULT_VECTOR_SEARCH
        self._include_search_metadata = include_search_metadata
        self._index_creation_kwargs = index_creation_kwargs

    @property
    def client(self) -> SearchClient:
        resolved_endpoint = self._azure_endpoint.resolve_value()
        resolved_key = self._api_key.resolve_value()

        credential = AzureKeyCredential(resolved_key) if resolved_key else DefaultAzureCredential()

        # build a UserAgentPolicy to be used for the request
        ua_policy = UserAgentPolicy(user_agent=USER_AGENT)
        try:
            if not self._index_client:
                self._index_client = SearchIndexClient(
                    # resolve_value, with Secret.from_env_var (strict=True), returns a string or raises an error
                    endpoint=resolved_endpoint,  # type: ignore[arg-type]
                    credential=credential,
                    user_agent=ua_policy,
                )
            if not self._index_exists(self._index_name):
                # Create a new index if it does not exist
                logger.debug(
                    "The index '{idx_name}' does not exist. A new index will be created.",
                    idx_name=self._index_name,
                )
                self._create_index()
        except (HttpResponseError, ClientAuthenticationError) as error:
            msg = f"Failed to authenticate with Azure Search: {error}"
            raise AzureAISearchDocumentStoreConfigError(msg) from error

        if self._index_client:
            # Get the search client, if index client is initialized
            index_fields = self._index_client.get_index(self._index_name).fields
            self._index_fields = [field.name for field in index_fields]
            self._client = self._index_client.get_search_client(self._index_name, user_agent=ua_policy)
        else:
            msg = "Search Index Client is not initialized."
            raise AzureAISearchDocumentStoreConfigError(msg)

        return self._client

    def _normalize_metadata_index_fields(
        self, metadata_fields: Optional[dict[str, Union[SearchField, type]]]
    ) -> dict[str, SearchField]:
        """Create a list of index fields for storing metadata values."""

        if not metadata_fields:
            return {}

        normalized_fields = {}

        for key, value in metadata_fields.items():
            if isinstance(value, SearchField):
                if value.name == key:
                    normalized_fields[key] = value
                else:
                    msg = f"Name of SearchField ('{value.name}') must match metadata field name ('{key}')"
                    raise ValueError(msg)
            else:
                if not key[0].isalpha():
                    msg = (
                        f"Azure Search index only allows field names starting with letters. "
                        f"Invalid key: {key} will be dropped."
                    )
                    logger.warning(msg)
                    continue

                field_type = type_mapping.get(value)
                if not field_type:
                    error_message = f"Unsupported field type for key '{key}': {value}"
                    raise ValueError(error_message)

                normalized_fields[key] = SimpleField(
                    name=key,
                    type=field_type,
                    filterable=True,
                )

        return normalized_fields

    def _create_index(self) -> None:
        """
        Internally creates a new search index.
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

        if self._metadata_fields:
            default_fields.extend(self._metadata_fields.values())

        index = SearchIndex(
            name=self._index_name,
            fields=default_fields,
            vector_search=self._vector_search_configuration,
            **self._index_creation_kwargs,
        )

        if self._index_client:
            self._index_client.create_index(index)

    @staticmethod
    def _serialize_index_creation_kwargs(
        index_creation_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Serializes the index creation kwargs to a dictionary.
        This is needed to handle serialization of Azure AI Search classes
        that are passed in the index creation kwargs.
        """
        result = {}
        for key, value in index_creation_kwargs.items():
            if hasattr(value, "as_dict"):
                result[key] = value.as_dict()
            elif isinstance(value, list) and all(hasattr(item, "as_dict") for item in value):
                result[key] = [item.as_dict() for item in value]
            else:
                result[key] = value
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            azure_endpoint=(self._azure_endpoint.to_dict() if self._azure_endpoint else None),
            api_key=self._api_key.to_dict() if self._api_key else None,
            index_name=self._index_name,
            embedding_dimension=self._embedding_dimension,
            metadata_fields={key: value.as_dict() for key, value in self._metadata_fields.items()},
            vector_search_configuration=self._vector_search_configuration.as_dict(),
            **self._serialize_index_creation_kwargs(self._index_creation_kwargs),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AzureAISearchDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.

        :returns:
            Deserialized component.
        """
        if (fields := data["init_parameters"]["metadata_fields"]) is not None:
            data["init_parameters"]["metadata_fields"] = {
                key: SearchField.from_dict(field) for key, field in fields.items()
            }
        else:
            data["init_parameters"]["metadata_fields"] = {}

        for key, model_class in AZURE_CLASS_MAPPING.items():
            if key in data["init_parameters"]:
                value = data["init_parameters"][key]
                if isinstance(value, list):
                    data["init_parameters"][key] = [model_class.from_dict(item) for item in value]
                else:
                    data["init_parameters"][key] = model_class.from_dict(value)

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

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes the provided documents to search index.

        :param documents: documents to write to the index.
        :param policy: Policy to determine how duplicates are handled.
        :raises ValueError: If the documents are not of type Document.
        :raises TypeError: If the document ids are not strings.
        :return: the number of documents added to index.
        """

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
        documents_to_write = [self._convert_haystack_document_to_azure(doc) for doc in documents]

        if documents_to_write != []:
            client.upload_documents(documents_to_write)
        return len(documents_to_write)

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the search index.

        :param document_ids: ids of the documents to be deleted.
        """
        if self.count_documents() == 0:
            return
        documents = self._get_raw_documents_by_id(document_ids)
        if documents:
            self.client.delete_documents(documents)

    def delete_all_documents(self, recreate_index: bool = False) -> None:  # noqa: FBT002, FBT001
        """
        Deletes all documents in the document store.

        :param recreate_index: If True, the index will be deleted and recreated with the original schema.
            If False, all documents will be deleted while preserving the index.
        """
        try:
            if recreate_index:
                # Get current index definition
                if self._index_client is None:
                    msg = "Index client is not initialized"
                    raise ValueError(msg)
                current_index = self._index_client.get_index(self._index_name)

                # Delete and recreate index
                self._index_client.delete_index(self._index_name)
                self._index_client.create_index(current_index)
                logger.info("Index '{idx_name}' recreated with original schema.", idx_name=self._index_name)
            else:
                # Delete all documents without recreating index
                if self.count_documents() == 0:
                    return

                # Search for all documents (pagination handled by Azure SDK)
                all_docs = list(self.client.search(search_text="*", select=["id"], top=100000))

                if all_docs:
                    self.client.delete_documents(all_docs)
                    logger.info(
                        "Deleted {n_docs} documents from index '{idx_name}'.",
                        n_docs=len(all_docs),
                        idx_name=self._index_name,
                    )
        except Exception as e:
            msg = f"Failed to delete all documents from Azure AI Search: {e!s}"
            raise HttpResponseError(msg) from e

    def get_documents_by_id(self, document_ids: list[str]) -> list[Document]:
        return self._convert_search_result_to_documents(self._get_raw_documents_by_id(document_ids))

    def search_documents(self, search_text: str = "*", top_k: int = 10) -> list[Document]:
        """
        Returns all documents that match the provided search_text.
        If search_text is None, returns all documents.
        :param search_text: the text to search for in the Document list.
        :param top_k: Maximum number of documents to return.
        :returns: A list of Documents that match the given search_text.
        """
        result = self.client.search(search_text=search_text, top=top_k)
        return self._convert_search_result_to_documents(list(result))

    def filter_documents(self, filters: Optional[dict[str, Any]] = None) -> list[Document]:
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
            return self._convert_search_result_to_documents(list(result))
        else:
            return self.search_documents()

    def _convert_search_result_to_documents(self, azure_docs: list[dict[str, Any]]) -> list[Document]:
        """
        Converts Azure search results to Haystack Documents.
        """
        documents = []

        for azure_doc in azure_docs:
            embedding = azure_doc.get("embedding")
            score = azure_doc.get("@search.score", None)
            if embedding == self._dummy_vector:
                embedding = None
            meta = {}

            # Anything besides default fields (id, content, and embedding) is considered metadata
            if self._include_search_metadata:
                meta = {key: value for key, value in azure_doc.items() if key not in ["id", "content", "embedding"]}
            else:
                meta = {
                    key: value
                    for key, value in azure_doc.items()
                    if key not in ["id", "content", "embedding"] and key in self._index_fields and value is not None
                }

            # Create the document with meta only if it's non-empty
            doc = Document(
                id=azure_doc["id"],
                content=azure_doc["content"],
                embedding=embedding,
                meta=meta,
                score=score,
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

    def _get_raw_documents_by_id(self, document_ids: list[str]) -> list[dict]:
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

    def _convert_haystack_document_to_azure(self, document: Document) -> dict[str, Any]:
        """Convert a Haystack Document to an Azure Search document"""

        doc_dict = document.to_dict(flatten=False)

        # Because Azure Search does not allow dynamic fields, we only include fields that are part of the schema
        index_document = {k: v for k, v in {**doc_dict, **doc_dict.get("meta", {})}.items() if k in self._index_fields}
        if index_document["embedding"] is None:
            index_document["embedding"] = self._dummy_vector

        return index_document

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> list[Document]:
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
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> list[Document]:
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
        query_embedding: list[float],
        top_k: int = 10,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> list[Document]:
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
