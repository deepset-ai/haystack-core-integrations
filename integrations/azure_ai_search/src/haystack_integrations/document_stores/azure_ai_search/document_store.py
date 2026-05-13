# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging as python_logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from azure.core.credentials import AzureKeyCredential, TokenCredential
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
from azure.search.documents.models import LookupDocument, VectorizedQuery
from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.misc import _normalize_metadata_field_name

from .errors import AzureAISearchDocumentStoreConfigError, AzureAISearchDocumentStoreError
from .filters import _normalize_filters

USER_AGENT = "haystack-integrations/azure-ai-search"
BIG_TOP_K = 100000

type_mapping = {
    str: "Edm.String",
    bool: "Edm.Boolean",
    int: "Edm.Int32",
    float: "Edm.Double",
    datetime: "Edm.DateTimeOffset",
}


def _instantiate_azure_model(model_class: Any, data: Any) -> Any:
    """Instantiate an Azure SDK model from a dict, picking the right subclass if the model has subtypes."""
    # Some Azure base classes (e.g. LexicalAnalyzer) have multiple subclasses (e.g. CustomAnalyzer);
    # the concrete subclass to use is named in the "@odata.type" field of the dict, and the base class
    # exposes a __mapping__ from that name to the subclass.
    if isinstance(data, Mapping):
        subtype_name = data.get("@odata.type")
        subtypes = getattr(model_class, "__mapping__", {})
        if subtype_name in subtypes:
            return subtypes[subtype_name](data)
    return model_class(data)


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

SPECIAL_FIELDS = {"id", "embedding"}
FIELD_TYPE_MAPPING = {
    "Edm.String": "keyword",
    "Edm.Boolean": "boolean",
    "Edm.Int16": "long",
    "Edm.Int32": "long",
    "Edm.Int64": "long",
    "Edm.Single": "double",
    "Edm.Double": "double",
    "Edm.DateTimeOffset": "date",
}


class AzureAISearchDocumentStore:
    """
    Document store using [Azure AI Search](https://azure.microsoft.com/products/ai-services/ai-search/) as the backend.
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("AZURE_AI_SEARCH_API_KEY", strict=False),  # noqa: B008
        azure_endpoint: Secret = Secret.from_env_var("AZURE_AI_SEARCH_ENDPOINT", strict=True),  # noqa: B008
        index_name: str = "default",
        embedding_dimension: int = 768,
        metadata_fields: dict[str, SearchField | type] | None = None,
        vector_search_configuration: VectorSearch | None = None,
        include_search_metadata: bool = False,
        azure_token_credential: TokenCredential | None = None,
        **index_creation_kwargs: Any,
    ) -> None:
        """
        Creates a new instance of AzureAISearchDocumentStore.

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
        :param azure_token_credential: An Azure `TokenCredential` instance used to authenticate requests.
            When provided, this takes priority over `api_key`.
        :param index_creation_kwargs: Optional keyword parameters to be passed to `SearchIndex` class
            during index creation. Some of the supported parameters:
                - `semantic_search`: Defines semantic configuration of the search index. This parameter is needed
                to enable semantic search capabilities in index.
                - `similarity`: The type of similarity algorithm to be used when scoring and ranking the documents
                matching a search query. The similarity algorithm can only be defined at index creation time and
                cannot be modified on existing indexes.

        For more information on parameters, see the [official Azure AI Search documentation](https://learn.microsoft.com/en-us/azure/search/).
        """
        self._client: SearchClient | None = None
        self._index_client: SearchIndexClient | None = None
        self._index_fields = []  # type: list[Any]  # stores all fields in the final schema of index
        self._api_key = api_key
        self._azure_endpoint = azure_endpoint
        self._index_name = index_name
        self._embedding_dimension = embedding_dimension
        self._dummy_vector = [-10.0] * self._embedding_dimension
        self._metadata_fields = AzureAISearchDocumentStore._normalize_metadata_index_fields(metadata_fields)
        self._vector_search_configuration = vector_search_configuration or DEFAULT_VECTOR_SEARCH
        self._include_search_metadata = include_search_metadata
        self._azure_token_credential = azure_token_credential
        self._index_creation_kwargs = index_creation_kwargs

    @property
    def client(self) -> SearchClient:
        """Return the Azure SearchClient, creating the index if it does not exist."""
        resolved_endpoint = self._azure_endpoint.resolve_value()
        resolved_key = self._api_key.resolve_value()

        if self._azure_token_credential is not None:
            credential: TokenCredential | AzureKeyCredential = self._azure_token_credential
        elif resolved_key:
            credential = AzureKeyCredential(resolved_key)
        else:
            credential = DefaultAzureCredential()

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

    @staticmethod
    def _normalize_metadata_index_fields(
        metadata_fields: dict[str, SearchField | type] | None,
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
            SimpleField(name="id", type=SearchFieldDataType.STRING, key=True, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.STRING),
            SearchField(
                name="embedding",
                type=f"Collection({SearchFieldDataType.SINGLE.value})",
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
        if self._azure_token_credential:
            logger.warning(
                "AzureAISearchDocumentStore was initialized with `azure_token_credential`, "
                "which cannot be serialized. It will be excluded from the serialized output "
                "and must be provided again when deserializing."
            )

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
            data["init_parameters"]["metadata_fields"] = {key: SearchField(field) for key, field in fields.items()}
        else:
            data["init_parameters"]["metadata_fields"] = {}

        for key, model_class in AZURE_CLASS_MAPPING.items():
            if key in data["init_parameters"]:
                value = data["init_parameters"][key]
                if isinstance(value, list):
                    data["init_parameters"][key] = [_instantiate_azure_model(model_class, item) for item in value]
                else:
                    data["init_parameters"][key] = _instantiate_azure_model(model_class, value)

        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key", "azure_endpoint"])
        if (vector_search_configuration := data["init_parameters"].get("vector_search_configuration")) is not None:
            data["init_parameters"]["vector_search_configuration"] = VectorSearch(vector_search_configuration)
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the search index.

        :returns: list of retrieved documents.
        """
        return self.client.get_document_count()

    def _get_index_schema_fields(self) -> dict[str, Any]:
        """
        Returns the index schema fields keyed by field name.
        """
        _ = self.client
        if self._index_client is None:
            msg = "Search Index Client is not initialized."
            raise AzureAISearchDocumentStoreConfigError(msg)

        index_fields = self._index_client.get_index(self._index_name).fields
        return {field.name: field for field in index_fields}

    def _validate_index_fields(self, field_names: list[str]) -> None:
        """
        Validates that all provided field names exist in the index schema.
        """
        if not self._index_fields:
            self._index_fields = list(self._get_index_schema_fields().keys())

        missing_fields = [field for field in field_names if field not in self._index_fields]
        if missing_fields:
            msg = f"Fields {missing_fields} are not defined in index schema. Available fields: {self._index_fields}"
            raise ValueError(msg)

    def _fetch_raw_documents(
        self,
        *,
        filters: dict[str, Any] | None = None,
        select: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetches raw Azure documents, optionally filtered and projected to selected fields.
        """
        normalized_filters = _normalize_filters(filters) if filters else None
        result = self.client.search(search_text="*", filter=normalized_filters, select=select, top=BIG_TOP_K)
        return list(result)

    @staticmethod
    def _collect_unique_values(documents: list[dict[str, Any]], field_name: str) -> set[Any]:
        """
        Collects unique field values from Azure documents.
        """
        unique_values: set[Any] = set()
        for document in documents:
            value = document.get(field_name)
            if isinstance(value, list):
                unique_values.update(value)
            elif value is not None:
                unique_values.add(value)
        return unique_values

    @staticmethod
    def _get_min_max_from_documents(documents: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
        """
        Computes min and max values for a field from Azure documents.
        """
        values: list[bool | int | float | str | datetime] = []
        for document in documents:
            value = document.get(field_name)
            if isinstance(value, bool | int | float | str | datetime):
                values.append(value)

        if not values:
            return {"min": None, "max": None}

        return {"min": min(values), "max": max(values)}

    @staticmethod
    def _map_azure_field_type(field: Any) -> str:
        """
        Maps Azure Search field definitions to Haystack metadata field types.
        """
        field_type = getattr(field, "type", None)
        if field.name == "content":
            return "text"
        if field_type is None:
            return "keyword"

        field_type_name = field_type.value if hasattr(field_type, "value") else str(field_type)
        if field_type_name.startswith("Collection("):
            inner_type = field_type_name[len("Collection(") : -1]
            return FIELD_TYPE_MAPPING.get(inner_type, "keyword")

        if field_type_name == "Edm.String" and getattr(field, "searchable", False):
            return "text"

        return FIELD_TYPE_MAPPING.get(field_type_name, "keyword")

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Returns the count of documents that match the provided filters.

        :param filters: The filters to apply to the document list.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents that match the filters.
        """
        normalized_filters = _normalize_filters(filters)
        result = self.client.search(
            search_text="*",
            filter=normalized_filters,
            select=["id"],
            top=1,
            include_total_count=True,
        )
        count = result.get_count()
        return count if count is not None else len(self._fetch_raw_documents(filters=filters, select=["id"]))

    def count_unique_metadata_by_filter(self, filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]:
        """
        Counts unique values for each specified metadata field in documents matching the filters.

        :param filters: The filters to apply to select documents.
        :param metadata_fields: List of field names to count unique values for.
        :returns: Dictionary mapping field names to counts of unique values.
        """
        normalized_metadata_fields = [_normalize_metadata_field_name(field) for field in metadata_fields]
        self._validate_index_fields(normalized_metadata_fields)

        documents = self._fetch_raw_documents(filters=filters, select=normalized_metadata_fields)
        return {
            field_name: len(self._collect_unique_values(documents, field_name))
            for field_name in normalized_metadata_fields
        }

    def get_metadata_fields_info(self) -> dict[str, dict[str, str]]:
        """
        Returns the information about metadata fields in the index.

        :returns: Dictionary mapping field names to type information.
        """
        schema_fields = self._get_index_schema_fields()
        return {
            name: {"type": self._map_azure_field_type(field)}
            for name, field in schema_fields.items()
            if name not in SPECIAL_FIELDS
        }

    def get_metadata_field_min_max(self, metadata_field: str) -> dict[str, Any]:
        """
        Returns the minimum and maximum values for the given metadata field.

        :param metadata_field: The metadata field to get the minimum and maximum values for.
        :returns: A dictionary with the keys "min" and "max".
        """
        field_name = _normalize_metadata_field_name(metadata_field)
        self._validate_index_fields([field_name])

        documents = self._fetch_raw_documents(select=[field_name])
        return self._get_min_max_from_documents(documents, field_name)

    def get_metadata_field_unique_values(
        self, metadata_field: str, search_term: str | None = None, from_: int = 0, size: int = 10
    ) -> tuple[list[str], int]:
        """
        Retrieves unique values for a metadata field with optional search and pagination.

        :param metadata_field: The metadata field to get unique values for.
        :param search_term: Optional search term to filter unique values.
        :param from_: Starting offset for pagination.
        :param size: Number of values to return.
        :returns: Tuple of (list of unique values, total count of matching values).
        """
        field_name = _normalize_metadata_field_name(metadata_field)
        self._validate_index_fields([field_name])

        documents = self._fetch_raw_documents(select=[field_name])
        unique_values = sorted(str(value) for value in self._collect_unique_values(documents, field_name))

        if search_term:
            normalized_search_term = search_term.lower()
            unique_values = [value for value in unique_values if normalized_search_term in value.lower()]

        total_count = len(unique_values)
        return unique_values[from_ : from_ + size], total_count

    def query_sql(self, query: str) -> Any:
        """
        Executes an SQL query if supported by the document store backend.

        Azure AI Search does not support SQL queries.
        """
        msg = f"Azure AI Search does not support SQL queries: {query}"
        raise NotImplementedError(msg)

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
            self.client.delete_documents([dict(doc) for doc in documents])

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
                all_docs = list(self.client.search(search_text="*", select=["id"], top=BIG_TOP_K))

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

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Deletes all documents that match the provided filters.

        Azure AI Search does not support server-side delete by query, so this method
        first searches for matching documents, then deletes them in a batch operation.

        :param filters: The filters to apply to select documents for deletion.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents deleted.
        """
        try:
            normalized_filters = _normalize_filters(filters)

            results = list(self.client.search(search_text="*", filter=normalized_filters, select=["id"], top=BIG_TOP_K))

            if not results:
                return 0

            documents_to_delete = [{"id": doc["id"]} for doc in results]
            self.client.delete_documents(documents=documents_to_delete)

            return len(documents_to_delete)

        except Exception as e:
            msg = f"Failed to delete documents by filter from Azure AI Search: {e!s}"
            raise AzureAISearchDocumentStoreError(msg) from e

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Updates the fields of all documents that match the provided filters.

        Azure AI Search does not support server-side update by query, so this method
        first searches for matching documents, then updates them using merge operations.

        :param filters: The filters to apply to select documents for updating.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param meta: The fields to update. These fields must exist in the index schema.
        :returns: The number of documents updated.
        """
        try:
            # validate that fields to update exist in the index schema
            invalid_fields = [key for key in meta.keys() if key not in self._index_fields]
            if invalid_fields:
                msg = f"Fields {invalid_fields} are not defined in index schema. Available fields: {self._index_fields}"
                raise ValueError(msg)

            normalized_filters = _normalize_filters(filters)

            results = list(self.client.search(search_text="*", filter=normalized_filters, select=["id"], top=BIG_TOP_K))

            if not results:
                return 0

            documents_to_update = []
            for doc in results:
                update_doc = {"id": doc["id"]}
                update_doc.update(meta)
                documents_to_update.append(update_doc)

            self.client.merge_documents(documents=documents_to_update)

            logger.info(
                "Updated {n_docs} documents in index '{idx_name}' using filters.",
                n_docs=len(documents_to_update),
                idx_name=self._index_name,
            )
            return len(documents_to_update)

        except ValueError:
            # Re-raise ValueError for invalid fields without wrapping
            raise
        except Exception as e:
            msg = f"Failed to delete documents by filter from Azure AI Search: {e!s}"
            raise AzureAISearchDocumentStoreError(msg) from e

    def get_documents_by_id(self, document_ids: list[str]) -> list[Document]:
        """
        Retrieves documents by their IDs.

        :param document_ids: IDs of the documents to retrieve.
        :returns: List of documents with the given IDs.
        """
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

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
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

    def _convert_search_result_to_documents(self, azure_docs: Sequence[Mapping[str, Any]]) -> list[Document]:
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

    def _index_exists(self, index_name: str | None) -> bool:
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

    def _get_raw_documents_by_id(self, document_ids: list[str]) -> list[LookupDocument]:
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
        filters: str | None = None,
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
        filters: str | None = None,
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
        filters: str | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Retrieve documents using vector search combined with the BM25 algorithm.

        This method combines vector similarity and BM25 for improved retrieval.

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
