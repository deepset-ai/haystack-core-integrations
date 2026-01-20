# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import datetime
import json
from dataclasses import asdict
from typing import Any

from haystack import logging
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types.policy import DuplicatePolicy

import weaviate
from weaviate.collections.classes.aggregate import GroupByAggregate, Metrics
from weaviate.collections.classes.data import DataObject
from weaviate.config import AdditionalConfig
from weaviate.embedded import EmbeddedOptions
from weaviate.util import generate_uuid5

from ._filters import convert_filters, validate_filters
from .auth import AuthCredentials

logger = logging.getLogger(__name__)


# This is the default collection properties for Weaviate.
# It's a list of properties that will be created on the collection.
# These are extremely similar to the Document dataclass, but with a few differences:
# - `id` is renamed to `_original_id` as the `id` field is reserved by Weaviate.
# - `blob` is split into `blob_data` and `blob_mime_type` as it's more efficient to store them separately.
# Blob meta is missing as it's not usually serialized when saving a Document as we rely on the Document own meta.
#
# Also the Document `meta` fields are omitted as we can't make assumptions on the structure of the meta field.
# We recommend the user to create a proper collection with the correct meta properties for their use case.
# We mostly rely on these defaults for testing purposes using Weaviate automatic schema generation, but that's not
# recommended for production use.
DOCUMENT_COLLECTION_PROPERTIES = [
    {"name": "_original_id", "dataType": ["text"]},
    {"name": "content", "dataType": ["text"]},
    {"name": "blob_data", "dataType": ["blob"]},
    {"name": "blob_mime_type", "dataType": ["text"]},
    {"name": "score", "dataType": ["number"]},
]

# This is the default limit used when querying documents with WeaviateDocumentStore.
#
# We picked this as QUERY_MAXIMUM_RESULTS defaults to 10000, trying to get that many
# documents at once will fail, even if the query is paginated.
# This value will ensure we get the most documents possible without hitting that limit, it would
# still fail if the user lowers the QUERY_MAXIMUM_RESULTS environment variable for their Weaviate instance.
#
# See WeaviateDocumentStore._query_with_filters() for more information.
DEFAULT_QUERY_LIMIT = 9999


class WeaviateDocumentStore:
    """
    A WeaviateDocumentStore instance you
    can use with Weaviate Cloud Services or self-hosted instances.

    Usage example with Weaviate Cloud Services:
    ```python
    import os
    from haystack_integrations.document_stores.weaviate.auth import AuthApiKey
    from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore

    os.environ["WEAVIATE_API_KEY"] = "MY_API_KEY"

    document_store = WeaviateDocumentStore(
        url="rAnD0mD1g1t5.something.weaviate.cloud",
        auth_client_secret=AuthApiKey(),
    )
    ```

    Usage example with self-hosted Weaviate:
    ```python
    from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore

    document_store = WeaviateDocumentStore(url="http://localhost:8080")
    ```
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        collection_settings: dict[str, Any] | None = None,
        auth_client_secret: AuthCredentials | None = None,
        additional_headers: dict | None = None,
        embedded_options: EmbeddedOptions | None = None,
        additional_config: AdditionalConfig | None = None,
        grpc_port: int = 50051,
        grpc_secure: bool = False,
    ):
        """
        Create a new instance of WeaviateDocumentStore and connects to the Weaviate instance.

        :param url:
            The URL to the weaviate instance.
        :param collection_settings:
            The collection settings to use. If `None`, it will use a collection named `default` with the following
            properties:
            - _original_id: text
            - content: text
            - blob_data: blob
            - blob_mime_type: text
            - score: number
            The Document `meta` fields are omitted in the default collection settings as we can't make assumptions
            on the structure of the meta field.
            We heavily recommend to create a custom collection with the correct meta properties
            for your use case.
            Another option is relying on the automatic schema generation, but that's not recommended for
            production use.
            See the official [Weaviate documentation](https://weaviate.io/developers/weaviate/manage-data/collections)
            for more information on collections and their properties.
        :param auth_client_secret:
            Authentication credentials. Can be one of the following types depending on the authentication mode:
            - `AuthBearerToken` to use existing access and (optionally, but recommended) refresh tokens
            - `AuthClientPassword` to use username and password for oidc Resource Owner Password flow
            - `AuthClientCredentials` to use a client secret for oidc client credential flow
            - `AuthApiKey` to use an API key
        :param additional_headers:
            Additional headers to include in the requests. Can be used to set OpenAI/HuggingFace keys.
            OpenAI/HuggingFace key looks like this:
            ```
            {"X-OpenAI-Api-Key": "<THE-KEY>"}, {"X-HuggingFace-Api-Key": "<THE-KEY>"}
            ```
        :param embedded_options:
            If set, create an embedded Weaviate cluster inside the client. For a full list of options see
            `weaviate.embedded.EmbeddedOptions`.
        :param additional_config:
            Additional and advanced configuration options for weaviate.
        :param grpc_port:
            The port to use for the gRPC connection.
        :param grpc_secure:
            Whether to use a secure channel for the underlying gRPC API.
        """
        self._url = url
        self._auth_client_secret = auth_client_secret
        self._additional_headers = additional_headers
        self._embedded_options = embedded_options
        self._additional_config = additional_config
        self._grpc_port = grpc_port
        self._grpc_secure = grpc_secure
        self._client: weaviate.WeaviateClient | None = None
        self._async_client: weaviate.WeaviateAsyncClient | None = None
        self._collection: weaviate.Collection | None = None
        self._async_collection: weaviate.AsyncCollection | None = None
        # Store the connection settings dictionary
        self._collection_settings = collection_settings or {
            "class": "Default",
            "invertedIndexConfig": {"indexNullState": True},
            "properties": DOCUMENT_COLLECTION_PROPERTIES,
        }
        self._clean_connection_settings()

    def _clean_connection_settings(self):
        # Set the class if not set
        _class_name = self._collection_settings.get("class", "Default")
        _class_name = _class_name[0].upper() + _class_name[1:]
        self._collection_settings["class"] = _class_name
        # Set the properties if they're not set
        self._collection_settings["properties"] = self._collection_settings.get(
            "properties", DOCUMENT_COLLECTION_PROPERTIES
        )

    @property
    def client(self):
        if self._client:
            return self._client

        if self._url and self._url.endswith((".weaviate.network", ".weaviate.cloud")):
            # If we detect that the URL is a Weaviate Cloud URL, we use the utility function to connect
            # instead of using WeaviateClient directly like in other cases.
            # Among other things, the utility function takes care of parsing the URL.
            if not self._auth_client_secret:
                msg = "Auth credentials are required for Weaviate Cloud Services"
                raise ValueError(msg)
            self._client = weaviate.connect_to_weaviate_cloud(
                self._url,
                auth_credentials=self._auth_client_secret.resolve_value(),
                headers=self._additional_headers,
                additional_config=self._additional_config,
            )
        else:
            # Embedded, local Docker deployment or custom connection.
            # proxies, timeout_config, trust_env are part of additional_config now
            # startup_period has been removed
            self._client = weaviate.WeaviateClient(
                connection_params=(
                    weaviate.connect.base.ConnectionParams.from_url(
                        url=self._url, grpc_port=self._grpc_port, grpc_secure=self._grpc_secure
                    )
                    if self._url
                    else None
                ),
                auth_client_secret=self._auth_client_secret.resolve_value() if self._auth_client_secret else None,
                additional_config=self._additional_config,
                additional_headers=self._additional_headers,
                embedded_options=self._embedded_options,
                skip_init_checks=False,
            )

        self._client.connect()

        # Test connection, it will raise an exception if it fails.
        self._client.collections.list_all(simple=True)
        if not self._client.collections.exists(self._collection_settings["class"]):
            self._client.collections.create_from_dict(self._collection_settings)

        return self._client

    @property
    async def async_client(self):
        if self._async_client:
            return self._async_client

        if self._url and self._url.endswith((".weaviate.network", ".weaviate.cloud")):
            # If we detect that the URL is a Weaviate Cloud URL, we use the utility function to connect
            # instead of using WeaviateAsyncClient directly like in other cases.
            # Among other things, the utility function takes care of parsing the URL.
            if not self._auth_client_secret:
                msg = "Auth credentials are required for Weaviate Cloud Services"
                raise ValueError(msg)
            self._async_client = weaviate.use_async_with_weaviate_cloud(
                self._url,
                auth_credentials=self._auth_client_secret.resolve_value(),
                headers=self._additional_headers,
                additional_config=self._additional_config,
            )
        else:
            # Embedded, local Docker deployment or custom connection.
            # proxies, timeout_config, trust_env are part of additional_config now
            # startup_period has been removed
            self._async_client = weaviate.WeaviateAsyncClient(
                connection_params=(
                    weaviate.connect.base.ConnectionParams.from_url(
                        url=self._url, grpc_port=self._grpc_port, grpc_secure=self._grpc_secure
                    )
                    if self._url
                    else None
                ),
                auth_client_secret=self._auth_client_secret.resolve_value() if self._auth_client_secret else None,
                additional_config=self._additional_config,
                additional_headers=self._additional_headers,
                embedded_options=self._embedded_options,
                skip_init_checks=False,
            )

        await self._async_client.connect()
        # Test connection, it will raise an exception if it fails.
        await self._async_client.collections.list_all(simple=True)
        if not await self._async_client.collections.exists(self._collection_settings["class"]):
            await self._async_client.collections.create_from_dict(self._collection_settings)

        return self._async_client

    @property
    def collection(self):
        if self._collection:
            return self._collection

        client = self.client
        self._collection = client.collections.get(self._collection_settings["class"])
        return self._collection

    @property
    async def async_collection(self):
        if self._async_collection:
            return self._async_collection

        async_client = await self.async_client
        self._async_collection = async_client.collections.get(self._collection_settings["class"])
        return self._async_collection

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        embedded_options = asdict(self._embedded_options) if self._embedded_options else None
        additional_config = (
            json.loads(self._additional_config.model_dump_json(by_alias=True)) if self._additional_config else None
        )

        return default_to_dict(
            self,
            url=self._url,
            collection_settings=self._collection_settings,
            auth_client_secret=self._auth_client_secret.to_dict() if self._auth_client_secret else None,
            additional_headers=self._additional_headers,
            embedded_options=embedded_options,
            additional_config=additional_config,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeaviateDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        if (auth_client_secret := data["init_parameters"].get("auth_client_secret")) is not None:
            data["init_parameters"]["auth_client_secret"] = AuthCredentials.from_dict(auth_client_secret)
        if (embedded_options := data["init_parameters"].get("embedded_options")) is not None:
            data["init_parameters"]["embedded_options"] = EmbeddedOptions(**embedded_options)
        if (additional_config := data["init_parameters"].get("additional_config")) is not None:
            data["init_parameters"]["additional_config"] = AdditionalConfig(**additional_config)
        return default_from_dict(
            cls,
            data,
        )

    def count_documents(self) -> int:
        """
        Returns the number of documents present in the DocumentStore.
        """
        total = self.collection.aggregate.over_all(total_count=True).total_count
        return total if total else 0

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Returns the number of documents that match the provided filters.

        :param filters: The filters to apply to count documents.
            For filter syntax, see
            [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering).
        :returns: The number of documents that match the filters.
        """
        validate_filters(filters)
        weaviate_filter = convert_filters(filters)
        total = self.collection.aggregate.over_all(filters=weaviate_filter, total_count=True).total_count
        return total if total else 0

    async def count_documents_by_filter_async(self, filters: dict[str, Any]) -> int:
        """
        Asynchronously returns the number of documents that match the provided filters.

        :param filters: The filters to apply to count documents.
            For filter syntax, see
            [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering).
        :returns: The number of documents that match the filters.
        """
        validate_filters(filters)
        collection = await self.async_collection
        weaviate_filter = convert_filters(filters)
        result = await collection.aggregate.over_all(filters=weaviate_filter, total_count=True)
        return result.total_count if result.total_count else 0

    def get_metadata_fields_info(self) -> dict[str, dict[str, str]]:
        """
        Returns metadata field names and their types, excluding special fields.

        Special fields (content, blob_data, blob_mime_type, _original_id, score) are excluded
        as they are not user metadata fields.

        :returns: A dictionary where keys are field names and values are dictionaries
            containing type information, e.g., {"category": {"type": "text"}}.
        """
        config = self.collection.config.get()
        special_fields = {prop["name"] for prop in DOCUMENT_COLLECTION_PROPERTIES}
        fields_info = {}
        for prop in config.properties:
            if prop.name not in special_fields:
                data_type = str(prop.data_type.value) if hasattr(prop.data_type, "value") else str(prop.data_type)
                fields_info[prop.name] = {"type": data_type}
        return fields_info

    async def get_metadata_fields_info_async(self) -> dict[str, dict[str, str]]:
        """
        Asynchronously returns metadata field names and their types, excluding special fields.

        Special fields (content, blob_data, blob_mime_type, _original_id, score) are excluded
        as they are not user metadata fields.

        :returns: A dictionary where keys are field names and values are dictionaries
            containing type information, e.g., {"category": {"type": "text"}}.
        """
        collection = await self.async_collection
        config = await collection.config.get()
        special_fields = {prop["name"] for prop in DOCUMENT_COLLECTION_PROPERTIES}
        fields_info = {}
        for prop in config.properties:
            if prop.name not in special_fields:
                data_type = str(prop.data_type.value) if hasattr(prop.data_type, "value") else str(prop.data_type)
                fields_info[prop.name] = {"type": data_type}
        return fields_info

    @staticmethod
    def _normalize_metadata_field_name(metadata_field: str) -> str:
        """
        Removes 'meta.' prefix from field name if present.

        :param metadata_field: The field name, possibly prefixed with 'meta.'.
        :returns: The field name without the 'meta.' prefix.
        """
        return metadata_field[5:] if metadata_field.startswith("meta.") else metadata_field

    def get_metadata_field_min_max(self, metadata_field: str) -> dict[str, Any]:
        """
        Returns the minimum and maximum values for a numeric or date metadata field.

        :param metadata_field: The metadata field name to get min/max for.
            Can be prefixed with 'meta.' (e.g., 'meta.year' or 'year').
        :returns: A dictionary with 'min' and 'max' keys containing the respective values.
        :raises ValueError: If the field is not found or doesn't support min/max operations.
        """
        field_name = self._normalize_metadata_field_name(metadata_field)

        # Get field type from schema
        config = self.collection.config.get()
        field_type = None
        for prop in config.properties:
            if prop.name == field_name:
                field_type = prop.data_type
                break

        if field_type is None:
            msg = f"Field '{field_name}' not found in collection schema"
            raise ValueError(msg)

        data_type_str = str(field_type.value) if hasattr(field_type, "value") else str(field_type)

        # Build metrics based on type
        if data_type_str.lower() == "int":
            metrics = Metrics(field_name).integer(minimum=True, maximum=True)
        elif data_type_str.lower() == "number":
            metrics = Metrics(field_name).number(minimum=True, maximum=True)
        elif data_type_str.lower() == "date":
            metrics = Metrics(field_name).date_(minimum=True, maximum=True)
        else:
            msg = f"Field type '{data_type_str}' doesn't support min/max aggregation"
            raise ValueError(msg)

        result = self.collection.aggregate.over_all(return_metrics=metrics)
        field_metrics = result.properties.get(field_name)

        return {
            "min": getattr(field_metrics, "minimum", None) if field_metrics else None,
            "max": getattr(field_metrics, "maximum", None) if field_metrics else None,
        }

    async def get_metadata_field_min_max_async(self, metadata_field: str) -> dict[str, Any]:
        """
        Asynchronously returns the minimum and maximum values for a numeric or date metadata field.

        :param metadata_field: The metadata field name to get min/max for.
            Can be prefixed with 'meta.' (e.g., 'meta.year' or 'year').
        :returns: A dictionary with 'min' and 'max' keys containing the respective values.
        :raises ValueError: If the field is not found or doesn't support min/max operations.
        """
        field_name = self._normalize_metadata_field_name(metadata_field)

        # Get field type from schema
        collection = await self.async_collection
        config = await collection.config.get()
        field_type = None
        for prop in config.properties:
            if prop.name == field_name:
                field_type = prop.data_type
                break

        if field_type is None:
            msg = f"Field '{field_name}' not found in collection schema"
            raise ValueError(msg)

        data_type_str = str(field_type.value) if hasattr(field_type, "value") else str(field_type)

        # Build metrics based on type
        if data_type_str.lower() == "int":
            metrics = Metrics(field_name).integer(minimum=True, maximum=True)
        elif data_type_str.lower() == "number":
            metrics = Metrics(field_name).number(minimum=True, maximum=True)
        elif data_type_str.lower() == "date":
            metrics = Metrics(field_name).date_(minimum=True, maximum=True)
        else:
            msg = f"Field type '{data_type_str}' doesn't support min/max aggregation"
            raise ValueError(msg)

        result = await collection.aggregate.over_all(return_metrics=metrics)
        field_metrics = result.properties.get(field_name)

        return {
            "min": getattr(field_metrics, "minimum", None) if field_metrics else None,
            "max": getattr(field_metrics, "maximum", None) if field_metrics else None,
        }

    def count_unique_metadata_by_filter(
        self, filters: dict[str, Any], metadata_fields: list[str]
    ) -> dict[str, int]:
        """
        Returns the count of unique values for each specified metadata field.

        :param filters: The filters to apply when counting unique values.
            For filter syntax, see
            [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering).
        :param metadata_fields: List of metadata field names to count unique values for.
            Field names can be prefixed with 'meta.' (e.g., 'meta.category' or 'category').
        :returns: A dictionary mapping field names to counts of unique values.
        :raises ValueError: If any of the requested fields don't exist in the collection schema.
        """
        validate_filters(filters)
        weaviate_filter = convert_filters(filters)

        normalized_fields = [self._normalize_metadata_field_name(f) for f in metadata_fields]

        # Validate that all requested fields exist in the schema
        config = self.collection.config.get()
        schema_fields = {prop.name for prop in config.properties}
        missing_fields = [f for f in normalized_fields if f not in schema_fields]
        if missing_fields:
            msg = f"Fields not found in collection schema: {missing_fields}"
            raise ValueError(msg)

        result = {}
        for field in normalized_fields:
            agg_result = self.collection.aggregate.over_all(
                filters=weaviate_filter, group_by=GroupByAggregate(prop=field)
            )
            result[field] = len(agg_result.groups) if agg_result.groups else 0

        return result

    async def count_unique_metadata_by_filter_async(
        self, filters: dict[str, Any], metadata_fields: list[str]
    ) -> dict[str, int]:
        """
        Asynchronously returns the count of unique values for each specified metadata field.

        :param filters: The filters to apply when counting unique values.
            For filter syntax, see
            [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering).
        :param metadata_fields: List of metadata field names to count unique values for.
            Field names can be prefixed with 'meta.' (e.g., 'meta.category' or 'category').
        :returns: A dictionary mapping field names to counts of unique values.
        :raises ValueError: If any of the requested fields don't exist in the collection schema.
        """
        validate_filters(filters)
        collection = await self.async_collection
        weaviate_filter = convert_filters(filters)

        normalized_fields = [self._normalize_metadata_field_name(f) for f in metadata_fields]

        # Validate that all requested fields exist in the schema
        config = await collection.config.get()
        schema_fields = {prop.name for prop in config.properties}
        missing_fields = [f for f in normalized_fields if f not in schema_fields]
        if missing_fields:
            msg = f"Fields not found in collection schema: {missing_fields}"
            raise ValueError(msg)

        result = {}
        for field in normalized_fields:
            agg_result = await collection.aggregate.over_all(
                filters=weaviate_filter, group_by=GroupByAggregate(prop=field)
            )
            result[field] = len(agg_result.groups) if agg_result.groups else 0

        return result

    @staticmethod
    def _to_data_object(document: Document) -> dict[str, Any]:
        """
        Converts a Document to a Weaviate data object ready to be saved.
        """
        data = document.to_dict()
        # Weaviate forces a UUID as an id.
        # We don't know if the id of our Document is a UUID or not, so we save it on a different field
        # and let Weaviate a UUID that we're going to ignore completely.
        data["_original_id"] = data.pop("id")
        if (blob := data.pop("blob")) is not None:
            # Weaviate wants the blob data as a base64 encoded string
            # See the official docs for more information:
            # https://weaviate.io/developers/weaviate/config-refs/datatypes#datatype-blob
            data["blob_data"] = base64.b64encode(bytes(blob.pop("data"))).decode()
            data["blob_mime_type"] = blob.pop("mime_type")
        # The embedding vector is stored separately from the rest of the data
        del data["embedding"]

        # _split_overlap meta field is unsupported because of a bug
        # https://github.com/deepset-ai/haystack-core-integrations/issues/1172
        if "_split_overlap" in data:
            data.pop("_split_overlap")
            logger.warning(
                "Document {id} has the unsupported `_split_overlap` meta field. It will be ignored.",
                id=data["_original_id"],
            )

        if "sparse_embedding" in data:
            sparse_embedding = data.pop("sparse_embedding", None)
            if sparse_embedding:
                logger.warning(
                    "Document {id} has the `sparse_embedding` field set,"
                    "but storing sparse embeddings in Weaviate is not currently supported."
                    "The `sparse_embedding` field will be ignored.",
                    id=data["_original_id"],
                )

        return data

    @staticmethod
    def _to_document(data: DataObject[dict[str, Any], None]) -> Document:
        """
        Converts a data object read from Weaviate into a Document.
        """
        document_data = data.properties
        document_data["id"] = document_data.pop("_original_id")
        if isinstance(data.vector, list):
            document_data["embedding"] = data.vector
        elif isinstance(data.vector, dict):
            document_data["embedding"] = data.vector.get("default")
        else:
            document_data["embedding"] = None

        if (blob_data := document_data.get("blob_data")) is not None:
            document_data["blob"] = {
                "data": base64.b64decode(blob_data),
                "mime_type": document_data.get("blob_mime_type"),
            }

        # We always delete these fields as they're not part of the Document dataclass
        document_data.pop("blob_data", None)
        document_data.pop("blob_mime_type", None)

        for key, value in document_data.items():
            if isinstance(value, datetime.datetime):
                document_data[key] = value.strftime("%Y-%m-%dT%H:%M:%SZ")

        if weaviate_meta := getattr(data, "metadata", None):
            # Depending on the type of retrieval we get score from different fields.
            # score is returned when using BM25 retrieval.
            # certainty is returned when using embedding retrieval.
            if weaviate_meta.score is not None:
                document_data["score"] = weaviate_meta.score
            elif weaviate_meta.certainty is not None:
                document_data["score"] = weaviate_meta.certainty

        return Document.from_dict(document_data)

    def _query(self) -> list[DataObject[dict[str, Any], None]]:
        properties = [p.name for p in self.collection.config.get().properties]
        try:
            result = self.collection.iterator(include_vector=True, return_properties=properties)
        except weaviate.exceptions.WeaviateQueryError as e:
            msg = f"Failed to query documents in Weaviate. Error: {e.message}"
            raise DocumentStoreError(msg) from e
        return result

    def _query_with_filters(self, filters: dict[str, Any]) -> list[DataObject[dict[str, Any], None]]:
        properties = [p.name for p in self.collection.config.get().properties]
        # When querying with filters we need to paginate using limit and offset as using
        # a cursor with after is not possible. See the official docs:
        # https://weaviate.io/developers/weaviate/api/graphql/additional-operators#cursor-with-after
        #
        # Nonetheless there's also another issue, paginating with limit and offset is not efficient
        # and it's still restricted by the QUERY_MAXIMUM_RESULTS environment variable.
        # If the sum of limit and offset is greater than QUERY_MAXIMUM_RESULTS an error is raised.
        # See the official docs for more:
        # https://weaviate.io/developers/weaviate/api/graphql/additional-operators#performance-considerations
        offset = 0
        partial_result = None
        result = []
        # Keep querying until we get all documents matching the filters
        while partial_result is None or len(partial_result.objects) == DEFAULT_QUERY_LIMIT:
            try:
                partial_result = self.collection.query.fetch_objects(
                    filters=convert_filters(filters),
                    include_vector=True,
                    limit=DEFAULT_QUERY_LIMIT,
                    offset=offset,
                    return_properties=properties,
                )
            except weaviate.exceptions.WeaviateQueryError as e:
                msg = f"Failed to query documents in Weaviate. Error: {e.message}"
                raise DocumentStoreError(msg) from e
            result.extend(partial_result.objects)
            offset += DEFAULT_QUERY_LIMIT
        return result

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters, refer to the
        DocumentStore.filter_documents() protocol documentation.

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        validate_filters(filters)

        result = []
        if filters:
            result = self._query_with_filters(filters)
        else:
            result = self._query()
        return [WeaviateDocumentStore._to_document(doc) for doc in result]

    def _batch_write(self, documents: list[Document]) -> int:
        """
        Writes document to Weaviate in batches.
        Documents with the same id will be overwritten.
        Raises in case of errors.
        """

        with self.client.batch.dynamic() as batch:
            for doc in documents:
                if not isinstance(doc, Document):
                    msg = f"Expected a Document, got '{type(doc)}' instead."
                    raise ValueError(msg)

                batch.add_object(
                    properties=WeaviateDocumentStore._to_data_object(doc),
                    collection=self.collection.name,
                    uuid=generate_uuid5(doc.id),
                    vector=doc.embedding,
                )
        if failed_objects := self.client.batch.failed_objects:
            # We fall back to use the UUID if the _original_id is not present, this is just to be
            mapped_objects = {}
            for obj in failed_objects:
                properties = obj.object_.properties or {}
                # We get the object uuid just in case the _original_id is not present.
                # That's extremely unlikely to happen but let's stay on the safe side.
                id_ = properties.get("_original_id", obj.object_.uuid)
                mapped_objects[id_] = obj.message

            msg = "\n".join(
                [
                    f"Failed to write object with id '{id_}'. Error: '{message}'"
                    for id_, message in mapped_objects.items()
                ]
            )
            raise DocumentStoreError(msg)

        # If the document already exists we get no status message back from Weaviate.
        # So we assume that all Documents were written.
        return len(documents)

    def _write(self, documents: list[Document], policy: DuplicatePolicy) -> int:
        """
        Writes documents to Weaviate using the specified policy.
        This doesn't use the batch API, so it's slower than _batch_write.
        If policy is set to SKIP it will skip any document that already exists.
        If policy is set to FAIL it will raise an exception if any of the documents already exists.
        """
        written = 0
        duplicate_errors_ids = []
        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"Expected a Document, got '{type(doc)}' instead."
                raise ValueError(msg)

            if policy == DuplicatePolicy.SKIP and self.collection.data.exists(uuid=generate_uuid5(doc.id)):
                # This Document already exists, we skip it
                continue

            try:
                self.collection.data.insert(
                    uuid=generate_uuid5(doc.id),
                    properties=WeaviateDocumentStore._to_data_object(doc),
                    vector=doc.embedding,
                )

                written += 1
            except weaviate.exceptions.UnexpectedStatusCodeError:
                if policy == DuplicatePolicy.FAIL:
                    duplicate_errors_ids.append(doc.id)
        if duplicate_errors_ids:
            msg = f"IDs '{', '.join(duplicate_errors_ids)}' already exist in the document store."
            raise DuplicateDocumentError(msg)
        return written

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents to Weaviate using the specified policy.
        We recommend using a OVERWRITE policy as it's faster than other policies for Weaviate since it uses
        the batch API.
        We can't use the batch API for other policies as it doesn't return any information whether the document
        already exists or not. That prevents us from returning errors when using the FAIL policy or skipping a
        Document when using the SKIP policy.
        """
        if policy in [DuplicatePolicy.NONE, DuplicatePolicy.OVERWRITE]:
            return self._batch_write(documents)

        return self._write(documents, policy)

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes all documents with matching document_ids from the DocumentStore.

        :param document_ids: The object_ids to delete.
        """
        weaviate_ids = [generate_uuid5(doc_id) for doc_id in document_ids]
        self.collection.data.delete_many(where=weaviate.classes.query.Filter.by_id().contains_any(weaviate_ids))

    def delete_all_documents(self, *, recreate_index: bool = False, batch_size: int = 1000) -> None:
        """
        Deletes all documents in a collection.

        If recreate_index is False, it keeps the collection but deletes documents iteratively.
        If recreate_index is True, the collection is dropped and faithfully recreated.
        This is recommended for performance reasons.

        :param recreate_index: Use drop and recreate strategy. (recommended for performance)
        :param batch_size: Only relevant if recreate_index is false. Defines the deletion batch size.
            Note that this parameter needs to be less or equal to the set `QUERY_MAXIMUM_RESULTS` variable
            set for the weaviate deployment (default is 10000).
            Reference: https://docs.weaviate.io/weaviate/manage-objects/delete#delete-all-objects
        """

        if recreate_index:
            # get current up-to-date config from server, so we can recreate the collection faithfully
            cfg = self.client.collections.get(self._collection_settings["class"]).config.get().to_dict()
            class_name = cfg.get("class", self._collection_settings["class"])

            self.client.collections.delete(class_name)
            self.client.collections.create_from_dict(cfg)

            self._collection_settings = cfg
            self._collection = self.client.collections.get(class_name)
            return

        uuids = []
        batch_size = max(1, int(batch_size))

        for obj in self.collection.iterator(return_properties=[], include_vector=False):
            uuids.append(obj.uuid)
            if len(uuids) >= batch_size:
                res = self.collection.data.delete_many(where=weaviate.classes.query.Filter.by_id().contains_any(uuids))
                if res.successful < len(uuids):
                    logger.warning(
                        "Not all documents in the batch have been deleted. "
                        "Make sure to specify a deletion `batch_size` which is less than `QUERY_MAXIMUM_RESULTS`.",
                    )
                uuids.clear()

        if uuids:
            res = self.collection.data.delete_many(where=weaviate.classes.query.Filter.by_id().contains_any(uuids))
            if res.successful < len(uuids):
                logger.warning(
                    "Not all documents have been deleted. "
                    "Make sure to specify a deletion `batch_size` which is less than `QUERY_MAXIMUM_RESULTS`.",
                )

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Deletes all documents that match the provided filters.

        :param filters: The filters to apply to select documents for deletion.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents deleted.
        """
        validate_filters(filters)

        try:
            weaviate_filter = convert_filters(filters)
            result = self.collection.data.delete_many(where=weaviate_filter)
            deleted_count = result.successful
            logger.info(
                "Deleted {n_docs} documents from collection '{collection}' using filters.",
                n_docs=deleted_count,
                collection=self.collection.name,
            )
            return deleted_count
        except weaviate.exceptions.WeaviateQueryError as e:
            msg = f"Failed to delete documents by filter in Weaviate. Error: {e.message}"
            raise DocumentStoreError(msg) from e
        except Exception as e:
            msg = f"Failed to delete documents by filter in Weaviate: {e!s}"
            raise DocumentStoreError(msg) from e

    async def delete_by_filter_async(self, filters: dict[str, Any]) -> int:
        """
        Asynchronously deletes all documents that match the provided filters.

        :param filters: The filters to apply to select documents for deletion.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents deleted.
        """
        validate_filters(filters)

        try:
            collection = await self.async_collection
            weaviate_filter = convert_filters(filters)
            result = await collection.data.delete_many(where=weaviate_filter)
            deleted_count = result.successful
            logger.info(
                "Deleted {n_docs} documents from collection '{collection}' using filters.",
                n_docs=deleted_count,
                collection=collection.name,
            )
            return deleted_count
        except weaviate.exceptions.WeaviateQueryError as e:
            msg = f"Failed to delete documents by filter in Weaviate. Error: {e.message}"
            raise DocumentStoreError(msg) from e
        except Exception as e:
            msg = f"Failed to delete documents by filter in Weaviate: {e!s}"
            raise DocumentStoreError(msg) from e

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Updates the metadata of all documents that match the provided filters.

        :param filters: The filters to apply to select documents for updating.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param meta: The metadata fields to update. These will be merged with existing metadata.
        :returns: The number of documents updated.
        """
        validate_filters(filters)

        if not isinstance(meta, dict):
            msg = "Meta must be a dictionary"
            raise ValueError(msg)

        try:
            matching_objects = self._query_with_filters(filters)
            if not matching_objects:
                return 0

            # Update each object with the new metadata
            # Since metadata is stored flattened in Weaviate properties, we update properties directly
            updated_count = 0
            failed_updates = []

            for obj in matching_objects:
                try:
                    # Get current properties
                    current_properties = obj.properties.copy() if obj.properties else {}

                    # Update with new metadata values
                    # Note: metadata fields are stored directly in properties (flattened)
                    for key, value in meta.items():
                        current_properties[key] = value

                    # Update the object, preserving the vector
                    # Get the vector from the object to preserve it during replace
                    vector = None
                    if isinstance(obj.vector, list):
                        vector = obj.vector
                    elif isinstance(obj.vector, dict):
                        vector = obj.vector.get("default")

                    self.collection.data.replace(
                        uuid=obj.uuid,
                        properties=current_properties,
                        vector=vector,
                    )
                    updated_count += 1
                except Exception as e:
                    # Collect failed updates but continue with others
                    obj_properties = obj.properties or {}
                    id_ = obj_properties.get("_original_id", obj.uuid)
                    failed_updates.append((id_, str(e)))

            if failed_updates:
                msg = "\n".join(
                    [f"Failed to update object with id '{id_}'. Error: '{error}'" for id_, error in failed_updates]
                )
                raise DocumentStoreError(msg)

            logger.info(
                "Updated {n_docs} documents in collection '{collection}' using filters.",
                n_docs=updated_count,
                collection=self.collection.name,
            )
            return updated_count
        except weaviate.exceptions.WeaviateQueryError as e:
            msg = f"Failed to update documents by filter in Weaviate. Error: {e.message}"
            raise DocumentStoreError(msg) from e
        except Exception as e:
            msg = f"Failed to update documents by filter in Weaviate: {e!s}"
            raise DocumentStoreError(msg) from e

    async def update_by_filter_async(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Asynchronously updates the metadata of all documents that match the provided filters.

        :param filters: The filters to apply to select documents for updating.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param meta: The metadata fields to update. These will be merged with existing metadata.
        :returns: The number of documents updated.
        """
        validate_filters(filters)

        if not isinstance(meta, dict):
            msg = "Meta must be a dictionary"
            raise ValueError(msg)

        try:
            collection = await self.async_collection
            weaviate_filter = convert_filters(filters)
            config = await collection.config.get()
            properties = [p.name for p in config.properties]

            # Query all objects matching the filter
            matching_objects = []
            offset = 0
            partial_result = None

            # Paginate through all matching objects
            # We include vector=True to preserve vectors when updating
            while partial_result is None or len(partial_result.objects) == DEFAULT_QUERY_LIMIT:
                partial_result = await collection.query.fetch_objects(
                    filters=weaviate_filter,
                    include_vector=True,
                    limit=DEFAULT_QUERY_LIMIT,
                    offset=offset,
                    return_properties=properties,
                )
                matching_objects.extend(partial_result.objects)
                offset += DEFAULT_QUERY_LIMIT

            if not matching_objects:
                return 0

            # Update each object with the new metadata
            # Since metadata is stored flattened in Weaviate properties, we update properties directly
            updated_count = 0
            failed_updates = []

            for obj in matching_objects:
                try:
                    # Get current properties
                    current_properties = obj.properties.copy() if obj.properties else {}

                    # Update with new metadata values
                    # Note: metadata fields are stored directly in properties (flattened)
                    for key, value in meta.items():
                        current_properties[key] = value

                    # Update the object, preserving the vector
                    # Get the vector from the object to preserve it during replace
                    vector = None
                    if isinstance(obj.vector, list):
                        vector = obj.vector
                    elif isinstance(obj.vector, dict):
                        vector = obj.vector.get("default")

                    await collection.data.replace(
                        uuid=obj.uuid,
                        properties=current_properties,
                        vector=vector,
                    )
                    updated_count += 1
                except Exception as e:
                    # Collect failed updates but continue with others
                    obj_properties = obj.properties or {}
                    id_ = obj_properties.get("_original_id", obj.uuid)
                    failed_updates.append((id_, str(e)))

            if failed_updates:
                msg = "\n".join(
                    [f"Failed to update object with id '{id_}'. Error: '{error}'" for id_, error in failed_updates]
                )
                raise DocumentStoreError(msg)

            logger.info(
                "Updated {n_docs} documents in collection '{collection}' using filters.",
                n_docs=updated_count,
                collection=collection.name,
            )
            return updated_count
        except weaviate.exceptions.WeaviateQueryError as e:
            msg = f"Failed to update documents by filter in Weaviate. Error: {e.message}"
            raise DocumentStoreError(msg) from e
        except Exception as e:
            msg = f"Failed to update documents by filter in Weaviate: {e!s}"
            raise DocumentStoreError(msg) from e

    def _bm25_retrieval(
        self, query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
    ) -> list[Document]:
        properties = [p.name for p in self.collection.config.get().properties]
        result = self.collection.query.bm25(
            query=query,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
            include_vector=True,
            query_properties=["content"],
            return_properties=properties,
            return_metadata=["score"],
        )

        return [WeaviateDocumentStore._to_document(doc) for doc in result.objects]

    async def _bm25_retrieval_async(
        self, query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
    ) -> list[Document]:
        collection = await self.async_collection
        config = await collection.config.get()
        properties = [p.name for p in config.properties]
        result = await collection.query.bm25(
            query=query,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
            include_vector=True,
            query_properties=["content"],
            return_properties=properties,
            return_metadata=["score"],
        )

        return [WeaviateDocumentStore._to_document(doc) for doc in result.objects]

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        distance: float | None = None,
        certainty: float | None = None,
    ) -> list[Document]:
        if distance is not None and certainty is not None:
            msg = "Can't use 'distance' and 'certainty' parameters together"
            raise ValueError(msg)

        properties = [p.name for p in self.collection.config.get().properties]
        result = self.collection.query.near_vector(
            near_vector=query_embedding,
            distance=distance,
            certainty=certainty,
            include_vector=True,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
            return_properties=properties,
            return_metadata=["certainty"],
        )

        return [WeaviateDocumentStore._to_document(doc) for doc in result.objects]

    async def _embedding_retrieval_async(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        distance: float | None = None,
        certainty: float | None = None,
    ) -> list[Document]:
        if distance is not None and certainty is not None:
            msg = "Can't use 'distance' and 'certainty' parameters together"
            raise ValueError(msg)

        collection = await self.async_collection
        config = await collection.config.get()
        properties = [p.name for p in config.properties]
        result = await collection.query.near_vector(
            near_vector=query_embedding,
            distance=distance,
            certainty=certainty,
            include_vector=True,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
            return_properties=properties,
            return_metadata=["certainty"],
        )

        return [WeaviateDocumentStore._to_document(doc) for doc in result.objects]

    def _hybrid_retrieval(
        self,
        query: str,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        alpha: float | None = None,
        max_vector_distance: float | None = None,
    ) -> list[Document]:
        properties = [p.name for p in self.collection.config.get().properties]
        result = self.collection.query.hybrid(
            query=query,
            vector=query_embedding,
            alpha=alpha,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
            max_vector_distance=max_vector_distance,
            include_vector=True,
            query_properties=["content"],
            return_properties=properties,
            return_metadata=["score"],
        )

        return [WeaviateDocumentStore._to_document(doc) for doc in result.objects]

    async def _hybrid_retrieval_async(
        self,
        query: str,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        alpha: float | None = None,
        max_vector_distance: float | None = None,
    ) -> list[Document]:
        collection = await self.async_collection
        config = await collection.config.get()
        properties = [p.name for p in config.properties]
        result = await collection.query.hybrid(
            query=query,
            vector=query_embedding,
            alpha=alpha,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
            max_vector_distance=max_vector_distance,
            include_vector=True,
            query_properties=["content"],
            return_properties=properties,
            return_metadata=["score"],
        )

        return [WeaviateDocumentStore._to_document(doc) for doc in result.objects]
