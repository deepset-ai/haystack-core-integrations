# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import base64
import datetime
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types.policy import DuplicatePolicy

import weaviate
from weaviate.collections.classes.data import DataObject
from weaviate.config import AdditionalConfig
from weaviate.embedded import EmbeddedOptions
from weaviate.util import generate_uuid5

from ._filters import convert_filters
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
    {"name": "dataframe", "dataType": ["text"]},
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
    WeaviateDocumentStore is a Document Store for Weaviate.
    It can be used with Weaviate Cloud Services or self-hosted instances.

    Usage example with Weaviate Cloud Services:
    ```python
    import os
    from haystack_integrations.document_stores.weaviate.auth import AuthApiKey
    from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore

    os.environ["WEAVIATE_API_KEY"] = "MY_API_KEY

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
        url: Optional[str] = None,
        collection_settings: Optional[Dict[str, Any]] = None,
        auth_client_secret: Optional[AuthCredentials] = None,
        additional_headers: Optional[Dict] = None,
        embedded_options: Optional[EmbeddedOptions] = None,
        additional_config: Optional[AdditionalConfig] = None,
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
            - dataframe: text
            - blob_data: blob
            - blob_mime_type: text
            - score: number
            The Document `meta` fields are omitted in the default collection settings as we can't make assumptions
            on the structure of the meta field.
            We heavily recommend to create a custom collection with the correct meta properties
            for your use case.
            Another option is relying on the automatic schema generation, but that's not recommended for
            production use.
            See the official `Weaviate documentation<https://weaviate.io/developers/weaviate/manage-data/collections>`_
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
        # proxies, timeout_config, trust_env are part of additional_config now
        # startup_period has been removed
        self._client = weaviate.WeaviateClient(
            connection_params=(
                weaviate.connect.base.ConnectionParams.from_url(url=url, grpc_port=grpc_port, grpc_secure=grpc_secure)
                if url
                else None
            ),
            auth_client_secret=auth_client_secret.resolve_value() if auth_client_secret else None,
            additional_config=additional_config,
            additional_headers=additional_headers,
            embedded_options=embedded_options,
            skip_init_checks=False,
        )
        self._client.connect()

        # Test connection, it will raise an exception if it fails.
        self._client.collections._get_all(simple=True)

        if collection_settings is None:
            collection_settings = {
                "class": "Default",
                "invertedIndexConfig": {"indexNullState": True},
                "properties": DOCUMENT_COLLECTION_PROPERTIES,
            }
        else:
            # Set the class if not set
            collection_settings["class"] = collection_settings.get("class", "default").capitalize()
            # Set the properties if they're not set
            collection_settings["properties"] = collection_settings.get("properties", DOCUMENT_COLLECTION_PROPERTIES)

        if not self._client.collections.exists(collection_settings["class"]):
            self._client.collections.create_from_dict(collection_settings)

        self._url = url
        self._collection_settings = collection_settings
        self._auth_client_secret = auth_client_secret
        self._additional_headers = additional_headers
        self._embedded_options = embedded_options
        self._additional_config = additional_config
        self._collection = self._client.collections.get(collection_settings["class"])

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "WeaviateDocumentStore":
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
        total = self._collection.aggregate.over_all(total_count=True).total_count
        return total if total else 0

    def _to_data_object(self, document: Document) -> Dict[str, Any]:
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

        if "sparse_embedding" in data:
            sparse_embedding = data.pop("sparse_embedding", None)
            if sparse_embedding:
                logger.warning(
                    "Document %s has the `sparse_embedding` field set,"
                    "but storing sparse embeddings in Weaviate is not currently supported."
                    "The `sparse_embedding` field will be ignored.",
                    data["_original_id"],
                )

        return data

    def _to_document(self, data: DataObject[Dict[str, Any], None]) -> Document:
        """
        Converts a data object read from Weaviate into a Document.
        """
        document_data = data.properties
        document_data["id"] = document_data.pop("_original_id")
        if isinstance(data.vector, List):
            document_data["embedding"] = data.vector
        elif isinstance(data.vector, Dict):
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

    def _query(self) -> List[Dict[str, Any]]:
        properties = [p.name for p in self._collection.config.get().properties]
        try:
            result = self._collection.iterator(include_vector=True, return_properties=properties)
        except weaviate.exceptions.WeaviateQueryError as e:
            msg = f"Failed to query documents in Weaviate. Error: {e.message}"
            raise DocumentStoreError(msg) from e
        return result

    def _query_with_filters(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        properties = [p.name for p in self._collection.config.get().properties]
        # When querying with filters we need to paginate using limit and offset as using
        # a cursor with after is not possible. See the official docs:
        # https://weaviate.io/developers/weaviate/api/graphql/additional-operators#cursor-with-after
        #
        # Nonetheless there's also another issue, paginating with limit and offset is not efficient
        # and it's still restricted by the QUERY_MAXIMUM_RESULTS environment variable.
        # If the sum of limit and offest is greater than QUERY_MAXIMUM_RESULTS an error is raised.
        # See the official docs for more:
        # https://weaviate.io/developers/weaviate/api/graphql/additional-operators#performance-considerations
        offset = 0
        partial_result = None
        result = []
        # Keep querying until we get all documents matching the filters
        while partial_result is None or len(partial_result.objects) == DEFAULT_QUERY_LIMIT:
            try:
                partial_result = self._collection.query.fetch_objects(
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

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters, refer to the
        DocumentStore.filter_documents() protocol documentation.

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        result = []
        if filters:
            result = self._query_with_filters(filters)
        else:
            result = self._query()
        return [self._to_document(doc) for doc in result]

    def _batch_write(self, documents: List[Document]) -> int:
        """
        Writes document to Weaviate in batches.
        Documents with the same id will be overwritten.
        Raises in case of errors.
        """

        with self._client.batch.dynamic() as batch:
            for doc in documents:
                if not isinstance(doc, Document):
                    msg = f"Expected a Document, got '{type(doc)}' instead."
                    raise ValueError(msg)

                batch.add_object(
                    properties=self._to_data_object(doc),
                    collection=self._collection.name,
                    uuid=generate_uuid5(doc.id),
                    vector=doc.embedding,
                )
        if failed_objects := self._client.batch.failed_objects:
            # We fallback to use the UUID if the _original_id is not present, this is just to be
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

    def _write(self, documents: List[Document], policy: DuplicatePolicy) -> int:
        """
        Writes documents to Weaviate using the specified policy.
        This doesn't uses the batch API, so it's slower than _batch_write.
        If policy is set to SKIP it will skip any document that already exists.
        If policy is set to FAIL it will raise an exception if any of the documents already exists.
        """
        written = 0
        duplicate_errors_ids = []
        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"Expected a Document, got '{type(doc)}' instead."
                raise ValueError(msg)

            if policy == DuplicatePolicy.SKIP and self._collection.data.exists(uuid=generate_uuid5(doc.id)):
                # This Document already exists, we skip it
                continue

            try:
                self._collection.data.insert(
                    uuid=generate_uuid5(doc.id),
                    properties=self._to_data_object(doc),
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

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
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

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with matching document_ids from the DocumentStore.

        :param document_ids: The object_ids to delete.
        """
        weaviate_ids = [generate_uuid5(doc_id) for doc_id in document_ids]
        self._collection.data.delete_many(where=weaviate.classes.query.Filter.by_id().contains_any(weaviate_ids))

    def _bm25_retrieval(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None
    ) -> List[Document]:
        properties = [p.name for p in self._collection.config.get().properties]
        result = self._collection.query.bm25(
            query=query,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
            include_vector=True,
            query_properties=["content"],
            return_properties=properties,
            return_metadata=["score"],
        )

        return [self._to_document(doc) for doc in result.objects]

    def _embedding_retrieval(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        distance: Optional[float] = None,
        certainty: Optional[float] = None,
    ) -> List[Document]:
        if distance is not None and certainty is not None:
            msg = "Can't use 'distance' and 'certainty' parameters together"
            raise ValueError(msg)

        properties = [p.name for p in self._collection.config.get().properties]
        result = self._collection.query.near_vector(
            near_vector=query_embedding,
            distance=distance,
            certainty=certainty,
            include_vector=True,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
            return_properties=properties,
            return_metadata=["certainty"],
        )

        return [self._to_document(doc) for doc in result.objects]
