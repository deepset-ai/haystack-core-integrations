# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import base64
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types.policy import DuplicatePolicy

import weaviate
from weaviate.collections.classes.internal import Object
from weaviate.config import AdditionalConfig
from weaviate.embedded import EmbeddedOptions
from weaviate.util import generate_uuid5

from ._filters import convert_filters
from .auth import AuthCredentials

Number = Union[int, float]
TimeoutType = Union[Tuple[Number, Number], Number]


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


class WeaviateDocumentStore:
    """
    WeaviateDocumentStore is a Document Store for Weaviate.
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

        :param url: The URL to the weaviate instance, defaults to None.
        :param collection_settings: The collection settings to use, defaults to None.
            If None it will use a collection named `default` with the following properties:
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
        :param auth_client_secret: Authentication credentials, defaults to None.
            Can be one of the following types depending on the authentication mode:
            - `AuthBearerToken` to use existing access and (optionally, but recommended) refresh tokens
            - `AuthClientPassword` to use username and password for oidc Resource Owner Password flow
            - `AuthClientCredentials` to use a client secret for oidc client credential flow
            - `AuthApiKey` to use an API key
        :param additional_headers: Additional headers to include in the requests, defaults to None.
            Can be used to set OpenAI/HuggingFace keys. OpenAI/HuggingFace key looks like this:
            ```
            {"X-OpenAI-Api-Key": "<THE-KEY>"}, {"X-HuggingFace-Api-Key": "<THE-KEY>"}
            ```
        :param embedded_options: If set create an embedded Weaviate cluster inside the client, defaults to None.
            For a full list of options see `weaviate.embedded.EmbeddedOptions`.
        :param additional_config: Additional and advanced configuration options for weaviate, defaults to None.
        :param grpc_port: The port to use for the gRPC connection, defaults to 50051.
        :param grpc_secure: Whether to use a secure channel for the underlying gRPC API.
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
        embedded_options = asdict(self._embedded_options) if self._embedded_options else None
        additional_config = json.loads(self._additional_config.model_dump_json()) if self._additional_config else None

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
        total = self._collection.aggregate.over_all(total_count=True).total_count
        return total if total else 0

    def _to_data_object(self, document: Document) -> Dict[str, Any]:
        """
        Convert a Document to a Weviate data object ready to be saved.
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

        return data

    def _convert_weaviate_v4_object_to_v3_object(self, data: Object) -> Dict[str, Any]:
        properties = self._collection.config.get().properties
        properties_with_date_type = [
            p.name for p in properties if p.data_type.name == "DATE" and data.properties.get(p.name)
        ]

        v4_object = data.__dict__
        v3_object = v4_object.pop("properties")
        v3_object["_additional"] = {"vector": v4_object.pop("vector").get("default")}

        if "blob_data" not in v3_object:
            v3_object["blob_data"] = None
            v3_object["blob_mime_type"] = None

        for date_prop in properties_with_date_type:
            v3_object[date_prop] = v3_object[date_prop].strftime("%Y-%m-%dT%H:%M:%SZ")
        return v3_object

    def _to_document(self, data: Dict[str, Any]) -> Document:
        """
        Convert a data object read from Weaviate into a Document.
        """
        data["id"] = data.pop("_original_id")
        data["embedding"] = data["_additional"].pop("vector") if data["_additional"].get("vector") else None

        if (blob_data := data.get("blob_data")) is not None:
            data["blob"] = {
                "data": base64.b64decode(blob_data),
                "mime_type": data.get("blob_mime_type"),
            }
        # We always delete these fields as they're not part of the Document dataclass
        data.pop("blob_data")
        data.pop("blob_mime_type")

        # We don't need these fields anymore, this usually only contains the uuid
        # used by Weaviate to identify the object and the embedding vector that we already extracted.
        del data["_additional"]

        return Document.from_dict(data)

    def _query_paginated(self, properties: List[str]):

        try:
            result = self._collection.iterator(include_vector=True, return_properties=properties)
        except weaviate.exceptions.WeaviateQueryError as e:
            msg = f"Failed to query documents in Weaviate. Error: {e.message}"
            raise DocumentStoreError(msg) from e
        return list(result)

    def _query_with_filters(self, filters: weaviate.collections.classes.filters.Filter) -> List[Dict[str, Any]]:

        try:
            # this is the default value for max number of objects to retrieve in Weaviate
            # see QUERY_MAXIMUM_RESULTS
            # see https://weaviate.io/developers/weaviate/config-refs/env-vars#overview
            # and https://weaviate.io/developers/weaviate/api/graphql/additional-operators#pagination-with-offset
            limit = 10_000
            result = self._collection.query.fetch_objects(
                filters=convert_filters(filters), include_vector=True, limit=limit
            )
        except weaviate.exceptions.WeaviateQueryError as e:
            msg = f"Failed to query documents in Weaviate. Error: {e.message}"
            raise DocumentStoreError(msg) from e

        return result.objects

    def filter_documents(self, filters: Optional[weaviate.collections.classes.filters.Filter] = None) -> List[Document]:
        properties = self._collection.config.get().properties
        properties = [prop.name for prop in properties]

        if filters:
            result = self._query_with_filters(filters)
            return [self._to_document(self._convert_weaviate_v4_object_to_v3_object(doc)) for doc in result]

        result = self._query_paginated(properties)
        result = [self._to_document(self._convert_weaviate_v4_object_to_v3_object(doc)) for doc in result]
        return result

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
        failed_objects = self._client.batch.failed_objects
        if failed_objects:
            msg = "\n".join(
                [
                    f"Failed to write object with id '{obj.object_._original_id}'. Error: '{obj.message}'"
                    for obj in failed_objects
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

        weaviate_ids = [generate_uuid5(doc_id) for doc_id in document_ids]
        self._collection.data.delete_many(where=weaviate.classes.query.Filter.by_id().contains_any(weaviate_ids))

    def _bm25_retrieval(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None
    ) -> List[Document]:

        result = self._collection.query.bm25(
            query=query,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
            include_vector=True,
            query_properties=["content"],
        )

        return [self._to_document(self._convert_weaviate_v4_object_to_v3_object(doc)) for doc in result.objects]

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

        result = self._collection.query.near_vector(
            near_vector=query_embedding,
            distance=distance,
            certainty=certainty,
            include_vector=True,
            filters=convert_filters(filters) if filters else None,
            limit=top_k,
        )

        return [self._to_document(self._convert_weaviate_v4_object_to_v3_object(doc)) for doc in result.objects]
