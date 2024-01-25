# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import base64
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses.document import Document
from haystack.document_stores.types.policy import DuplicatePolicy

import weaviate
from weaviate.auth import AuthCredentials
from weaviate.config import Config, ConnectionConfig
from weaviate.embedded import EmbeddedOptions

Number = Union[int, float]
TimeoutType = Union[Tuple[Number, Number], Number]

# This simplifies a bit how we handle deserialization of the auth credentials.
# Otherwise we would need to use importlib to dynamically import the correct class.
_AUTH_CLASSES = {
    "weaviate.auth.AuthClientCredentials": weaviate.auth.AuthClientCredentials,
    "weaviate.auth.AuthClientPassword": weaviate.auth.AuthClientPassword,
    "weaviate.auth.AuthBearerToken": weaviate.auth.AuthBearerToken,
    "weaviate.auth.AuthApiKey": weaviate.auth.AuthApiKey,
}

# This is the default collection properties for Weaviate.
# It's a list of properties that will be created on the collection.
# These are extremely similar to the Document dataclass, but with a few differences:
# - `id` is renamed to `_original_id` as the `id` field is reserved by Weaviate.
# - `blob` is split into `blob_data` and `blob_mime_type` as it's more efficient to store them separately.
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
        timeout_config: TimeoutType = (10, 60),
        proxies: Optional[Union[Dict, str]] = None,
        trust_env: bool = False,
        additional_headers: Optional[Dict] = None,
        startup_period: Optional[int] = 5,
        embedded_options: Optional[EmbeddedOptions] = None,
        additional_config: Optional[Config] = None,
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
            See the official `Weaviate documentation<https://weaviate.io/developers/weaviate/manage-data/collections>`_
            for more information on collections.
        :param auth_client_secret: Authentication credentials, defaults to None.
            Can be one of the following types depending on the authentication mode:
            - `weaviate.auth.AuthBearerToken` to use existing access and (optionally, but recommended) refresh tokens
            - `weaviate.auth.AuthClientPassword` to use username and password for oidc Resource Owner Password flow
            - `weaviate.auth.AuthClientCredentials` to use a client secret for oidc client credential flow
            - `weaviate.auth.AuthApiKey` to use an API key
        :param timeout_config: Timeout configuration for all requests to the Weaviate server, defaults to (10, 60).
            It can be a real number or, a tuple of two real numbers: (connect timeout, read timeout).
            If only one real number is passed then both connect and read timeout will be set to
            that value, by default (2, 20).
        :param proxies: Proxy configuration, defaults to None.
            Can be passed as a dict using the
            ``requests` format<https://docs.python-requests.org/en/stable/user/advanced/#proxies>`_,
            or a string. If a string is passed it will be used for both HTTP and HTTPS requests.
        :param trust_env: Whether to read proxies from the ENV variables, defaults to False.
            Proxies will be read from the following ENV variables:
            * `HTTP_PROXY`
            * `http_proxy`
            * `HTTPS_PROXY`
            * `https_proxy`
            If `proxies` is not None, `trust_env` is ignored.
        :param additional_headers: Additional headers to include in the requests, defaults to None.
            Can be used to set OpenAI/HuggingFace keys. OpenAI/HuggingFace key looks like this:
            ```
            {"X-OpenAI-Api-Key": "<THE-KEY>"}, {"X-HuggingFace-Api-Key": "<THE-KEY>"}
            ```
        :param startup_period: How many seconds the client will wait for Weaviate to start before
            raising a RequestsConnectionError, defaults to 5.
        :param embedded_options: If set create an embedded Weaviate cluster inside the client, defaults to None.
            For a full list of options see `weaviate.embedded.EmbeddedOptions`.
        :param additional_config: Additional and advanced configuration options for weaviate, defaults to None.
        """
        self._client = weaviate.Client(
            url=url,
            auth_client_secret=auth_client_secret,
            timeout_config=timeout_config,
            proxies=proxies,
            trust_env=trust_env,
            additional_headers=additional_headers,
            startup_period=startup_period,
            embedded_options=embedded_options,
            additional_config=additional_config,
        )

        # Test connection, it will raise an exception if it fails.
        self._client.schema.get()

        if collection_settings is None:
            collection_settings = {
                "class": "Default",
                "properties": DOCUMENT_COLLECTION_PROPERTIES,
            }
        else:
            # Set the class if not set
            collection_settings["class"] = collection_settings.get("class", "default").capitalize()
            # Set the properties if they're not set
            collection_settings["properties"] = collection_settings.get("properties", DOCUMENT_COLLECTION_PROPERTIES)

        if not self._client.schema.exists(collection_settings["class"]):
            self._client.schema.create_class(collection_settings)

        self._url = url
        self._collection_settings = collection_settings
        self._auth_client_secret = auth_client_secret
        self._timeout_config = timeout_config
        self._proxies = proxies
        self._trust_env = trust_env
        self._additional_headers = additional_headers
        self._startup_period = startup_period
        self._embedded_options = embedded_options
        self._additional_config = additional_config

    def to_dict(self) -> Dict[str, Any]:
        auth_client_secret = None
        if self._auth_client_secret:
            # There are different types of AuthCredentials, so even thought it's a dataclass
            # and we could just use asdict, we need to save the type too.
            auth_client_secret = default_to_dict(self._auth_client_secret, **asdict(self._auth_client_secret))
        embedded_options = asdict(self._embedded_options) if self._embedded_options else None
        additional_config = asdict(self._additional_config) if self._additional_config else None

        return default_to_dict(
            self,
            url=self._url,
            collection_settings=self._collection_settings,
            auth_client_secret=auth_client_secret,
            timeout_config=self._timeout_config,
            proxies=self._proxies,
            trust_env=self._trust_env,
            additional_headers=self._additional_headers,
            startup_period=self._startup_period,
            embedded_options=embedded_options,
            additional_config=additional_config,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeaviateDocumentStore":
        if (timeout_config := data["init_parameters"].get("timeout_config")) is not None:
            data["init_parameters"]["timeout_config"] = (
                tuple(timeout_config) if isinstance(timeout_config, list) else timeout_config
            )
        if (auth_client_secret := data["init_parameters"].get("auth_client_secret")) is not None:
            auth_class = _AUTH_CLASSES[auth_client_secret["type"]]
            data["init_parameters"]["auth_client_secret"] = default_from_dict(auth_class, auth_client_secret)
        if (embedded_options := data["init_parameters"].get("embedded_options")) is not None:
            data["init_parameters"]["embedded_options"] = EmbeddedOptions(**embedded_options)
        if (additional_config := data["init_parameters"].get("additional_config")) is not None:
            additional_config["connection_config"] = ConnectionConfig(**additional_config["connection_config"])
            data["init_parameters"]["additional_config"] = Config(**additional_config)
        return default_from_dict(
            cls,
            data,
        )

    def count_documents(self) -> int:
        collection_name = self._collection_settings["class"]
        res = self._client.query.aggregate(collection_name).with_meta_count().do()
        return res.get("data", {}).get("Aggregate", {}).get(collection_name, [{}])[0].get("meta", {}).get("count", 0)

    def _to_data_object(self, document: Document) -> Dict[str, Any]:
        """
        Convert a Document to a Weviate data object ready to be saved.
        """
        data = document.to_dict(flatten=False)
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

        # Weaviate doesn't like empty objects, let's delete meta if it's empty
        if data["meta"] == {}:
            del data["meta"]

        return data

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

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:  # noqa: ARG002
        return []

    def write_documents(
        self,
        documents: List[Document],  # noqa: ARG002
        policy: DuplicatePolicy = DuplicatePolicy.NONE,  # noqa: ARG002
    ) -> int:
        return 0

    def delete_documents(self, document_ids: List[str]) -> None:  # noqa: ARG002
        return
