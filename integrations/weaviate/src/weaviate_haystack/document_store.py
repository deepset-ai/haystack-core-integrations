# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Union, Dict, Tuple, Optional, List, Any
from dataclasses import asdict

import weaviate
from weaviate.auth import AuthCredentials
from weaviate.embedded import EmbeddedOptions
from weaviate.config import Config, ConnectionConfig

from haystack.core.serialization import default_to_dict, default_from_dict
from haystack.dataclasses.document import Document
from haystack.document_stores.protocol import DuplicatePolicy

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


class WeaviateDocumentStore:
    """
    WeaviateDocumentStore is a Document Store for Weaviate.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
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

        self._url = url
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
        ...

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        ...

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        ...

    def delete_documents(self, document_ids: List[str]) -> None:
        ...
