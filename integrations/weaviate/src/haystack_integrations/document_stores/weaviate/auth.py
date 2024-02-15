from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Type

from haystack.core.errors import DeserializationError
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from weaviate.auth import AuthApiKey as WeaviateAuthApiKey
from weaviate.auth import AuthBearerToken as WeaviateAuthBearerToken
from weaviate.auth import AuthClientCredentials as WeaviateAuthClientCredentials
from weaviate.auth import AuthClientPassword as WeaviateAuthClientPassword


@dataclass
class AuthCredentials(ABC):
    """
    Base class for all auth credentials supported by WeaviateDocumentStore.
    Can be used to deserialize from dict any of the supported auth credentials.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dictionary representation for serialization.
        """
        _fields = {}
        for field in fields(self):
            if field.type is Secret:
                _fields[field.name] = getattr(self, field.name).to_dict()
            else:
                _fields[field.name] = getattr(self, field.name)

        return default_to_dict(
            self,
            **_fields,
        )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AuthCredentials":
        """
        Converts a dictionary representation to an auth credentials object.
        """
        if "type" not in data:
            msg = "Missing 'type' in serialization data"
            raise DeserializationError(msg)
        return _AUTH_CLASSES[data["type"]]._from_dict(data)

    @classmethod
    @abstractmethod
    def _from_dict(cls, data: Dict[str, Any]):
        """
        Internal method to convert a dictionary representation to an auth credentials object.
        All subclasses must implement this method.
        """

    @abstractmethod
    def resolve_value(self):
        """
        Resolves all the secrets in the auth credentials object and returns the corresponding Weaviate object.
        All subclasses must implement this method.
        """


@dataclass
class AuthApiKey(AuthCredentials):
    """
    AuthCredentials for API key authentication.
    By default it will load `api_key` from the environment variable `WEAVIATE_API_KEY`.
    """

    api_key: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_API_KEY"]))

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AuthApiKey":
        deserialize_secrets_inplace(data["init_parameters"], ["api_key"])
        return default_from_dict(cls, data)

    def resolve_value(self) -> WeaviateAuthApiKey:
        return WeaviateAuthApiKey(api_key=self.api_key.resolve_value())


@dataclass
class AuthBearerToken(AuthCredentials):
    """
    AuthCredentials for Bearer token authentication.
    By default it will load `access_token` from the environment variable `WEAVIATE_ACCESS_TOKEN`,
    and `refresh_token` from the environment variable
    `WEAVIATE_REFRESH_TOKEN`.
    `WEAVIATE_REFRESH_TOKEN` environment variable is optional.
    """

    access_token: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_ACCESS_TOKEN"]))
    expires_in: int = field(default=60)
    refresh_token: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_REFRESH_TOKEN"], strict=False))

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AuthBearerToken":
        deserialize_secrets_inplace(data["init_parameters"], ["access_token", "refresh_token"])
        return default_from_dict(cls, data)

    def resolve_value(self) -> WeaviateAuthBearerToken:
        access_token = self.access_token.resolve_value()
        refresh_token = self.refresh_token.resolve_value()

        return WeaviateAuthBearerToken(
            access_token=access_token,
            expires_in=self.expires_in,
            refresh_token=refresh_token,
        )


@dataclass
class AuthClientCredentials(AuthCredentials):
    """
    AuthCredentials for client credentials authentication.
    By default it will load `client_secret` from the environment variable `WEAVIATE_CLIENT_SECRET`, and
    `scope` from the environment variable `WEAVIATE_SCOPE`.
    `WEAVIATE_SCOPE` environment variable is optional, if set it can either be a string or a list of space
    separated strings. e.g "scope1" or "scope1 scope2".
    """

    client_secret: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_CLIENT_SECRET"]))
    scope: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_SCOPE"], strict=False))

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AuthClientCredentials":
        deserialize_secrets_inplace(data["init_parameters"], ["client_secret", "scope"])
        return default_from_dict(cls, data)

    def resolve_value(self) -> WeaviateAuthClientCredentials:
        return WeaviateAuthClientCredentials(
            client_secret=self.client_secret.resolve_value(),
            scope=self.scope.resolve_value(),
        )


@dataclass
class AuthClientPassword(AuthCredentials):
    """
    AuthCredentials for username and password authentication.
    By default it will load `username` from the environment variable `WEAVIATE_USERNAME`,
    `password` from the environment variable `WEAVIATE_PASSWORD`, and
    `scope` from the environment variable `WEAVIATE_SCOPE`.
    `WEAVIATE_SCOPE` environment variable is optional, if set it can either be a string or a list of space
    separated strings. e.g "scope1" or "scope1 scope2".
    """

    username: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_USERNAME"]))
    password: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_PASSWORD"]))
    scope: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_SCOPE"], strict=False))

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AuthClientPassword":
        deserialize_secrets_inplace(data["init_parameters"], ["username", "password", "scope"])
        return default_from_dict(cls, data)

    def resolve_value(self) -> WeaviateAuthClientPassword:
        return WeaviateAuthClientPassword(
            username=self.username.resolve_value(),
            password=self.password.resolve_value(),
            scope=self.scope.resolve_value(),
        )


# This simplifies a bit how we handle deserialization of the auth credentials.
_AUTH_CLASSES: Dict[str, Type[AuthCredentials]] = {
    "haystack_integrations.document_stores.weaviate.auth.AuthClientCredentials": AuthClientCredentials,
    "haystack_integrations.document_stores.weaviate.auth.AuthClientPassword": AuthClientPassword,
    "haystack_integrations.document_stores.weaviate.auth.AuthBearerToken": AuthBearerToken,
    "haystack_integrations.document_stores.weaviate.auth.AuthApiKey": AuthApiKey,
}
