# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, Type

from haystack.core.errors import DeserializationError
from haystack.utils.auth import Secret, deserialize_secrets_inplace

from weaviate.auth import AuthApiKey as WeaviateAuthApiKey
from weaviate.auth import AuthBearerToken as WeaviateAuthBearerToken
from weaviate.auth import AuthClientCredentials as WeaviateAuthClientCredentials
from weaviate.auth import AuthClientPassword as WeaviateAuthClientPassword


class SupportedAuthTypes(Enum):
    """
    Supported auth credentials for WeaviateDocumentStore.
    """

    API_KEY = "api_key"
    BEARER = "bearer"
    CLIENT_CREDENTIALS = "client_credentials"
    CLIENT_PASSWORD = "client_password"

    def __str__(self):
        return self.value

    @staticmethod
    def from_class(auth_class) -> "SupportedAuthTypes":
        auth_types = {
            AuthApiKey: SupportedAuthTypes.API_KEY,
            AuthBearerToken: SupportedAuthTypes.BEARER,
            AuthClientCredentials: SupportedAuthTypes.CLIENT_CREDENTIALS,
            AuthClientPassword: SupportedAuthTypes.CLIENT_PASSWORD,
        }
        return auth_types[auth_class]


@dataclass(frozen=True)
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
        for _field in fields(self):
            if _field.type is Secret:
                _fields[_field.name] = getattr(self, _field.name).to_dict()
            else:
                _fields[_field.name] = getattr(self, _field.name)

        return {"type": str(SupportedAuthTypes.from_class(self.__class__)), "init_parameters": _fields}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "AuthCredentials":
        """
        Converts a dictionary representation to an auth credentials object.
        """
        if "type" not in data:
            msg = "Missing 'type' in serialization data"
            raise DeserializationError(msg)

        auth_classes: Dict[str, Type[AuthCredentials]] = {
            str(SupportedAuthTypes.API_KEY): AuthApiKey,
            str(SupportedAuthTypes.BEARER): AuthBearerToken,
            str(SupportedAuthTypes.CLIENT_CREDENTIALS): AuthClientCredentials,
            str(SupportedAuthTypes.CLIENT_PASSWORD): AuthClientPassword,
        }

        return auth_classes[data["type"]]._from_dict(data)

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


@dataclass(frozen=True)
class AuthApiKey(AuthCredentials):
    """
    AuthCredentials for API key authentication.
    By default it will load `api_key` from the environment variable `WEAVIATE_API_KEY`.
    """

    api_key: Secret = field(default_factory=lambda: Secret.from_env_var(["WEAVIATE_API_KEY"]))

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "AuthApiKey":
        deserialize_secrets_inplace(data["init_parameters"], ["api_key"])
        return cls(**data["init_parameters"])

    def resolve_value(self) -> WeaviateAuthApiKey:
        return WeaviateAuthApiKey(api_key=self.api_key.resolve_value())


@dataclass(frozen=True)
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
        return cls(**data["init_parameters"])

    def resolve_value(self) -> WeaviateAuthBearerToken:
        access_token = self.access_token.resolve_value()
        refresh_token = self.refresh_token.resolve_value()

        return WeaviateAuthBearerToken(
            access_token=access_token,
            expires_in=self.expires_in,
            refresh_token=refresh_token,
        )


@dataclass(frozen=True)
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
        return cls(**data["init_parameters"])

    def resolve_value(self) -> WeaviateAuthClientCredentials:
        return WeaviateAuthClientCredentials(
            client_secret=self.client_secret.resolve_value(),
            scope=self.scope.resolve_value(),
        )


@dataclass(frozen=True)
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
        return cls(**data["init_parameters"])

    def resolve_value(self) -> WeaviateAuthClientPassword:
        return WeaviateAuthClientPassword(
            username=self.username.resolve_value(),
            password=self.password.resolve_value(),
            scope=self.scope.resolve_value(),
        )
