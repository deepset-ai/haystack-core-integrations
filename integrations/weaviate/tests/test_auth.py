# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack_integrations.document_stores.weaviate.auth import (
    AuthApiKey,
    AuthBearerToken,
    AuthClientCredentials,
    AuthClientPassword,
    AuthCredentials,
)
from weaviate.auth import AuthApiKey as WeaviateAuthApiKey
from weaviate.auth import AuthBearerToken as WeaviateAuthBearerToken
from weaviate.auth import AuthClientCredentials as WeaviateAuthClientCredentials
from weaviate.auth import AuthClientPassword as WeaviateAuthClientPassword


class TestAuthApiKey:
    def test_init(self):
        credentials = AuthApiKey()
        assert credentials.api_key._env_vars == ("WEAVIATE_API_KEY",)
        assert credentials.api_key._strict

    def test_to_dict(self):
        credentials = AuthApiKey()
        assert credentials.to_dict() == {
            "type": "api_key",
            "init_parameters": {"api_key": {"env_vars": ["WEAVIATE_API_KEY"], "strict": True, "type": "env_var"}},
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("WEAVIATE_API_KEY", "fake_key")
        credentials = AuthCredentials.from_dict(
            {
                "type": "api_key",
                "init_parameters": {"api_key": {"env_vars": ["WEAVIATE_API_KEY"], "strict": True, "type": "env_var"}},
            }
        )
        assert isinstance(credentials, AuthApiKey)
        assert credentials.api_key._env_vars == ("WEAVIATE_API_KEY",)
        assert credentials.api_key._strict

    def test_resolve_value(self, monkeypatch):
        monkeypatch.setenv("WEAVIATE_API_KEY", "fake_key")
        credentials = AuthApiKey()
        resolved = credentials.resolve_value()
        assert isinstance(resolved, WeaviateAuthApiKey)
        assert resolved.api_key == "fake_key"


class TestAuthBearerToken:
    def test_init(self):
        credentials = AuthBearerToken()
        assert credentials.access_token._env_vars == ("WEAVIATE_ACCESS_TOKEN",)
        assert credentials.access_token._strict
        assert credentials.expires_in == 60
        assert credentials.refresh_token._env_vars == ("WEAVIATE_REFRESH_TOKEN",)
        assert not credentials.refresh_token._strict

    def test_to_dict(self):
        credentials = AuthBearerToken()
        assert credentials.to_dict() == {
            "type": "bearer",
            "init_parameters": {
                "access_token": {"env_vars": ["WEAVIATE_ACCESS_TOKEN"], "strict": True, "type": "env_var"},
                "expires_in": 60,
                "refresh_token": {"env_vars": ["WEAVIATE_REFRESH_TOKEN"], "strict": False, "type": "env_var"},
            },
        }

    def test_from_dict(self):
        credentials = AuthCredentials.from_dict(
            {
                "type": "bearer",
                "init_parameters": {
                    "access_token": {"env_vars": ["WEAVIATE_ACCESS_TOKEN"], "strict": True, "type": "env_var"},
                    "expires_in": 10,
                    "refresh_token": {"env_vars": ["WEAVIATE_REFRESH_TOKEN"], "strict": False, "type": "env_var"},
                },
            }
        )
        assert credentials.access_token._env_vars == ("WEAVIATE_ACCESS_TOKEN",)
        assert credentials.access_token._strict
        assert credentials.expires_in == 10
        assert credentials.refresh_token._env_vars == ("WEAVIATE_REFRESH_TOKEN",)
        assert not credentials.refresh_token._strict

    def test_resolve_value(self, monkeypatch):
        monkeypatch.setenv("WEAVIATE_ACCESS_TOKEN", "fake_key")
        monkeypatch.setenv("WEAVIATE_REFRESH_TOKEN", "fake_refresh_token")
        credentials = AuthBearerToken(expires_in=10)
        resolved = credentials.resolve_value()
        assert isinstance(resolved, WeaviateAuthBearerToken)
        assert resolved.access_token == "fake_key"
        assert resolved.expires_in == 10
        assert resolved.refresh_token == "fake_refresh_token"


class TestAuthClientCredentials:
    def test_init(self):
        credentials = AuthClientCredentials()
        assert credentials.client_secret._env_vars == ("WEAVIATE_CLIENT_SECRET",)
        assert credentials.client_secret._strict
        assert credentials.scope._env_vars == ("WEAVIATE_SCOPE",)
        assert not credentials.scope._strict

    def test_to_dict(self):
        credentials = AuthClientCredentials()
        assert credentials.to_dict() == {
            "type": "client_credentials",
            "init_parameters": {
                "client_secret": {"env_vars": ["WEAVIATE_CLIENT_SECRET"], "strict": True, "type": "env_var"},
                "scope": {"env_vars": ["WEAVIATE_SCOPE"], "strict": False, "type": "env_var"},
            },
        }

    def test_from_dict(self):
        credentials = AuthCredentials.from_dict(
            {
                "type": "client_credentials",
                "init_parameters": {
                    "client_secret": {"env_vars": ["WEAVIATE_CLIENT_SECRET"], "strict": True, "type": "env_var"},
                    "scope": {"env_vars": ["WEAVIATE_SCOPE"], "strict": False, "type": "env_var"},
                },
            }
        )
        assert credentials.client_secret._env_vars == ("WEAVIATE_CLIENT_SECRET",)
        assert credentials.client_secret._strict
        assert credentials.scope._env_vars == ("WEAVIATE_SCOPE",)
        assert not credentials.scope._strict

    def test_resolve_value(self, monkeypatch):
        monkeypatch.setenv("WEAVIATE_CLIENT_SECRET", "fake_secret")
        monkeypatch.setenv("WEAVIATE_SCOPE", "fake_scope another_fake_scope")
        credentials = AuthClientCredentials()
        resolved = credentials.resolve_value()
        assert isinstance(resolved, WeaviateAuthClientCredentials)
        assert resolved.client_secret == "fake_secret"
        assert resolved.scope_list == ["fake_scope", "another_fake_scope"]


class TestAuthClientPassword:
    def test_init(self):
        credentials = AuthClientPassword()
        assert credentials.username._env_vars == ("WEAVIATE_USERNAME",)
        assert credentials.username._strict
        assert credentials.password._env_vars == ("WEAVIATE_PASSWORD",)
        assert credentials.password._strict
        assert credentials.scope._env_vars == ("WEAVIATE_SCOPE",)
        assert not credentials.scope._strict

    def test_to_dict(self):
        credentials = AuthClientPassword()
        assert credentials.to_dict() == {
            "type": "client_password",
            "init_parameters": {
                "username": {"env_vars": ["WEAVIATE_USERNAME"], "strict": True, "type": "env_var"},
                "password": {"env_vars": ["WEAVIATE_PASSWORD"], "strict": True, "type": "env_var"},
                "scope": {"env_vars": ["WEAVIATE_SCOPE"], "strict": False, "type": "env_var"},
            },
        }

    def test_from_dict(self):
        credentials = AuthCredentials.from_dict(
            {
                "type": "client_password",
                "init_parameters": {
                    "username": {"env_vars": ["WEAVIATE_USERNAME"], "strict": True, "type": "env_var"},
                    "password": {"env_vars": ["WEAVIATE_PASSWORD"], "strict": True, "type": "env_var"},
                    "scope": {"env_vars": ["WEAVIATE_SCOPE"], "strict": False, "type": "env_var"},
                },
            }
        )
        assert credentials.username._env_vars == ("WEAVIATE_USERNAME",)
        assert credentials.username._strict
        assert credentials.password._env_vars == ("WEAVIATE_PASSWORD",)
        assert credentials.password._strict
        assert credentials.scope._env_vars == ("WEAVIATE_SCOPE",)
        assert not credentials.scope._strict

    def test_resolve_value(self, monkeypatch):
        monkeypatch.setenv("WEAVIATE_USERNAME", "fake_username")
        monkeypatch.setenv("WEAVIATE_PASSWORD", "fake_password")
        monkeypatch.setenv("WEAVIATE_SCOPE", "fake_scope another_fake_scope")
        credentials = AuthClientPassword()
        resolved = credentials.resolve_value()
        assert isinstance(resolved, WeaviateAuthClientPassword)
        assert resolved.username == "fake_username"
        assert resolved.password == "fake_password"
        assert resolved.scope_list == ["fake_scope", "another_fake_scope"]
