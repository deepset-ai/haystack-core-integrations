# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

import pytest
from haystack import Pipeline
from haystack.utils import Secret

from haystack_integrations.components.connectors.oauth import (
    OAuthResolver,
    RefreshTokenSource,
    TokenExchangeSource,
)
from haystack_integrations.components.connectors.oauth.errors import OAuthConfigError

TOKEN_URL = "https://idp.example.com/oauth2/token"


class _FakeConfigSource:
    """Config-only TokenSource: resolves with no per-request input."""

    requires_subject_token = False

    def __init__(self, token: str = "fake-token") -> None:
        self.token = token
        self.calls = 0

    def resolve(self) -> str:
        self.calls += 1
        return self.token

    async def resolve_async(self) -> str:
        self.calls += 1
        return self.token

    def to_dict(self) -> dict[str, Any]:
        return {"type": "tests.test_resolver._FakeConfigSource", "init_parameters": {}}


class _FakeExchangeSource:
    """Request-scoped SubjectTokenSource: needs a subject_token each call."""

    requires_subject_token = True

    def __init__(self, token: str = "fake-token") -> None:
        self.token = token
        self.subject_tokens: list[str] = []

    def resolve(self, subject_token: str) -> str:
        self.subject_tokens.append(subject_token)
        return self.token

    async def resolve_async(self, subject_token: str) -> str:
        self.subject_tokens.append(subject_token)
        return self.token

    def to_dict(self) -> dict[str, Any]:
        return {"type": "tests.test_resolver._FakeExchangeSource", "init_parameters": {}}


class TestOAuthResolverInit:
    def test_rejects_non_token_source(self):
        with pytest.raises(OAuthConfigError):
            OAuthResolver(token_source=object())

    def test_accepts_config_source(self):
        resolver = OAuthResolver(token_source=_FakeConfigSource())
        assert isinstance(resolver.token_source, _FakeConfigSource)

    def test_accepts_exchange_source(self):
        resolver = OAuthResolver(token_source=_FakeExchangeSource())
        assert isinstance(resolver.token_source, _FakeExchangeSource)


class TestOAuthResolverRun:
    def test_config_source_delegates_with_no_input(self):
        source = _FakeConfigSource(token="resolved")
        resolver = OAuthResolver(token_source=source)
        assert resolver.run() == {"access_token": "resolved"}
        assert source.calls == 1

    def test_exchange_source_forwards_subject_token(self):
        source = _FakeExchangeSource(token="resolved")
        resolver = OAuthResolver(token_source=source)
        resolver.run(subject_token="user-jwt")
        assert source.subject_tokens == ["user-jwt"]

    def test_exchange_source_without_subject_token_raises(self):
        # Direct (non-pipeline) call with no subject_token -> the source guard fires.
        resolver = OAuthResolver(token_source=TokenExchangeSource(token_url=TOKEN_URL, client_id="c"))
        with pytest.raises(OAuthConfigError):
            resolver.run()

    @pytest.mark.asyncio
    async def test_config_source_run_async(self):
        source = _FakeConfigSource(token="resolved-async")
        resolver = OAuthResolver(token_source=source)
        assert await resolver.run_async() == {"access_token": "resolved-async"}

    @pytest.mark.asyncio
    async def test_exchange_source_run_async_forwards_subject_token(self):
        source = _FakeExchangeSource(token="resolved-async")
        resolver = OAuthResolver(token_source=source)
        await resolver.run_async(subject_token="user-jwt")
        assert source.subject_tokens == ["user-jwt"]


class TestOAuthResolverInPipeline:
    def test_config_resolver_is_source_node(self):
        pipe = Pipeline()
        pipe.add_component("oauth", OAuthResolver(token_source=_FakeConfigSource(token="t")))
        out = pipe.run({})  # no inputs needed — config-only resolver is a source node
        assert out["oauth"]["access_token"] == "t"

    def test_exchange_resolver_requires_subject_token_at_wire_time(self):
        pipe = Pipeline()
        pipe.add_component("oauth", OAuthResolver(token_source=_FakeExchangeSource(token="t")))
        with pytest.raises(ValueError):  # missing mandatory input, caught before any component runs
            pipe.run({})
        out = pipe.run({"oauth": {"subject_token": "user-jwt"}})
        assert out["oauth"]["access_token"] == "t"


class TestOAuthResolverSerialization:
    def test_round_trip_config_source(self, monkeypatch):
        monkeypatch.setenv("MS_REFRESH_TOKEN", "rt")
        resolver = OAuthResolver(
            token_source=RefreshTokenSource(
                token_url=TOKEN_URL,
                client_id="client-1",
                refresh_token=Secret.from_env_var("MS_REFRESH_TOKEN"),
                scopes=["Files.Read.All"],
            ),
        )

        data = resolver.to_dict()
        assert data["type"] == "haystack_integrations.components.connectors.oauth.resolver.OAuthResolver"
        assert data["init_parameters"]["token_source"]["type"].endswith("RefreshTokenSource")

        restored = OAuthResolver.from_dict(resolver.to_dict())
        assert isinstance(restored.token_source, RefreshTokenSource)
        assert restored.to_dict() == data

    def test_round_trip_exchange_source_rebuilds_mandatory_socket(self):
        resolver = OAuthResolver(token_source=TokenExchangeSource(token_url=TOKEN_URL, client_id="c"))
        restored = OAuthResolver.from_dict(resolver.to_dict())
        assert isinstance(restored.token_source, TokenExchangeSource)

        # The dynamic mandatory socket must survive serialization.
        pipe = Pipeline()
        pipe.add_component("oauth", restored)
        with pytest.raises(ValueError):
            pipe.run({})

    def test_from_dict_unknown_source_type_raises(self):
        data = {
            "type": "haystack_integrations.components.connectors.oauth.resolver.OAuthResolver",
            "init_parameters": {"token_source": {"type": "some.unknown.Source", "init_parameters": {}}},
        }
        with pytest.raises(ImportError):
            OAuthResolver.from_dict(data)


_REAL_IDP_VARS = ("OAUTH_TOKEN_URL", "OAUTH_CLIENT_ID", "OAUTH_REFRESH_TOKEN")


@pytest.mark.integration
@pytest.mark.skipif(
    not all(os.environ.get(var) for var in _REAL_IDP_VARS),
    reason=f"{', '.join(_REAL_IDP_VARS)} env vars not set",
)
def test_run_against_real_idp():
    resolver = OAuthResolver(
        token_source=RefreshTokenSource(
            token_url=os.environ["OAUTH_TOKEN_URL"],
            client_id=os.environ["OAUTH_CLIENT_ID"],
            refresh_token=Secret.from_env_var("OAUTH_REFRESH_TOKEN"),
            client_secret=Secret.from_env_var("OAUTH_CLIENT_SECRET", strict=False),
            scopes=os.environ["OAUTH_SCOPES"].split() if os.environ.get("OAUTH_SCOPES") else None,
        ),
    )
    assert resolver.run()["access_token"]
