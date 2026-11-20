# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import httpx
import jwt
import pytest
from haystack import Pipeline
from haystack.utils import Secret

from haystack_integrations.components.connectors.oauth import OAuthTokenResolver
from haystack_integrations.utils.oauth import (
    EntraAgentOBOTokenSource,
    OAuthConfigError,
    OAuthStaticTokenSource,
    TokenRefreshError,
)

ENTRA_MODULE = "haystack_integrations.utils.oauth.entra"
TOKEN_URL = "https://login.microsoftonline.com/tenant-1/oauth2/v2.0/token"
JWT_BEARER_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"

IDENTIFIERS = {
    "tenant_id": "tenant-1",
    "blueprint_client_id": "blueprint-1",
    "agent_identity_client_id": "agent-1",
    "scopes": ["https://graph.microsoft.com/.default"],
}

# The two request bodies of the Entra Agent ID on-behalf-of flow, as Microsoft documents them. The blueprint's own
# credential is added to the first one per test.
BLUEPRINT_TOKEN_REQUEST = {
    "grant_type": "client_credentials",
    "client_id": "blueprint-1",
    "scope": "api://AzureADTokenExchange/.default",
    "fmi_path": "agent-1",
}
ON_BEHALF_OF_REQUEST = {
    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
    "client_id": "agent-1",
    "client_assertion_type": JWT_BEARER_ASSERTION_TYPE,
    "client_assertion": "T1",
    "assertion": "user-token",
    "requested_token_use": "on_behalf_of",
    "scope": "https://graph.microsoft.com/.default",
}


def _json(status=200, **payload):
    return httpx.Response(status, json=payload)


def _flow_responses():
    return [_json(access_token="T1", expires_in=3600), _json(access_token="resource-token", expires_in=3600)]


class TestEntraAgentOBOTokenSource:
    def test_client_secret_flow_sends_the_two_documented_requests(self):
        resolver = OAuthTokenResolver(
            token_source=EntraAgentOBOTokenSource(**IDENTIFIERS, client_secret=Secret.from_token("secret-1"))
        )
        with patch("httpx.post", side_effect=_flow_responses()) as mock_post:
            access_token = resolver.run(subject_token="user-token")["access_token"]
        assert access_token.resolve_value() == "resource-token"
        blueprint_call, exchange_call = mock_post.call_args_list
        assert blueprint_call.args == (TOKEN_URL,)
        assert blueprint_call.kwargs["data"] == {**BLUEPRINT_TOKEN_REQUEST, "client_secret": "secret-1"}
        assert exchange_call.args == (TOKEN_URL,)
        assert exchange_call.kwargs["data"] == ON_BEHALF_OF_REQUEST

    def test_certificate_flow_signs_the_blueprint_assertion_as_entra_requires(self, signing_material):
        source = EntraAgentOBOTokenSource(
            **IDENTIFIERS,
            certificate=Secret.from_token(signing_material.certificate_pem),
            private_key=Secret.from_token(signing_material.private_key_pem),
        )
        with patch("httpx.post", side_effect=_flow_responses()) as mock_post:
            assert source.resolve("user-token") == "resource-token"
        blueprint_call, exchange_call = mock_post.call_args_list
        blueprint_request = dict(blueprint_call.kwargs["data"])
        assertion = blueprint_request.pop("client_assertion")
        assert blueprint_request == {**BLUEPRINT_TOKEN_REQUEST, "client_assertion_type": JWT_BEARER_ASSERTION_TYPE}
        # Entra: PS256 signature, `aud` is the tenant's token endpoint, `iss` and `sub` are the client ID.
        claims = jwt.decode(assertion, signing_material.public_key, algorithms=["PS256"], audience=TOKEN_URL)
        assert claims["iss"] == claims["sub"] == "blueprint-1"
        assert claims["exp"] - claims["nbf"] <= 600
        header = jwt.get_unverified_header(assertion)
        assert header["alg"] == "PS256"
        assert header["x5t#S256"] == signing_material.thumbprint
        assert exchange_call.kwargs["data"] == ON_BEHALF_OF_REQUEST

    def test_custom_client_assertion_source_authenticates_the_blueprint(self):
        source = EntraAgentOBOTokenSource(
            **IDENTIFIERS, client_assertion_source=OAuthStaticTokenSource(token=Secret.from_token("custom-assertion"))
        )
        with patch("httpx.post", side_effect=_flow_responses()) as mock_post:
            source.resolve("user-token")
        assert mock_post.call_args_list[0].kwargs["data"] == {
            **BLUEPRINT_TOKEN_REQUEST,
            "client_assertion_type": JWT_BEARER_ASSERTION_TYPE,
            "client_assertion": "custom-assertion",
        }

    def test_blueprint_token_is_reused_and_resource_tokens_are_cached_per_user(self):
        source = EntraAgentOBOTokenSource(**IDENTIFIERS, client_secret=Secret.from_token("secret-1"))
        responses = [_json(access_token=t, expires_in=3600) for t in ("T1", "resource-a", "resource-b")]
        with patch("httpx.post", side_effect=responses) as mock_post:
            assert source.resolve("user-a") == "resource-a"
            assert source.resolve("user-a") == "resource-a"
            assert source.resolve("user-b") == "resource-b"
        assert mock_post.call_count == 3
        assert mock_post.call_args.kwargs["data"] == {**ON_BEHALF_OF_REQUEST, "assertion": "user-b"}

    def test_sovereign_cloud_authority_and_audience(self):
        source = EntraAgentOBOTokenSource(
            **IDENTIFIERS,
            client_secret=Secret.from_token("secret-1"),
            authority="https://login.microsoftonline.us/",
            token_exchange_audience="api://AzureADTokenExchangeUSGov",
        )
        with patch("httpx.post", side_effect=_flow_responses()) as mock_post:
            source.resolve("user-token")
        us_token_url = "https://login.microsoftonline.us/tenant-1/oauth2/v2.0/token"
        assert [call.args for call in mock_post.call_args_list] == [(us_token_url,), (us_token_url,)]
        assert mock_post.call_args_list[0].kwargs["data"]["scope"] == "api://AzureADTokenExchangeUSGov/.default"

    def test_entra_error_is_reported_with_its_aadsts_code(self):
        source = EntraAgentOBOTokenSource(**IDENTIFIERS, client_secret=Secret.from_token("secret-1"))
        error = _json(401, error="invalid_client", error_description="AADSTS7000215: Invalid client secret provided.")
        with patch("httpx.post", return_value=error) as mock_post:
            with pytest.raises(TokenRefreshError, match="AADSTS7000215") as raised:
                source.resolve("user-token")
        assert raised.value.status_code == 401
        assert mock_post.call_count == 1  # the exchange is not attempted without a blueprint token

    def test_empty_subject_token_raises(self):
        source = EntraAgentOBOTokenSource(**IDENTIFIERS, client_secret=Secret.from_token("secret-1"))
        with pytest.raises(OAuthConfigError):
            source.resolve("")

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({}, id="no-credential"),
            pytest.param(
                {
                    "client_secret": Secret.from_token("s"),
                    "client_assertion_source": OAuthStaticTokenSource(token=Secret.from_token("assertion")),
                },
                id="two-credentials",
            ),
            pytest.param({"certificate": Secret.from_token("cert")}, id="certificate-without-private-key"),
            pytest.param({"private_key": Secret.from_token("key")}, id="private-key-without-certificate"),
            pytest.param({"client_secret": Secret.from_token("s"), "tenant_id": ""}, id="empty-tenant"),
            pytest.param({"client_secret": Secret.from_token("s"), "agent_identity_client_id": ""}, id="empty-agent"),
            pytest.param({"client_secret": Secret.from_token("s"), "scopes": []}, id="no-scopes"),
        ],
    )
    def test_invalid_config_raises(self, kwargs):
        with pytest.raises(OAuthConfigError):
            EntraAgentOBOTokenSource(**{**IDENTIFIERS, **kwargs})

    @pytest.mark.parametrize(
        "credential",
        [
            pytest.param({"client_secret": Secret.from_env_var("BLUEPRINT_SECRET")}, id="client-secret"),
            pytest.param(
                {
                    "certificate": Secret.from_env_var("BLUEPRINT_CERTIFICATE"),
                    "private_key": Secret.from_env_var("BLUEPRINT_PRIVATE_KEY"),
                },
                id="certificate",
            ),
            pytest.param(
                {"client_assertion_source": OAuthStaticTokenSource(token=Secret.from_env_var("BLUEPRINT_ASSERTION"))},
                id="client-assertion-source",
            ),
        ],
    )
    def test_round_trip(self, credential):
        source = EntraAgentOBOTokenSource(
            **IDENTIFIERS,
            **credential,
            authority="https://login.microsoftonline.us",
            token_exchange_audience="api://AzureADTokenExchangeUSGov",
            expiry_buffer_seconds=60,
            cache_max_size=10,
            timeout=5.0,
        )
        data = source.to_dict()
        assert data["type"] == f"{ENTRA_MODULE}.EntraAgentOBOTokenSource"
        assert data["init_parameters"]["tenant_id"] == "tenant-1"
        restored = EntraAgentOBOTokenSource.from_dict(source.to_dict())
        assert restored.to_dict() == data

    def test_restored_secret_resolves_and_resolver_keeps_its_mandatory_input(self, monkeypatch):
        monkeypatch.setenv("BLUEPRINT_SECRET", "secret-1")
        resolver = OAuthTokenResolver(
            token_source=EntraAgentOBOTokenSource(**IDENTIFIERS, client_secret=Secret.from_env_var("BLUEPRINT_SECRET"))
        )
        restored = OAuthTokenResolver.from_dict(resolver.to_dict())
        assert isinstance(restored.token_source, EntraAgentOBOTokenSource)

        pipe = Pipeline()
        pipe.add_component("oauth", restored)
        with pytest.raises(ValueError):  # missing mandatory `subject_token`, caught before the component runs
            pipe.run({})
        with patch("httpx.post", side_effect=_flow_responses()) as mock_post:
            output = pipe.run({"oauth": {"subject_token": "user-token"}})
        assert output["oauth"]["access_token"].resolve_value() == "resource-token"
        assert mock_post.call_args_list[0].kwargs["data"]["client_secret"] == "secret-1"

    @pytest.mark.asyncio
    async def test_run_async_sends_the_two_documented_requests(self):
        resolver = OAuthTokenResolver(
            token_source=EntraAgentOBOTokenSource(**IDENTIFIERS, client_secret=Secret.from_token("secret-1"))
        )
        with patch("httpx.AsyncClient.post", side_effect=_flow_responses()) as mock_post:
            access_token = (await resolver.run_async(subject_token="user-token"))["access_token"]
        assert access_token.resolve_value() == "resource-token"
        blueprint_call, exchange_call = mock_post.call_args_list
        assert blueprint_call.args == (TOKEN_URL,)
        assert blueprint_call.kwargs["data"] == {**BLUEPRINT_TOKEN_REQUEST, "client_secret": "secret-1"}
        assert exchange_call.kwargs["data"] == ON_BEHALF_OF_REQUEST
