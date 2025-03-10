# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack_integrations.utils.nvidia import is_hosted
from haystack_integrations.utils.nvidia.models import CHAT_MODEL_TABLE, EMBEDDING_MODEL_TABLE, RANKING_MODEL_TABLE
from haystack_integrations.utils.nvidia.utils import (
    determine_model,
    lookup_model,
    url_validation,
    validate_hosted_model,
)


# url_validation
def test_url_validation() -> None:
    api_url = "https://integrate.api.nvidia.com/v1"
    assert api_url == url_validation(api_url)


def test_url_validation_not_ending_with_v1() -> None:
    with pytest.warns(UserWarning, match="you may have inference and listing issues"):
        api_url = url_validation("https://integrate.api.nvidia.com")
        assert api_url.endswith("/v1")


def test_url_validation_invalid_format() -> None:
    with pytest.raises(ValueError, match="Expected format is"):
        url_validation("not-a-domain")


# is_hosted
@pytest.mark.parametrize(
    "api_url", ["https://integrate.api.nvidia.com/v1", "https://ai.api.nvidia.com/v1/retrieval/nvidia"]
)
def test_is_hosted(api_url) -> None:
    assert is_hosted(api_url)


@pytest.mark.parametrize("api_url", ["https://example.com", "http://localhost:8000", "https://api.different.com"])
def test_is_hosted_false(api_url) -> None:
    assert is_hosted(api_url) is False


# lookup_model
@pytest.mark.parametrize(
    "name, model",
    [
        ("meta/codellama-70b", CHAT_MODEL_TABLE["meta/codellama-70b"]),
        ("nv-rerank-qa-mistral-4b:1", RANKING_MODEL_TABLE["nv-rerank-qa-mistral-4b:1"]),
        ("NV-Embed-QA", EMBEDDING_MODEL_TABLE["NV-Embed-QA"]),
        ("nvidia/nv-embed-v1", EMBEDDING_MODEL_TABLE["nvidia/nv-embed-v1"]),
    ],
)
def test_lookup_model_found(name, model) -> None:
    assert lookup_model(name) == model


def test_lookup_model_found_alias() -> None:
    assert lookup_model("ai-embed-qa-4") == EMBEDDING_MODEL_TABLE["NV-Embed-QA"]


def test_lookup_model_not_found() -> None:
    assert lookup_model("not-a-model") is None


# determine_model
def test_determine_model() -> None:
    assert determine_model("NV-Embed-QA") == EMBEDDING_MODEL_TABLE["NV-Embed-QA"]


def test_determine_model_alias() -> None:
    with pytest.warns(UserWarning, match="is deprecated"):
        assert determine_model("ai-embed-qa-4") == EMBEDDING_MODEL_TABLE["NV-Embed-QA"]


def test_determine_model_not_found() -> None:
    assert determine_model("not-a-model") is None


# validate_hosted_model
def test_validate_hosted_model_no_model_client() -> None:
    with pytest.warns(UserWarning, match="determine validity"):
        assert validate_hosted_model("snowflake/arctic-embed-l")


def test_validate_hosted_model_client_incompatible() -> None:
    with pytest.raises(ValueError, match="is incompatible"):
        assert validate_hosted_model("snowflake/arctic-embed-l", "NvidiaGenerator")  # has no client

    with pytest.raises(ValueError, match="is incompatible"):
        assert validate_hosted_model("meta/codellama-70b", "NvidiaRanker")


def test_validate_hosted_model_is_unknown() -> None:
    with pytest.raises(ValueError, match="is unknown"):
        assert validate_hosted_model("not-a-model", "NvidiaGenerator")
    with pytest.raises(ValueError, match="is unknown"):
        assert validate_hosted_model("not-a-model")


def test_validate_hosted_model_without_client() -> None:
    assert validate_hosted_model("snowflake/arctic-embed-l", "NvidiaTextEmbedder")


def test_validate_hosted_model_with_client() -> None:
    """Test when model's client matches the provided client."""
    model = validate_hosted_model("meta/codellama-70b", "NvidiaGenerator")
    assert model is not None
    assert model.client == "NvidiaGenerator"
