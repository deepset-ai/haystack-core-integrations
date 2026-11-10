# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack_integrations.components.embedders.cohere import (
    CohereDocumentEmbedder,
    CohereDocumentImageEmbedder,
    CohereTextEmbedder,
)
from haystack_integrations.components.generators.cohere import CohereChatGenerator
from haystack_integrations.components.rankers.cohere import CohereRanker
from haystack_integrations.utils.cohere import validate_api_base_url


class TestValidateApiBaseUrl:
    @pytest.mark.parametrize(
        "api_base_url",
        [
            "https://api.cohere.com",
            "https://api.cohere.com/",
            "https://resource.services.ai.azure.com/models",
            # only a trailing endpoint path is rejected, not one in the middle of the base path
            "https://gateway.example.com/v2/rerank/proxy",
        ],
    )
    def test_accepts_base_urls(self, api_base_url):
        validate_api_base_url(api_base_url)

    @pytest.mark.parametrize(
        "api_base_url",
        [
            "https://api.cohere.com/v1/rerank",
            "https://api.cohere.com/v2/embed",
            "https://api.cohere.com/v2/chat",
            # a future API version is rejected without needing a new entry in the pattern
            "https://api.cohere.com/v3/rerank",
        ],
    )
    def test_rejects_endpoint_urls(self, api_base_url):
        with pytest.raises(ValueError, match="must be a base URL"):
            validate_api_base_url(api_base_url)

    @pytest.mark.parametrize("suffix", ["", "/", "//"])
    def test_error_names_the_endpoint_path_and_the_corrected_url(self, suffix):
        url = f"https://resource.services.ai.azure.com/models/v2/rerank{suffix}"
        with pytest.raises(ValueError) as exc_info:
            validate_api_base_url(url)

        message = str(exc_info.value)
        assert "remove the trailing '/v2/rerank'" in message
        assert "pass 'https://resource.services.ai.azure.com/models' instead" in message

    def test_omits_the_hint_when_nothing_would_remain(self):
        with pytest.raises(ValueError) as exc_info:
            validate_api_base_url("/v2/rerank")

        assert "instead" not in str(exc_info.value)


@pytest.mark.parametrize(
    "component_cls",
    [
        CohereRanker,
        CohereTextEmbedder,
        CohereDocumentEmbedder,
        CohereDocumentImageEmbedder,
        CohereChatGenerator,
    ],
)
def test_components_reject_full_endpoint_urls(component_cls, monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-api-key")

    with pytest.raises(ValueError, match="must be a base URL"):
        component_cls(api_base_url="https://resource.services.ai.azure.com/models/v2/rerank")
