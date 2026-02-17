# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from haystack_integrations.components.generators.google_genai.chat.cache_creator import (
    GoogleGenAICacheCreator,
)


@pytest.fixture
def mock_client():
    client = Mock()
    cache = Mock()
    cache.name = "projects/123/locations/us-central1/cachedContents/abc"
    expire_time = Mock()
    expire_time.isoformat.return_value = "2025-12-31T23:59:59+00:00"
    cache.expire_time = expire_time
    usage = Mock()
    usage.total_token_count = 1500
    cache.usage_metadata = usage
    client.caches.create.return_value = cache
    return client


class TestGoogleGenAICacheCreator:
    def test_run_success(self, monkeypatch, mock_client):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        with patch(
            "haystack_integrations.components.generators.google_genai.chat.cache_creator._get_client",
            return_value=mock_client,
        ):
            component = GoogleGenAICacheCreator(model="gemini-2.0-flash-001")
            result = component.run(
                contents=["Some long enough text content to cache. " * 50],
                display_name="test-cache",
                ttl="3600s",
            )
        assert result["cache_name"] == "projects/123/locations/us-central1/cachedContents/abc"
        assert result["expire_time"] == "2025-12-31T23:59:59+00:00"
        assert result["total_token_count"] == 1500
        mock_client.caches.create.assert_called_once()
        call_kwargs = mock_client.caches.create.call_args
        assert call_kwargs[1]["model"] == "gemini-2.0-flash-001"
        assert call_kwargs[1]["config"].display_name == "test-cache"
        assert call_kwargs[1]["config"].ttl == "3600s"

    def test_run_with_system_instruction(self, monkeypatch, mock_client):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        with patch(
            "haystack_integrations.components.generators.google_genai.chat.cache_creator._get_client",
            return_value=mock_client,
        ):
            component = GoogleGenAICacheCreator(model="gemini-2.0-flash-001")
            result = component.run(
                contents=["Content to cache." * 30],
                system_instruction="You are a helpful assistant.",
                ttl="7200s",
            )
        assert result["cache_name"] == "projects/123/locations/us-central1/cachedContents/abc"
        call_kwargs = mock_client.caches.create.call_args
        assert call_kwargs[1]["config"].system_instruction == "You are a helpful assistant."

    def test_run_empty_contents_raises(self, monkeypatch, mock_client):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        with patch(
            "haystack_integrations.components.generators.google_genai.chat.cache_creator._get_client",
            return_value=mock_client,
        ):
            component = GoogleGenAICacheCreator()
            with pytest.raises(ValueError, match="contents must be a non-empty list"):
                component.run(contents=[])
        mock_client.caches.create.assert_not_called()

    def test_run_api_error_propagates(self, monkeypatch, mock_client):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        mock_client.caches.create.side_effect = Exception("API error 400")
        with patch(
            "haystack_integrations.components.generators.google_genai.chat.cache_creator._get_client",
            return_value=mock_client,
        ):
            component = GoogleGenAICacheCreator()
            with pytest.raises(Exception, match="API error 400"):
                component.run(contents=["Valid content here. " * 30])

    def test_to_dict_from_dict_roundtrip(self, monkeypatch, mock_client):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        with patch(
            "haystack_integrations.components.generators.google_genai.chat.cache_creator._get_client",
            return_value=mock_client,
        ):
            component = GoogleGenAICacheCreator(
                model="gemini-2.0-flash-001",
                vertex_ai_project="my-project",
                vertex_ai_location="us-central1",
            )
            data = component.to_dict()
            restored = GoogleGenAICacheCreator.from_dict(data)

        assert restored._model == component._model
        assert restored._vertex_ai_project == component._vertex_ai_project
        assert restored._vertex_ai_location == component._vertex_ai_location

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        with patch(
            "haystack_integrations.components.generators.google_genai.chat.cache_creator._get_client",
            return_value=Mock(),
        ):
            component = GoogleGenAICacheCreator()
        assert component._model == "gemini-2.0-flash-001"
        assert component._api is not None

    def test__contents_to_config_parts_valid(self):
        result = GoogleGenAICacheCreator._contents_to_config_parts(["Hello", "World"])
        assert len(result) == 1
        assert result[0].role == "user"
        assert len(result[0].parts) == 2
        assert result[0].parts[0].text == "Hello"
        assert result[0].parts[1].text == "World"
