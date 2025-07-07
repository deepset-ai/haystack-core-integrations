from unittest.mock import patch

import pytest
from haystack.utils import Secret

from haystack_integrations.components.common.google_genai.utils import _get_client


def test_get_client_invalid_api_raises():
    with pytest.raises(ValueError):
        _get_client(
            api_key=Secret.from_token("test-api-key"), api="invalid", vertex_ai_project=None, vertex_ai_location=None
        )


def test_get_client_vertex_no_credentials_raises(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    api_key = Secret.from_env_var("GOOGLE_API_KEY", strict=False)
    with pytest.raises(ValueError):
        _get_client(api_key=api_key, api="vertex", vertex_ai_project=None, vertex_ai_location=None)


def test_get_client_vertex_project_and_location(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    api_key = Secret.from_env_var("GOOGLE_API_KEY", strict=False)

    with patch("haystack_integrations.components.common.google_genai.utils.Client") as mock_client:
        client = _get_client(
            api_key=api_key, api="vertex", vertex_ai_project="test-project", vertex_ai_location="test-location"
        )
        mock_client.assert_called_once_with(vertexai=True, project="test-project", location="test-location")
    assert client is not None


def test_get_client_vertex_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
    api_key = Secret.from_env_var("GOOGLE_API_KEY", strict=False)

    with patch("haystack_integrations.components.common.google_genai.utils.Client") as mock_client:
        client = _get_client(api_key=api_key, api="vertex", vertex_ai_project=None, vertex_ai_location=None)
        mock_client.assert_called_once_with(vertexai=True, api_key="test-api-key")
    assert client is not None


def test_get_client_gemini_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    api_key = Secret.from_env_var("GEMINI_API_KEY", strict=False)

    with patch("haystack_integrations.components.common.google_genai.utils.Client") as mock_client:
        client = _get_client(api_key=api_key, api="gemini", vertex_ai_project=None, vertex_ai_location=None)
        mock_client.assert_called_once_with(api_key="test-api-key")
    assert client is not None


def test_get_client_gemini_api_key_no_env_var_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    api_key = Secret.from_env_var("GEMINI_API_KEY", strict=False)
    with pytest.raises(ValueError):
        _get_client(api_key=api_key, api="gemini", vertex_ai_project=None, vertex_ai_location=None)
