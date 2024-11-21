import json
import os
from unittest.mock import patch

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.connectors.jina import JinaReaderConnector, JinaReaderMode


class TestJinaReaderConnector:
    def test_init_with_custom_parameters(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "test-api-key")
        reader = JinaReaderConnector(mode="read", api_key=Secret.from_env_var("TEST_KEY"), json_response=False)

        assert reader.mode == JinaReaderMode.READ
        assert reader.api_key.resolve_value() == "test-api-key"
        assert reader.json_response is False

    def test_init_with_invalid_mode(self):
        with pytest.raises(ValueError):
            JinaReaderConnector(mode="INVALID")

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "test-api-key")
        reader = JinaReaderConnector(mode="search", api_key=Secret.from_env_var("TEST_KEY"), json_response=True)

        serialized = reader.to_dict()

        assert serialized["type"] == "haystack_integrations.components.connectors.jina.reader.JinaReaderConnector"
        assert "init_parameters" in serialized

        init_params = serialized["init_parameters"]
        assert init_params["mode"] == "search"
        assert init_params["json_response"] is True
        assert "api_key" in init_params
        assert init_params["api_key"]["type"] == "env_var"

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "test-api-key")
        component_dict = {
            "type": "haystack_integrations.components.connectors.jina.reader.JinaReaderConnector",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["JINA_API_KEY"], "strict": True},
                "mode": "read",
                "json_response": True,
            },
        }

        reader = JinaReaderConnector.from_dict(component_dict)

        assert isinstance(reader, JinaReaderConnector)
        assert reader.mode == JinaReaderMode.READ
        assert reader.json_response is True
        assert reader.api_key.resolve_value() == "test-api-key"

    def test_json_to_document_read_mode(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "test-api-key")
        reader = JinaReaderConnector(mode="read")

        data = {"content": "Mocked content", "title": "Mocked Title", "url": "https://example.com"}
        document = reader._json_to_document(data)

        assert isinstance(document, Document)
        assert document.content == "Mocked content"
        assert document.meta["title"] == "Mocked Title"
        assert document.meta["url"] == "https://example.com"

    def test_json_to_document_ground_mode(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "test-api-key")
        reader = JinaReaderConnector(mode="ground")

        data = {
            "factuality": 0,
            "result": False,
            "reason": "The statement is contradicted by...",
            "references": [{"url": "https://example.com", "keyQuote": "Mocked key quote", "isSupportive": False}],
        }

        document = reader._json_to_document(data)
        assert isinstance(document, Document)
        assert document.content == "The statement is contradicted by..."
        assert document.meta["factuality"] == 0
        assert document.meta["result"] is False
        assert document.meta["references"] == [
            {"url": "https://example.com", "keyQuote": "Mocked key quote", "isSupportive": False}
        ]

    @patch("requests.get")
    def test_run_with_mocked_response(self, mock_get, monkeypatch):
        monkeypatch.setenv("JINA_API_KEY", "test-api-key")
        mock_json_response = {
            "data": {"content": "Mocked content", "title": "Mocked Title", "url": "https://example.com"}
        }
        mock_get.return_value.content = json.dumps(mock_json_response).encode("utf-8")
        mock_get.return_value.headers = {"Content-Type": "application/json"}

        reader = JinaReaderConnector(mode="read")
        result = reader.run(query="https://example.com")

        assert mock_get.call_count == 1
        assert mock_get.call_args[0][0] == "https://r.jina.ai/https%3A%2F%2Fexample.com"
        assert mock_get.call_args[1]["headers"] == {
            "Authorization": "Bearer test-api-key",
            "Accept": "application/json",
        }

        assert len(result) == 1
        document = result["documents"][0]
        assert isinstance(document, Document)
        assert document.content == "Mocked content"
        assert document.meta["title"] == "Mocked Title"
        assert document.meta["url"] == "https://example.com"

    @pytest.mark.skipif(not os.environ.get("JINA_API_KEY", None), reason="JINA_API_KEY env var not set")
    @pytest.mark.integration
    def test_run_reader_mode(self):
        reader = JinaReaderConnector(mode="read")
        result = reader.run(query="https://example.com")

        assert len(result) == 1
        document = result["documents"][0]
        assert isinstance(document, Document)
        assert "This domain is for use in illustrative examples" in document.content
        assert document.meta["title"] == "Example Domain"
        assert document.meta["url"] == "https://example.com/"

    @pytest.mark.skipif(not os.environ.get("JINA_API_KEY", None), reason="JINA_API_KEY env var not set")
    @pytest.mark.integration
    def test_run_search_mode(self):
        reader = JinaReaderConnector(mode="search")
        result = reader.run(query="When was Jina AI founded?")

        assert len(result) >= 1
        for doc in result["documents"]:
            assert isinstance(doc, Document)
            assert doc.content
            assert "title" in doc.meta
            assert "url" in doc.meta
            assert "description" in doc.meta
