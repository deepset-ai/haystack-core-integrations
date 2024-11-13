import os
from unittest.mock import patch

import pytest
from haystack.utils import Secret

from haystack_integrations.components.connectors.jina import JinaReaderConnector, JinaReaderMode

os.environ["JINA_API_KEY"] = "test-api-key"
os.environ["TEST_KEY"] = "test-api-key"


class TestJinaReaderConnector:
    @pytest.fixture
    def mock_session(self):
        with patch("requests.Session") as mock:
            yield mock.return_value

    def test_init_with_custom_parameters(self):
        reader = JinaReaderConnector(mode="READ", api_key=Secret.from_env_var("TEST_KEY"), json_response=False)

        assert reader.mode == JinaReaderMode.READ
        assert reader.api_key.resolve_value() == "test-api-key"
        assert reader.json_response is False
        assert "application/json" not in reader._session.headers.values()

    def test_to_dict(self):
        reader = JinaReaderConnector(mode="SEARCH", api_key=Secret.from_env_var("TEST_KEY"), json_response=True)

        serialized = reader.to_dict()

        assert serialized["type"] == "haystack_integrations.components.connectors.jina.reader.JinaReaderConnector"
        assert "init_parameters" in serialized

        init_params = serialized["init_parameters"]
        assert init_params["mode"] == "SEARCH"
        assert init_params["json_response"] is True
        assert "api_key" in init_params
        assert init_params["api_key"]["type"] == "env_var"

    def test_from_dict(self):
        component_dict = {
            "type": "haystack_integrations.components.connectors.jina.reader.JinaReaderConnector",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["JINA_API_KEY"], "strict": True},
                "mode": "READ",
                "json_response": True,
            },
        }

        reader = JinaReaderConnector.from_dict(component_dict)

        assert isinstance(reader, JinaReaderConnector)
        assert reader.mode == JinaReaderMode.READ
        assert reader.json_response is True
        assert reader.api_key.resolve_value() == "test-api-key"
