from unittest.mock import MagicMock, patch
from haystack import Document
from haystack_integrations.components.connectors.llamahub.connector import (
    LlamaHubConnector,
)


class MockLlamaIndexDocument:

    def __init__(self, text, metadata):
        self.text = text
        self.metadata = metadata


def test_connector_initialization():
    """Verify the connector dynamically imports explicitly installed reader classes."""
    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_class = MagicMock()
        setattr(mock_module, "SimpleWebPageReader", mock_class)
        mock_import.return_value = mock_module

        connector = LlamaHubConnector(
            reader_module="llama_index.readers.web",
            reader_class="SimpleWebPageReader",
            reader_kwargs={"val": "test"},
        )

        assert connector.reader_module == "llama_index.readers.web"
        assert connector.reader_class == "SimpleWebPageReader"
        mock_import.assert_called_once_with("llama_index.readers.web")
        mock_class.assert_called_once_with(val="test")


def test_connector_run_execution():
    """Verify that run() successfully converts data node structures to Haystack 2.0 Documents."""
    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_instance = MagicMock()
        mock_instance.load_data.return_value = [
            MockLlamaIndexDocument(
                text="Hello Modern Integration",
                metadata={"url": "https://example.com"},
            )
        ]
        mock_class = MagicMock(return_value=mock_instance)
        setattr(mock_module, "SimpleWebPageReader", mock_class)
        mock_import.return_value = mock_module

        connector = LlamaHubConnector(
            reader_module="llama_index.readers.web",
            reader_class="SimpleWebPageReader",
        )
        results = connector.run(urls=["https://example.com"])

        assert "documents" in results
        documents = results["documents"]
        assert len(documents) == 1
        assert isinstance(documents[0], Document)
        assert documents[0].content == "Hello Modern Integration"
        assert documents[0].meta["url"] == "https://example.com"
