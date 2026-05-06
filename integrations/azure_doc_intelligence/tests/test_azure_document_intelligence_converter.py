# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.components.converters.azure_doc_intelligence import AzureDocumentIntelligenceConverter


def make_mock_analyze_result(content="# Title\n\nBody", pages=("p1",), as_dict_value=None):
    result = MagicMock()
    result.content = content
    result.pages = list(pages) if pages is not None else None
    result.as_dict.return_value = as_dict_value if as_dict_value is not None else {"content": content}
    return result


@pytest.fixture
def warmed_converter():
    converter = AzureDocumentIntelligenceConverter(
        endpoint="https://test.cognitiveservices.azure.com/",
        api_key=Secret.from_token("test_api_key"),
    )
    converter.client = MagicMock()
    converter.client.begin_analyze_document.return_value.result.return_value = make_mock_analyze_result()
    return converter


class TestAzureDocumentIntelligenceConverter:
    def test_init_default(self):
        """Test basic initialization with defaults"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint="https://test.cognitiveservices.azure.com/",
            api_key=Secret.from_token("test_api_key"),
        )

        assert converter.endpoint == "https://test.cognitiveservices.azure.com/"
        assert converter.model_id == "prebuilt-document"
        assert converter.store_full_path is False

    def test_init_custom_params(self):
        """Test initialization with custom parameters"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint="https://test.cognitiveservices.azure.com/",
            api_key=Secret.from_token("test_api_key"),
            model_id="prebuilt-layout",
            store_full_path=True,
        )

        assert converter.endpoint == "https://test.cognitiveservices.azure.com/"
        assert converter.model_id == "prebuilt-layout"
        assert converter.store_full_path is True

    def test_to_dict(self):
        """Test serialization with Secret handling"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint="https://test.cognitiveservices.azure.com/",
            api_key=Secret.from_env_var("AZURE_DI_API_KEY"),
            model_id="prebuilt-layout",
            store_full_path=True,
        )

        data = converter.to_dict()

        expected_type = (
            "haystack_integrations.components.converters.azure_doc_intelligence.converter."
            "AzureDocumentIntelligenceConverter"
        )
        assert data == {
            "type": expected_type,
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["AZURE_DI_API_KEY"], "strict": True},
                "endpoint": "https://test.cognitiveservices.azure.com/",
                "model_id": "prebuilt-layout",
                "store_full_path": True,
            },
        }

    def test_to_dict_default_params(self):
        """Test serialization with default parameters"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint="https://test.cognitiveservices.azure.com/",
            api_key=Secret.from_env_var("AZURE_DI_API_KEY"),
        )

        data = converter.to_dict()

        assert data["init_parameters"]["endpoint"] == "https://test.cognitiveservices.azure.com/"
        assert data["init_parameters"]["model_id"] == "prebuilt-document"
        assert data["init_parameters"]["store_full_path"] is False

    def test_from_dict(self):
        """Test deserialization"""
        expected_type = (
            "haystack_integrations.components.converters.azure_doc_intelligence.converter."
            "AzureDocumentIntelligenceConverter"
        )
        data = {
            "type": expected_type,
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["AZURE_DI_API_KEY"], "strict": True},
                "endpoint": "https://test.cognitiveservices.azure.com/",
                "model_id": "prebuilt-layout",
                "store_full_path": False,
            },
        }

        converter = AzureDocumentIntelligenceConverter.from_dict(data)

        assert converter.endpoint == "https://test.cognitiveservices.azure.com/"
        assert converter.model_id == "prebuilt-layout"
        assert converter.store_full_path is False

    def test_warm_up_initializes_client_only_once(self):
        converter = AzureDocumentIntelligenceConverter(
            endpoint="https://test.cognitiveservices.azure.com/",
            api_key=Secret.from_token("test_api_key"),
        )
        assert converter.client is None
        with patch(
            "haystack_integrations.components.converters.azure_doc_intelligence.converter.DocumentIntelligenceClient"
        ) as mock_client_cls:
            converter.warm_up()
            assert converter.client is mock_client_cls.return_value
            mock_client_cls.assert_called_once()
            converter.warm_up()
            mock_client_cls.assert_called_once()

    def test_run_calls_warm_up_when_client_is_none(self):
        converter = AzureDocumentIntelligenceConverter(
            endpoint="https://test.cognitiveservices.azure.com/",
            api_key=Secret.from_token("test_api_key"),
        )
        with patch(
            "haystack_integrations.components.converters.azure_doc_intelligence.converter.DocumentIntelligenceClient"
        ) as mock_client_cls:
            mock_client_cls.return_value.begin_analyze_document.return_value.result.return_value = (
                make_mock_analyze_result()
            )
            result = converter.run(sources=[ByteStream.from_string("data")])

        assert converter.client is mock_client_cls.return_value
        assert len(result["documents"]) == 1

    def test_run_returns_document_with_markdown_content_and_meta(self, warmed_converter):
        warmed_converter.client.begin_analyze_document.return_value.result.return_value = make_mock_analyze_result(
            content="# Heading\n\nHello", pages=("p1", "p2", "p3")
        )

        result = warmed_converter.run(sources=[ByteStream.from_string("data")])

        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert doc.content == "# Heading\n\nHello"
        assert doc.meta["model_id"] == "prebuilt-document"
        assert doc.meta["page_count"] == 3

    def test_run_returns_raw_azure_response(self, warmed_converter):
        raw_dict = {"content": "text", "pages": [{"page_number": 1}]}
        warmed_converter.client.begin_analyze_document.return_value.result.return_value = make_mock_analyze_result(
            content="text", as_dict_value=raw_dict
        )

        result = warmed_converter.run(sources=[ByteStream.from_string("data")])

        assert result["raw_azure_response"] == [raw_dict]

    def test_run_with_multiple_sources(self, warmed_converter):
        sources = [ByteStream.from_string("one"), ByteStream.from_string("two")]

        result = warmed_converter.run(sources=sources)

        assert len(result["documents"]) == 2
        assert len(result["raw_azure_response"]) == 2

    @pytest.mark.parametrize("store_full_path", [True, False])
    def test_run_respects_store_full_path(self, store_full_path):
        pdf_path = Path(__file__).parent / "test_files" / "pdf" / "sample_pdf_1.pdf"
        converter = AzureDocumentIntelligenceConverter(
            endpoint="https://test.cognitiveservices.azure.com/",
            api_key=Secret.from_token("test_api_key"),
            store_full_path=store_full_path,
        )
        converter.client = MagicMock()
        converter.client.begin_analyze_document.return_value.result.return_value = make_mock_analyze_result()

        result = converter.run(sources=[str(pdf_path)])

        expected_path = str(pdf_path) if store_full_path else "sample_pdf_1.pdf"
        assert result["documents"][0].meta["file_path"] == expected_path

    def test_run_applies_single_meta_dict_to_all_documents(self, warmed_converter):
        sources = [ByteStream.from_string("one"), ByteStream.from_string("two")]

        result = warmed_converter.run(sources=sources, meta={"shared": "value"})

        assert all(doc.meta["shared"] == "value" for doc in result["documents"])

    def test_run_applies_meta_list_pairwise(self, warmed_converter):
        sources = [ByteStream.from_string("a"), ByteStream.from_string("b")]

        result = warmed_converter.run(sources=sources, meta=[{"index": 0}, {"index": 1}])

        assert result["documents"][0].meta["index"] == 0
        assert result["documents"][1].meta["index"] == 1

    def test_run_skips_unreadable_source(self, warmed_converter):
        result = warmed_converter.run(sources=["/nonexistent/missing.pdf"])

        assert result["documents"] == []
        assert result["raw_azure_response"] == []

    def test_run_skips_source_when_azure_analysis_fails(self, warmed_converter):
        warmed_converter.client.begin_analyze_document.side_effect = RuntimeError("Azure failure")

        result = warmed_converter.run(sources=[ByteStream.from_string("data")])

        assert result["documents"] == []
        assert result["raw_azure_response"] == []

    def test_run_uses_empty_string_when_result_content_is_none(self, warmed_converter):
        warmed_converter.client.begin_analyze_document.return_value.result.return_value = make_mock_analyze_result(
            content=None
        )

        result = warmed_converter.run(sources=[ByteStream.from_string("data")])

        assert result["documents"][0].content == ""

    def test_run_sets_page_count_zero_when_result_has_no_pages(self, warmed_converter):
        warmed_converter.client.begin_analyze_document.return_value.result.return_value = make_mock_analyze_result(
            pages=None
        )

        result = warmed_converter.run(sources=[ByteStream.from_string("data")])

        assert result["documents"][0].meta["page_count"] == 0


@pytest.mark.integration
class TestAzureDocumentIntelligenceConverterIntegration:
    @pytest.fixture
    def test_files_path(self):
        """Provide path to test files"""
        return Path(__file__).parent / "test_files"

    @pytest.mark.skipif(not os.environ.get("AZURE_DI_ENDPOINT"), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("AZURE_DI_API_KEY"), reason="Azure credentials not available")
    def test_run_markdown_output(self, test_files_path):
        """Integration test with real Azure API - markdown mode"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint=os.environ["AZURE_DI_ENDPOINT"],
            api_key=Secret.from_env_var("AZURE_DI_API_KEY"),
        )
        results = converter.run(sources=[test_files_path / "pdf" / "sample_pdf_1.pdf"])

        assert "documents" in results
        assert len(results["documents"]) == 1
        assert len(results["documents"][0].content) > 0
        assert results["documents"][0].meta["model_id"] == "prebuilt-document"

    @pytest.mark.skipif(not os.environ.get("AZURE_DI_ENDPOINT"), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("AZURE_DI_API_KEY"), reason="Azure credentials not available")
    def test_run_with_metadata(self, test_files_path):
        """Integration test - verify metadata handling"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint=os.environ["AZURE_DI_ENDPOINT"],
            api_key=Secret.from_env_var("AZURE_DI_API_KEY"),
            store_full_path=False,
        )
        results = converter.run(
            sources=[test_files_path / "pdf" / "sample_pdf_1.pdf"],
            meta={"custom_key": "custom_value"},
        )

        doc = results["documents"][0]
        assert doc.meta["custom_key"] == "custom_value"
        assert doc.meta["file_path"] == "sample_pdf_1.pdf"

    @pytest.mark.skipif(not os.environ.get("AZURE_DI_ENDPOINT"), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("AZURE_DI_API_KEY"), reason="Azure credentials not available")
    def test_run_with_multiple_files(self, test_files_path):
        """Integration test - process multiple files"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint=os.environ["AZURE_DI_ENDPOINT"],
            api_key=Secret.from_env_var("AZURE_DI_API_KEY"),
        )
        results = converter.run(
            sources=[
                test_files_path / "pdf" / "sample_pdf_1.pdf",
                test_files_path / "pdf" / "sample_pdf_2.pdf",
            ]
        )

        assert "documents" in results
        assert len(results["documents"]) == 2

    @pytest.mark.skipif(not os.environ.get("AZURE_DI_ENDPOINT"), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("AZURE_DI_API_KEY"), reason="Azure credentials not available")
    def test_run_with_prebuilt_layout(self, test_files_path):
        """Integration test with prebuilt-layout model for better table detection"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint=os.environ["AZURE_DI_ENDPOINT"],
            api_key=Secret.from_env_var("AZURE_DI_API_KEY"),
            model_id="prebuilt-layout",
        )
        results = converter.run(sources=[test_files_path / "pdf" / "sample_pdf_1.pdf"])

        assert "documents" in results
        assert len(results["documents"]) == 1
        assert results["documents"][0].meta["model_id"] == "prebuilt-layout"

    @pytest.mark.skipif(not os.environ.get("AZURE_DI_ENDPOINT"), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("AZURE_DI_API_KEY"), reason="Azure credentials not available")
    def test_run_with_jpg_image(self, test_files_path):
        """Integration test - convert JPG image with text"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint=os.environ["AZURE_DI_ENDPOINT"],
            api_key=Secret.from_env_var("AZURE_DI_API_KEY"),
        )
        results = converter.run(sources=[test_files_path / "images" / "sample_text.jpg"])

        assert "documents" in results
        assert len(results["documents"]) == 1
        doc = results["documents"][0]
        assert len(doc.content) > 0
        # Verify OCR extracted some expected text from the image
        assert "Sample" in doc.content or "OCR" in doc.content or "Azure" in doc.content

    @pytest.mark.skipif(not os.environ.get("AZURE_DI_ENDPOINT"), reason="Azure endpoint not available")
    @pytest.mark.skipif(not os.environ.get("AZURE_DI_API_KEY"), reason="Azure credentials not available")
    def test_run_with_docx(self, test_files_path):
        """Integration test - convert DOCX document"""
        converter = AzureDocumentIntelligenceConverter(
            endpoint=os.environ["AZURE_DI_ENDPOINT"],
            api_key=Secret.from_env_var("AZURE_DI_API_KEY"),
        )
        results = converter.run(sources=[test_files_path / "docx" / "sample.docx"])

        assert "documents" in results
        assert len(results["documents"]) == 1
        doc = results["documents"][0]
        assert len(doc.content) > 0
        # Verify some expected content from the DOCX
        assert "Sample Document" in doc.content or "sample" in doc.content.lower()
