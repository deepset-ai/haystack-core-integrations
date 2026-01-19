# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest
from haystack.utils import Secret

from haystack_integrations.components.converters.azure_doc_intelligence import AzureDocumentIntelligenceConverter


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
