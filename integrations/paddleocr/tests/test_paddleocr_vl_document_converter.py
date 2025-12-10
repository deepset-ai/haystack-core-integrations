# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import MagicMock, patch

import pypdfium2 as pdfium
import pytest
import requests
from haystack import Document
from haystack.dataclasses import ByteStream
from haystack.utils import Secret
from PIL import Image

from haystack_integrations.components.converters.paddleocr import (
    PaddleOCRVLDocumentConverter,
)


def create_empty_pdf(tmp_path, filename="test.pdf"):
    """Create an empty PDF file using pypdfium2."""
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(595, 842)  # A4 size in points
    pdf.save(tmp_path / filename)
    return tmp_path / filename


def create_empty_image(tmp_path, filename="test.png"):
    """Create an empty image file using PIL."""
    img = Image.new("RGB", (800, 600), color="white")
    img.save(tmp_path / filename)
    return tmp_path / filename


class TestPaddleOCRVLDocumentConverter:
    CLASS_TYPE = "haystack_integrations.components.converters.paddleocr.paddleocr_vl_document_converter.PaddleOCRVLDocumentConverter"  # noqa: E501

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("AISTUDIO_ACCESS_TOKEN", "test-access-token")
        converter = PaddleOCRVLDocumentConverter(api_url="http://test-api-url.com")

        assert converter.access_token == Secret.from_env_var("AISTUDIO_ACCESS_TOKEN")
        assert converter.api_url == "http://test-api-url.com"
        assert converter.file_type is None
        assert converter.use_doc_orientation_classify is None
        assert converter.use_doc_unwarping is None
        assert converter.use_layout_detection is None
        assert converter.use_chart_recognition is None
        assert converter.layout_threshold is None
        assert converter.layout_nms is None
        assert converter.layout_unclip_ratio is None
        assert converter.layout_merge_bboxes_mode is None
        assert converter.prompt_label is None
        assert converter.format_block_content is None
        assert converter.repetition_penalty is None
        assert converter.temperature is None
        assert converter.top_p is None
        assert converter.min_pixels is None
        assert converter.max_pixels is None
        assert converter.prettify_markdown is None
        assert converter.show_formula_number is None
        assert converter.visualize is None
        assert converter.additional_params is None

    def test_init_with_all_optional_parameters(self):
        converter = PaddleOCRVLDocumentConverter(
            api_url="http://custom-api-url.com",
            access_token=Secret.from_token("test-access-token"),
            file_type="pdf",
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_layout_detection=True,
            use_chart_recognition=True,
            layout_threshold=0.5,
            layout_nms=True,
            layout_unclip_ratio=1.5,
            layout_merge_bboxes_mode="merge",
            prompt_label="ocr",
            format_block_content=True,
            repetition_penalty=1.1,
            temperature=0.7,
            top_p=0.9,
            min_pixels=100,
            max_pixels=1000,
            prettify_markdown=False,
            show_formula_number=True,
            visualize=True,
            additional_params={},
        )

        assert converter.api_url == "http://custom-api-url.com"
        assert converter.access_token == Secret.from_token("test-access-token")
        assert converter.file_type == 0  # "pdf" normalized to 0
        assert converter.use_doc_orientation_classify is True
        assert converter.use_doc_unwarping is True
        assert converter.use_layout_detection is True
        assert converter.use_chart_recognition is True
        assert converter.layout_threshold == 0.5
        assert converter.layout_nms is True
        assert converter.layout_unclip_ratio == 1.5
        assert converter.layout_merge_bboxes_mode == "merge"
        assert converter.prompt_label == "ocr"
        assert converter.format_block_content is True
        assert converter.repetition_penalty == 1.1
        assert converter.temperature == 0.7
        assert converter.top_p == 0.9
        assert converter.min_pixels == 100
        assert converter.max_pixels == 1000
        assert converter.prettify_markdown is False
        assert converter.show_formula_number is True
        assert converter.visualize is True
        assert converter.additional_params == {}

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("AISTUDIO_ACCESS_TOKEN", "test-access-token")
        converter = PaddleOCRVLDocumentConverter(api_url="http://test-api-url.com")
        converter_dict = converter.to_dict()

        assert converter_dict == {
            "type": self.CLASS_TYPE,
            "init_parameters": {
                "api_url": "http://test-api-url.com",
                "access_token": {
                    "env_vars": ["AISTUDIO_ACCESS_TOKEN"],
                    "strict": True,
                    "type": "env_var",
                },
                "file_type": None,
                "use_doc_orientation_classify": None,
                "use_doc_unwarping": None,
                "use_layout_detection": None,
                "use_chart_recognition": None,
                "layout_threshold": None,
                "layout_nms": None,
                "layout_unclip_ratio": None,
                "layout_merge_bboxes_mode": None,
                "prompt_label": None,
                "format_block_content": None,
                "repetition_penalty": None,
                "temperature": None,
                "top_p": None,
                "min_pixels": None,
                "max_pixels": None,
                "prettify_markdown": None,
                "show_formula_number": None,
                "visualize": None,
                "additional_params": None,
            },
        }

    def test_to_dict_with_custom_parameters(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_ACCESS_TOKEN", "test-access-token")
        converter = PaddleOCRVLDocumentConverter(
            api_url="http://custom-api-url.com",
            access_token=Secret.from_env_var("CUSTOM_ACCESS_TOKEN", strict=False),
            file_type="image",
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_layout_detection=True,
            use_chart_recognition=False,
            layout_threshold=0.7,
            layout_nms=False,
            layout_unclip_ratio=2.0,
            layout_merge_bboxes_mode="separate",
            prompt_label="formula",
            format_block_content=False,
            repetition_penalty=1.2,
            temperature=0.8,
            top_p=0.95,
            min_pixels=200,
            max_pixels=2000,
            prettify_markdown=True,
            show_formula_number=True,
            visualize=False,
            additional_params={},
        )
        converter_dict = converter.to_dict()

        assert converter_dict == {
            "type": self.CLASS_TYPE,
            "init_parameters": {
                "api_url": "http://custom-api-url.com",
                "access_token": {
                    "type": "env_var",
                    "env_vars": ["CUSTOM_ACCESS_TOKEN"],
                    "strict": False,
                },
                "file_type": 1,  # "image" normalized to 1
                "use_doc_orientation_classify": True,
                "use_doc_unwarping": False,
                "use_layout_detection": True,
                "use_chart_recognition": False,
                "layout_threshold": 0.7,
                "layout_nms": False,
                "layout_unclip_ratio": 2.0,
                "layout_merge_bboxes_mode": "separate",
                "prompt_label": "formula",
                "format_block_content": False,
                "repetition_penalty": 1.2,
                "temperature": 0.8,
                "top_p": 0.95,
                "min_pixels": 200,
                "max_pixels": 2000,
                "prettify_markdown": True,
                "show_formula_number": True,
                "visualize": False,
                "additional_params": {},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("AISTUDIO_ACCESS_TOKEN", "test-access-token")
        converter_dict = {
            "type": self.CLASS_TYPE,
            "init_parameters": {
                "api_url": "http://test-api-url.com",
                "access_token": {
                    "env_vars": ["AISTUDIO_ACCESS_TOKEN"],
                    "strict": True,
                    "type": "env_var",
                },
                "file_type": None,
                "use_doc_orientation_classify": None,
                "use_doc_unwarping": None,
                "use_layout_detection": None,
                "use_chart_recognition": None,
                "layout_threshold": None,
                "layout_nms": None,
                "layout_unclip_ratio": None,
                "layout_merge_bboxes_mode": None,
                "prompt_label": None,
                "format_block_content": None,
                "repetition_penalty": None,
                "temperature": None,
                "top_p": None,
                "min_pixels": None,
                "max_pixels": None,
                "prettify_markdown": None,
                "show_formula_number": None,
                "visualize": None,
                "additional_params": None,
            },
        }

        converter = PaddleOCRVLDocumentConverter.from_dict(converter_dict)

        assert converter.api_url == "http://test-api-url.com"
        assert converter.file_type is None
        assert converter.use_doc_orientation_classify is None
        assert converter.use_doc_unwarping is None
        assert converter.use_layout_detection is None
        assert converter.use_chart_recognition is None
        assert converter.layout_threshold is None
        assert converter.layout_nms is None
        assert converter.layout_unclip_ratio is None
        assert converter.layout_merge_bboxes_mode is None
        assert converter.prompt_label is None
        assert converter.format_block_content is None
        assert converter.repetition_penalty is None
        assert converter.temperature is None
        assert converter.top_p is None
        assert converter.min_pixels is None
        assert converter.max_pixels is None
        assert converter.prettify_markdown is None
        assert converter.show_formula_number is None
        assert converter.visualize is None
        assert converter.additional_params is None

    def test_from_dict_with_custom_parameters(self, monkeypatch):
        monkeypatch.setenv("AISTUDIO_ACCESS_TOKEN", "test-access-token")
        converter_dict = {
            "type": self.CLASS_TYPE,
            "init_parameters": {
                "api_url": "http://custom-api-url.com",
                "access_token": {
                    "env_vars": ["AISTUDIO_ACCESS_TOKEN"],
                    "strict": True,
                    "type": "env_var",
                },
                "file_type": "pdf",
                "use_doc_orientation_classify": True,
                "use_doc_unwarping": False,
                "use_layout_detection": True,
                "use_chart_recognition": False,
                "layout_threshold": 0.6,
                "layout_nms": True,
                "layout_unclip_ratio": 1.8,
                "layout_merge_bboxes_mode": "merge",
                "prompt_label": "table",
                "format_block_content": True,
                "repetition_penalty": 1.1,
                "temperature": 0.9,
                "top_p": 0.8,
                "min_pixels": 150,
                "max_pixels": 1500,
                "prettify_markdown": False,
                "show_formula_number": True,
                "visualize": True,
                "additional_params": {},
            },
        }

        converter = PaddleOCRVLDocumentConverter.from_dict(converter_dict)

        assert converter.api_url == "http://custom-api-url.com"
        assert converter.file_type == 0
        assert converter.use_doc_orientation_classify is True
        assert converter.use_doc_unwarping is False
        assert converter.use_layout_detection is True
        assert converter.use_chart_recognition is False
        assert converter.layout_threshold == 0.6
        assert converter.layout_nms is True
        assert converter.layout_unclip_ratio == 1.8
        assert converter.layout_merge_bboxes_mode == "merge"
        assert converter.prompt_label == "table"
        assert converter.format_block_content is True
        assert converter.repetition_penalty == 1.1
        assert converter.temperature == 0.9
        assert converter.top_p == 0.8
        assert converter.min_pixels == 150
        assert converter.max_pixels == 1500
        assert converter.prettify_markdown is False
        assert converter.show_formula_number is True
        assert converter.visualize is True
        assert converter.additional_params == {}

    @pytest.fixture
    def mock_ocr_response(self):
        """Create a mock PaddleOCR response"""
        mock_response = {
            "logId": "123",
            "errorCode": "0",
            "errorMsg": "Success",
            "result": {
                "layoutParsingResults": [
                    {
                        "markdown": {"text": "# Sample Document\n\nThis is page 1."},
                        "prunedResult": {},
                    }
                ],
                "dataInfo": {
                    "width": 1024,
                    "height": 1024,
                    "type": "image",
                },
            },
        }
        return mock_response

    @pytest.fixture
    def mock_ocr_response_with_multiple_pages(self):
        """Create a mock PaddleOCR response with multiple pages"""
        mock_response = {
            "logId": "123",
            "errorCode": "0",
            "errorMsg": "Success",
            "result": {
                "layoutParsingResults": [
                    {
                        "markdown": {"text": "# Page 1"},
                        "prunedResult": {},
                    },
                    {
                        "markdown": {"text": "# Page 2"},
                        "prunedResult": {},
                    },
                ],
                "dataInfo": {
                    "numPages": 2,
                    "pages": [
                        {"width": 512, "height": 512},
                        {"width": 512, "height": 512},
                    ],
                    "type": "pdf",
                },
            },
        }
        return mock_response

    @pytest.mark.parametrize(
        "source_type",
        ["file_path_str", "path_object", "bytestream"],
    )
    def test_run_with_local_sources(self, mock_ocr_response, tmp_path, source_type):
        """Test processing with local source types (str, Path, ByteStream)"""
        converter = PaddleOCRVLDocumentConverter(
            api_url="http://test-api-url.com", access_token=Secret.from_token("test-access-token")
        )

        # Create temporary file
        test_file = create_empty_image(tmp_path, "test.png")

        # Create the source based on type
        if source_type == "file_path_str":
            source = str(test_file)
        elif source_type == "path_object":
            source = test_file
        else:  # bytestream
            source = ByteStream(data=test_file.read_bytes(), meta={"file_path": str(test_file)})

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ocr_response
            mock_post.return_value = mock_response

            result = converter.run(sources=[source])

            assert len(result["documents"]) == 1
            assert isinstance(result["documents"][0], Document)
            assert result["documents"][0].content == "# Sample Document\n\nThis is page 1."
            assert len(result["raw_paddleocr_responses"]) == 1
            assert result["raw_paddleocr_responses"][0] == mock_ocr_response

    def test_run_with_multiple_sources(self, mock_ocr_response, tmp_path):
        """Test processing with multiple source types"""
        converter = PaddleOCRVLDocumentConverter(
            api_url="http://test-api-url.com", access_token=Secret.from_token("test-access-token")
        )

        # Create temporary files
        test_file1 = create_empty_image(tmp_path, "test1.png")
        test_file2 = create_empty_image(tmp_path, "test2.png")

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ocr_response
            mock_post.return_value = mock_response

            sources = [
                str(test_file1),
                test_file2,
                ByteStream(data=test_file1.read_bytes(), meta={"file_path": str(test_file1)}),
            ]
            result = converter.run(sources=sources)

            assert len(result["documents"]) == 3
            assert all(isinstance(doc, Document) for doc in result["documents"])
            assert len(result["raw_paddleocr_responses"]) == 3

    def test_run_handles_api_error(self, mock_ocr_response, tmp_path):
        """Test error handling when API fails"""
        converter = PaddleOCRVLDocumentConverter(
            api_url="http://test-api-url.com", access_token=Secret.from_token("test-access-token")
        )

        test_file1 = create_empty_image(tmp_path, "test1.png")
        test_file2 = create_empty_image(tmp_path, "test2.png")
        test_file3 = create_empty_image(tmp_path, "test3.png")

        with patch("requests.post") as mock_post:
            error_response = MagicMock()
            error_response.status_code = 404
            error_response.raise_for_status.side_effect = requests.HTTPError("404 Client Error")

            # First call succeeds, second fails, third succeeds
            mock_post.side_effect = [
                MagicMock(status_code=200, json=lambda: mock_ocr_response),
                error_response,
                MagicMock(status_code=200, json=lambda: mock_ocr_response),
            ]

            sources = [str(test_file1), str(test_file2), str(test_file3)]
            result = converter.run(sources=sources)

            # Should only return 2 documents (failed source skipped)
            assert len(result["documents"]) == 2
            assert len(result["raw_paddleocr_responses"]) == 2

    def test_run_with_meta_single_dict(self, mock_ocr_response, tmp_path):
        """Test that meta parameter with single dict is applied to all documents"""
        converter = PaddleOCRVLDocumentConverter(
            api_url="http://test-api-url.com", access_token=Secret.from_token("test-access-token")
        )

        test_file1 = create_empty_image(tmp_path, "test1.png")
        test_file2 = create_empty_image(tmp_path, "test2.png")

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ocr_response
            mock_post.return_value = mock_response

            sources = [str(test_file1), str(test_file2)]
            result = converter.run(sources=sources, meta={"department": "engineering", "year": 2024})

            assert len(result["documents"]) == 2
            # Both documents should have the same metadata
            for doc in result["documents"]:
                assert doc.meta["department"] == "engineering"
                assert doc.meta["year"] == 2024
                # File path metadata should still be present
                assert "file_path" in doc.meta

    def test_run_with_meta_list_of_dicts(self, mock_ocr_response, tmp_path):
        """Test that meta parameter with list of dicts applies each dict to corresponding document"""
        converter = PaddleOCRVLDocumentConverter(
            api_url="http://test-api-url.com", access_token=Secret.from_token("test-access-token")
        )

        test_file1 = create_empty_image(tmp_path, "test1.png")
        test_file2 = create_empty_image(tmp_path, "test2.png")

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ocr_response
            mock_post.return_value = mock_response

            sources = [str(test_file1), str(test_file2)]
            result = converter.run(
                sources=sources,
                meta=[
                    {"author": "Alice", "category": "report"},
                    {"author": "Bob", "category": "invoice"},
                ],
            )

            assert len(result["documents"]) == 2
            # First document
            assert result["documents"][0].meta["author"] == "Alice"
            assert result["documents"][0].meta["category"] == "report"
            # Second document
            assert result["documents"][1].meta["author"] == "Bob"
            assert result["documents"][1].meta["category"] == "invoice"
            # File path metadata should still be present in both
            assert "file_path" in result["documents"][0].meta
            assert "file_path" in result["documents"][1].meta

    def test_run_with_meta_none(self, mock_ocr_response, tmp_path):
        """Test that meta parameter with `None` works correctly"""
        converter = PaddleOCRVLDocumentConverter(
            api_url="http://test-api-url.com", access_token=Secret.from_token("test-access-token")
        )

        test_file = create_empty_image(tmp_path, "test.png")

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ocr_response
            mock_post.return_value = mock_response

            sources = [str(test_file)]
            result = converter.run(sources=sources, meta=None)

            assert len(result["documents"]) == 1
            # Only file path metadata should be present
            assert "file_path" in result["documents"][0].meta

    def test_file_type_auto_detection_pdf(self, mock_ocr_response_with_multiple_pages, tmp_path):
        """Test that file_type is automatically detected as PDF from .pdf extension"""
        converter = PaddleOCRVLDocumentConverter(
            api_url="http://test-api-url.com", access_token=Secret.from_token("test-access-token")
        )

        # Create a PDF file
        pdf_file = create_empty_pdf(tmp_path, "test.pdf")

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ocr_response_with_multiple_pages
            mock_post.return_value = mock_response

            result = converter.run(sources=[str(pdf_file)])

            assert len(result["documents"]) == 1
            # Verify that the correct file type (0 for PDF) was used in the API call
            call_args = mock_post.call_args[1]["json"]
            assert call_args["fileType"] == 0  # Should be 0 for PDF

    def test_file_type_auto_detection_image(self, mock_ocr_response, tmp_path):
        """Test that file_type is automatically detected as image from .png extension"""
        converter = PaddleOCRVLDocumentConverter(
            api_url="http://test-api-url.com", access_token=Secret.from_token("test-access-token")
        )

        # Create an image file
        image_file = create_empty_image(tmp_path, "test.png")

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ocr_response
            mock_post.return_value = mock_response

            result = converter.run(sources=[str(image_file)])

            assert len(result["documents"]) == 1
            # Verify that the correct file type (1 for image) was used in the API call
            call_args = mock_post.call_args[1]["json"]
            assert call_args["fileType"] == 1  # Should be 1 for image

    def test_file_type_manual_specification_pdf(self, mock_ocr_response_with_multiple_pages, tmp_path):
        """Test that manually specified file_type overrides auto-detection"""
        converter = PaddleOCRVLDocumentConverter(
            api_url="http://test-api-url.com",
            access_token=Secret.from_token("test-access-token"),
            file_type="pdf",  # Manually specify as PDF
        )

        # Create an image file (which would normally auto-detect as image)
        image_file = create_empty_image(tmp_path, "test.png")

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ocr_response_with_multiple_pages
            mock_post.return_value = mock_response

            result = converter.run(sources=[str(image_file)])

            assert len(result["documents"]) == 1
            # Verify that the manually specified file type (0 for PDF) was used
            call_args = mock_post.call_args[1]["json"]
            assert call_args["fileType"] == 0  # Should be 0 (PDF) despite being a PNG file

    def test_file_type_manual_specification_image(self, mock_ocr_response, tmp_path):
        """Test that manually specified file_type overrides auto-detection"""
        converter = PaddleOCRVLDocumentConverter(
            api_url="http://test-api-url.com",
            access_token=Secret.from_token("test-access-token"),
            file_type="image",  # Manually specify as image
        )

        # Create a PDF file (which would normally auto-detect as PDF)
        pdf_file = create_empty_pdf(tmp_path, "test.pdf")

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ocr_response
            mock_post.return_value = mock_response

            result = converter.run(sources=[str(pdf_file)])

            assert len(result["documents"]) == 1
            # Verify that the manually specified file type (1 for image) was used
            call_args = mock_post.call_args[1]["json"]
            assert call_args["fileType"] == 1  # Should be 1 (image) despite being a PDF file

    def test_file_type_auto_detection_unknown_extension(self, mock_ocr_response, tmp_path):
        """Test that unknown file extensions result in skipping the file"""
        converter = PaddleOCRVLDocumentConverter(
            api_url="http://test-api-url.com", access_token=Secret.from_token("test-access-token")
        )

        # Create a file with unknown extension
        unknown_file = tmp_path / "test.unknown"
        unknown_file.write_bytes(b"dummy data")

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ocr_response
            mock_post.return_value = mock_response

            result = converter.run(sources=[str(unknown_file)])

            # Should skip the file due to unknown extension
            assert len(result["documents"]) == 0
            assert len(result["raw_paddleocr_responses"]) == 0
            # requests.post should not be called at all
            mock_post.assert_not_called()

    def test_file_type_auto_detection_bytestream(self, mock_ocr_response, tmp_path):
        """Test file_type auto-detection with ByteStream input"""
        converter = PaddleOCRVLDocumentConverter(
            api_url="http://test-api-url.com", access_token=Secret.from_token("test-access-token")
        )

        # Create files
        pdf_file = create_empty_pdf(tmp_path, "test.pdf")
        image_file = create_empty_image(tmp_path, "test.png")

        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_ocr_response
            mock_post.return_value = mock_response

            # Test PDF ByteStream
            pdf_bytestream = ByteStream(data=pdf_file.read_bytes(), meta={"file_path": str(pdf_file)})
            result_pdf = converter.run(sources=[pdf_bytestream])

            # Test image ByteStream
            image_bytestream = ByteStream(data=image_file.read_bytes(), meta={"file_path": str(image_file)})
            result_image = converter.run(sources=[image_bytestream])

            # Both should succeed
            assert len(result_pdf["documents"]) == 1
            assert len(result_image["documents"]) == 1

            # Verify API calls used correct file types
            assert mock_post.call_count == 2
            calls = mock_post.call_args_list
            # First call should be for PDF (fileType=0)
            assert calls[0][1]["json"]["fileType"] == 0
            # Second call should be for image (fileType=1)
            assert calls[1][1]["json"]["fileType"] == 1

    @pytest.mark.skipif(
        not os.environ.get("PADDLEOCR_VL_API_URL") or not os.environ.get("AISTUDIO_ACCESS_TOKEN"),
        reason="Export env vars `PADDLEOCR_VL_API_URL` and `AISTUDIO_ACCESS_TOKEN` to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "source_files,expected_docs",
        [
            (
                [
                    "sample_pdf.pdf",
                ],
                1,
            ),
            (
                [
                    "sample_img.jpg",
                ],
                1,
            ),
            (
                [
                    "sample_pdf.pdf",
                    "sample_img.jpg",
                ],
                2,
            ),
        ],
        ids=["pdf_only", "image_only", "mixed_pdf_image"],
    )
    def test_integration_run_with_files(self, test_files_path, source_files, expected_docs):
        """Integration test with real API call using various file types"""

        source_files = [test_files_path / file for file in source_files]

        converter = PaddleOCRVLDocumentConverter(api_url=os.environ["PADDLEOCR_VL_API_URL"])

        result = converter.run(sources=source_files)

        assert len(result["documents"]) == expected_docs
        assert all(isinstance(doc, Document) for doc in result["documents"])
        assert all(len(doc.content) > 0 for doc in result["documents"])
        assert "raw_paddleocr_responses" in result
        assert len(result["raw_paddleocr_responses"]) == expected_docs
