# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.dataclasses import ByteStream
from haystack.utils import Secret
from mistralai.models import DocumentURLChunk, FileChunk, ImageURLChunk
from pydantic import BaseModel, Field

from haystack_integrations.components.converters.mistral import (
    MistralOCRDocumentConverter,
)


class TestMistralOCRDocumentConverter:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        converter = MistralOCRDocumentConverter()

        assert converter.api_key == Secret.from_env_var("MISTRAL_API_KEY")
        assert converter.model == "mistral-ocr-2505"
        assert converter.include_image_base64 is False
        assert converter.pages is None
        assert converter.image_limit is None
        assert converter.image_min_size is None

    def test_init_with_parameters(self):
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key"),
            model="mistral-ocr-custom",
            include_image_base64=True,
        )

        assert converter.api_key == Secret.from_token("test-api-key")
        assert converter.model == "mistral-ocr-custom"
        assert converter.include_image_base64 is True
        assert converter.pages is None
        assert converter.image_limit is None
        assert converter.image_min_size is None

    def test_init_with_all_optional_parameters(self):
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key"),
            model="mistral-ocr-custom",
            include_image_base64=True,
            pages=[0, 1, 2],
            image_limit=10,
            image_min_size=100,
        )

        assert converter.api_key == Secret.from_token("test-api-key")
        assert converter.model == "mistral-ocr-custom"
        assert converter.include_image_base64 is True
        assert converter.pages == [0, 1, 2]
        assert converter.image_limit == 10
        assert converter.image_min_size == 100

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        converter = MistralOCRDocumentConverter()
        converter_dict = converter.to_dict()

        assert converter_dict == {
            "type": (
                "haystack_integrations.components.converters.mistral."
                "ocr_document_converter.MistralOCRDocumentConverter"
            ),
            "init_parameters": {
                "api_key": {
                    "env_vars": ["MISTRAL_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "model": "mistral-ocr-2505",
                "include_image_base64": False,
                "pages": None,
                "image_limit": None,
                "image_min_size": None,
                "cleanup_uploaded_files": True,
            },
        }

    def test_to_dict_with_custom_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="mistral-ocr-custom",
            include_image_base64=True,
            pages=[0, 1, 2],
            image_limit=10,
            image_min_size=100,
            cleanup_uploaded_files=False,
        )
        converter_dict = converter.to_dict()

        assert converter_dict == {
            "type": (
                "haystack_integrations.components.converters.mistral."
                "ocr_document_converter.MistralOCRDocumentConverter"
            ),
            "init_parameters": {
                "api_key": {
                    "type": "env_var",
                    "env_vars": ["ENV_VAR"],
                    "strict": False,
                },
                "model": "mistral-ocr-custom",
                "include_image_base64": True,
                "pages": [0, 1, 2],
                "image_limit": 10,
                "image_min_size": 100,
                "cleanup_uploaded_files": False,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        converter_dict = {
            "type": (
                "haystack_integrations.components.converters.mistral."
                "ocr_document_converter.MistralOCRDocumentConverter"
            ),
            "init_parameters": {
                "api_key": {
                    "env_vars": ["MISTRAL_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "model": "mistral-ocr-2505",
                "include_image_base64": False,
                "pages": None,
                "image_limit": None,
                "image_min_size": None,
                "cleanup_uploaded_files": True,
            },
        }

        converter = MistralOCRDocumentConverter.from_dict(converter_dict)

        assert converter.model == "mistral-ocr-2505"
        assert converter.include_image_base64 is False
        assert converter.pages is None
        assert converter.image_limit is None
        assert converter.image_min_size is None
        assert converter.cleanup_uploaded_files is True

    def test_from_dict_with_custom_parameters(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        converter_dict = {
            "type": (
                "haystack_integrations.components.converters.mistral."
                "ocr_document_converter.MistralOCRDocumentConverter"
            ),
            "init_parameters": {
                "api_key": {
                    "env_vars": ["MISTRAL_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "model": "mistral-ocr-custom",
                "include_image_base64": True,
                "pages": [0, 1, 2],
                "image_limit": 10,
                "image_min_size": 100,
                "cleanup_uploaded_files": False,
            },
        }

        converter = MistralOCRDocumentConverter.from_dict(converter_dict)

        assert converter.model == "mistral-ocr-custom"
        assert converter.include_image_base64 is True
        assert converter.pages == [0, 1, 2]
        assert converter.image_limit == 10
        assert converter.image_min_size == 100
        assert converter.cleanup_uploaded_files is False

    @pytest.fixture
    def mock_ocr_response(self):
        """Create a mock OCR response"""
        mock_page = MagicMock()
        mock_page.markdown = "# Sample Document\n\nThis is page 1."
        mock_page.images = []

        mock_response = MagicMock()
        mock_response.pages = [mock_page]
        mock_response.document_annotation = None
        mock_response.model_dump.return_value = {
            "pages": [
                {"markdown": "# Sample Document\n\nThis is page 1.", "images": []}
            ],
            "document_annotation": None,
        }
        return mock_response

    @pytest.fixture
    def mock_ocr_response_with_multiple_pages(self):
        """Create a mock OCR response with multiple pages"""
        mock_page1 = MagicMock()
        mock_page1.markdown = "# Page 1"
        mock_page1.images = []

        mock_page2 = MagicMock()
        mock_page2.markdown = "# Page 2"
        mock_page2.images = []

        mock_response = MagicMock()
        mock_response.pages = [mock_page1, mock_page2]
        mock_response.document_annotation = None
        mock_response.model_dump.return_value = {
            "pages": [
                {"markdown": "# Page 1", "images": []},
                {"markdown": "# Page 2", "images": []},
            ],
            "document_annotation": None,
        }
        return mock_response

    def test_run_with_document_url_chunk(self, mock_ocr_response):
        """Test processing with DocumentURLChunk"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key")
        )

        with patch.object(
            converter.client.ocr, "process", return_value=mock_ocr_response
        ):
            sources = [DocumentURLChunk(document_url="https://example.com/doc.pdf")]
            result = converter.run(sources=sources)

            assert len(result["documents"]) == 1
            assert isinstance(result["documents"][0], Document)
            assert (
                result["documents"][0].content == "# Sample Document\n\nThis is page 1."
            )
            assert result["documents"][0].meta["source_page_count"] == 1
            assert result["documents"][0].meta["source_total_images"] == 0

    def test_run_with_file_chunk(self, mock_ocr_response):
        """Test processing with FileChunk"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key")
        )

        with patch.object(
            converter.client.ocr, "process", return_value=mock_ocr_response
        ):
            sources = [FileChunk(file_id="file-123")]
            result = converter.run(sources=sources)

            assert len(result["documents"]) == 1
            assert isinstance(result["documents"][0], Document)
            assert (
                result["documents"][0].content == "# Sample Document\n\nThis is page 1."
            )

    def test_run_with_image_url_chunk(self, mock_ocr_response):
        """Test processing with ImageURLChunk"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key")
        )

        with patch.object(
            converter.client.ocr, "process", return_value=mock_ocr_response
        ):
            sources = [ImageURLChunk(image_url="https://example.com/image.jpg")]
            result = converter.run(sources=sources)

            assert len(result["documents"]) == 1
            assert isinstance(result["documents"][0], Document)

    def test_run_with_file_path(self, mock_ocr_response, tmp_path):
        """Test processing with string file path"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key")
        )

        # Create a temporary file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.id = "uploaded-file-123"

        with patch.object(
            converter.client.files, "upload", return_value=mock_uploaded_file
        ):
            with patch.object(
                converter.client.ocr, "process", return_value=mock_ocr_response
            ):
                with patch.object(converter.client.files, "delete"):
                    sources = [str(test_file)]
                    result = converter.run(sources=sources)

                    assert len(result["documents"]) == 1
                    assert isinstance(result["documents"][0], Document)
                    converter.client.files.upload.assert_called_once()

    def test_run_with_path_object(self, mock_ocr_response, tmp_path):
        """Test processing with Path object"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key")
        )

        # Create a temporary file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.id = "uploaded-file-123"

        with patch.object(
            converter.client.files, "upload", return_value=mock_uploaded_file
        ):
            with patch.object(
                converter.client.ocr, "process", return_value=mock_ocr_response
            ):
                with patch.object(converter.client.files, "delete"):
                    sources = [test_file]
                    result = converter.run(sources=sources)

                    assert len(result["documents"]) == 1
                    assert isinstance(result["documents"][0], Document)

    def test_run_with_bytestream(self, mock_ocr_response):
        """Test processing with ByteStream"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key")
        )

        bytestream = ByteStream(
            data=b"fake pdf content", meta={"file_path": "test.pdf"}
        )

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.id = "uploaded-file-123"

        with patch.object(
            converter.client.files, "upload", return_value=mock_uploaded_file
        ):
            with patch.object(
                converter.client.ocr, "process", return_value=mock_ocr_response
            ):
                with patch.object(converter.client.files, "delete"):
                    sources = [bytestream]
                    result = converter.run(sources=sources)

                    assert len(result["documents"]) == 1
                    assert isinstance(result["documents"][0], Document)

    def test_run_with_multiple_sources(self, mock_ocr_response, tmp_path):
        """Test processing with multiple mixed source types"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key")
        )

        # Create a temporary file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.id = "uploaded-file-123"

        with patch.object(
            converter.client.files, "upload", return_value=mock_uploaded_file
        ):
            with patch.object(
                converter.client.ocr, "process", return_value=mock_ocr_response
            ):
                with patch.object(converter.client.files, "delete"):
                    sources = [
                        DocumentURLChunk(document_url="https://example.com/doc.pdf"),
                        FileChunk(file_id="file-123"),
                        str(test_file),
                    ]
                    result = converter.run(sources=sources)

                    assert len(result["documents"]) == 3
                    assert all(isinstance(doc, Document) for doc in result["documents"])

    def test_run_with_bbox_annotations(self):
        """Test processing with bbox annotation schema"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key")
        )

        # Define annotation schema
        class ImageAnnotation(BaseModel):
            image_type: str = Field(..., description="Type of image")

        # Create mock response with image annotation
        mock_image = MagicMock()
        mock_image.id = "img-1"
        mock_image.image_annotation = '{"image_type": "diagram"}'

        mock_page = MagicMock()
        mock_page.markdown = "# Document\n\n![img-1](img-1)"
        mock_page.images = [mock_image]

        mock_response = MagicMock()
        mock_response.pages = [mock_page]
        mock_response.document_annotation = None
        mock_response.model_dump.return_value = {
            "pages": [],
            "document_annotation": None,
        }

        with patch.object(converter.client.ocr, "process", return_value=mock_response):
            sources = [DocumentURLChunk(document_url="https://example.com/doc.pdf")]
            result = converter.run(
                sources=sources, bbox_annotation_schema=ImageAnnotation
            )

            assert len(result["documents"]) == 1
            # Check that image annotation was enriched in content
            assert "Image Annotation:" in result["documents"][0].content

    def test_run_with_document_annotations(self):
        """Test processing with document annotation schema"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key")
        )

        # Define annotation schema
        class DocumentAnnotation(BaseModel):
            language: str = Field(..., description="Document language")
            topics: List[str] = Field(..., description="Main topics")

        # Create mock response with document annotation
        mock_page = MagicMock()
        mock_page.markdown = "# Document"
        mock_page.images = []

        mock_response = MagicMock()
        mock_response.pages = [mock_page]
        mock_response.document_annotation = '{"language": "en", "topics": ["AI", "ML"]}'
        mock_response.model_dump.return_value = {
            "pages": [],
            "document_annotation": '{"language": "en", "topics": ["AI", "ML"]}',
        }

        with patch.object(converter.client.ocr, "process", return_value=mock_response):
            sources = [DocumentURLChunk(document_url="https://example.com/doc.pdf")]
            result = converter.run(
                sources=sources, document_annotation_schema=DocumentAnnotation
            )

            assert len(result["documents"]) == 1
            # Check that document annotations are in metadata
            assert result["documents"][0].meta["source_language"] == "en"
            assert result["documents"][0].meta["source_topics"] == ["AI", "ML"]

    def test_run_with_both_annotations(self):
        """Test processing with both bbox and document annotation schemas"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key")
        )

        class ImageAnnotation(BaseModel):
            image_type: str = Field(..., description="Type of image")

        class DocumentAnnotation(BaseModel):
            language: str = Field(..., description="Document language")

        # Create mock response
        mock_image = MagicMock()
        mock_image.id = "img-1"
        mock_image.image_annotation = '{"image_type": "chart"}'

        mock_page = MagicMock()
        mock_page.markdown = "![img-1](img-1)"
        mock_page.images = [mock_image]

        mock_response = MagicMock()
        mock_response.pages = [mock_page]
        mock_response.document_annotation = '{"language": "en"}'
        mock_response.model_dump.return_value = {
            "pages": [],
            "document_annotation": '{"language": "en"}',
        }

        with patch.object(converter.client.ocr, "process", return_value=mock_response):
            sources = [DocumentURLChunk(document_url="https://example.com/doc.pdf")]
            result = converter.run(
                sources=sources,
                bbox_annotation_schema=ImageAnnotation,
                document_annotation_schema=DocumentAnnotation,
            )

            assert len(result["documents"]) == 1
            assert "Image Annotation:" in result["documents"][0].content
            assert result["documents"][0].meta["source_language"] == "en"

    def test_run_with_pages_parameter(self, mock_ocr_response):
        """Test that pages parameter is passed to API"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key"), pages=[0, 1]
        )

        with patch.object(
            converter.client.ocr, "process", return_value=mock_ocr_response
        ) as mock_process:
            sources = [DocumentURLChunk(document_url="https://example.com/doc.pdf")]
            result = converter.run(sources=sources)

            # Verify pages parameter was passed
            call_args = mock_process.call_args
            assert call_args.kwargs["pages"] == [0, 1]
            assert len(result["documents"]) == 1

    def test_run_handles_api_error(self, mock_ocr_response):
        """Test error handling when API fails"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key")
        )

        with patch.object(converter.client.ocr, "process") as mock_process:
            # First call succeeds, second fails, third succeeds
            mock_process.side_effect = [
                mock_ocr_response,
                Exception("API Error"),
                mock_ocr_response,
            ]

            sources = [
                DocumentURLChunk(document_url="https://example.com/doc1.pdf"),
                DocumentURLChunk(document_url="https://example.com/doc2.pdf"),
                DocumentURLChunk(document_url="https://example.com/doc3.pdf"),
            ]
            result = converter.run(sources=sources)

            # Should only return 2 documents (failed source skipped)
            assert len(result["documents"]) == 2
            assert len(result["raw_mistral_response"]) == 2

    def test_process_ocr_response_basic(self, mock_ocr_response):
        """Test basic OCR response processing"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key")
        )

        # Test the internal _process_ocr_response method
        document = converter._process_ocr_response(
            mock_ocr_response, document_annotation_schema=None
        )

        assert isinstance(document, Document)
        assert document.content == "# Sample Document\n\nThis is page 1."
        assert document.meta["source_page_count"] == 1
        assert document.meta["source_total_images"] == 0

    def test_process_ocr_response_multiple_pages(
        self, mock_ocr_response_with_multiple_pages
    ):
        """Test multi-page document with form feed separator"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key")
        )

        document = converter._process_ocr_response(
            mock_ocr_response_with_multiple_pages, document_annotation_schema=None
        )

        assert isinstance(document, Document)
        # Pages should be separated by \f
        assert document.content == "# Page 1\f# Page 2"
        assert "\f" in document.content
        assert document.meta["source_page_count"] == 2

    def test_process_ocr_response_with_images(self):
        """Test metadata extraction with images"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key")
        )

        # Create mock response with images
        mock_image1 = MagicMock()
        mock_image1.id = "img-1"
        mock_image1.image_annotation = None

        mock_image2 = MagicMock()
        mock_image2.id = "img-2"
        mock_image2.image_annotation = None

        mock_page = MagicMock()
        mock_page.markdown = "# Document with images"
        mock_page.images = [mock_image1, mock_image2]

        mock_response = MagicMock()
        mock_response.pages = [mock_page]
        mock_response.document_annotation = None

        document = converter._process_ocr_response(
            mock_response, document_annotation_schema=None
        )

        assert document.meta["source_page_count"] == 1
        assert document.meta["source_total_images"] == 2

    def test_run_with_file_cleanup(self, mock_ocr_response, tmp_path):
        """Test that uploaded files are deleted after successful processing"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key"), cleanup_uploaded_files=True
        )

        # Create a temporary file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.id = "uploaded-file-123"

        with patch.object(
            converter.client.files, "upload", return_value=mock_uploaded_file
        ):
            with patch.object(
                converter.client.ocr, "process", return_value=mock_ocr_response
            ):
                with patch.object(converter.client.files, "delete") as mock_delete:
                    sources = [str(test_file)]
                    result = converter.run(sources=sources)

                    # Verify file was uploaded and deleted
                    assert len(result["documents"]) == 1
                    converter.client.files.upload.assert_called_once()
                    mock_delete.assert_called_once_with(file_id="uploaded-file-123")

    def test_run_user_provided_filechunk_not_deleted(self, mock_ocr_response):
        """Test that user-provided FileChunk is not deleted"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key"), cleanup_uploaded_files=True
        )

        with patch.object(
            converter.client.ocr, "process", return_value=mock_ocr_response
        ):
            with patch.object(converter.client.files, "delete") as mock_delete:
                # User provides FileChunk directly
                sources = [FileChunk(file_id="user-file-123")]
                result = converter.run(sources=sources)

                # Verify file was NOT deleted
                assert len(result["documents"]) == 1
                mock_delete.assert_not_called()

    def test_run_with_cleanup_disabled(self, mock_ocr_response, tmp_path):
        """Test that files are not deleted when cleanup_uploaded_files=False"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key"), cleanup_uploaded_files=False
        )

        # Create a temporary file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.id = "uploaded-file-123"

        with patch.object(
            converter.client.files, "upload", return_value=mock_uploaded_file
        ):
            with patch.object(
                converter.client.ocr, "process", return_value=mock_ocr_response
            ):
                with patch.object(converter.client.files, "delete") as mock_delete:
                    sources = [str(test_file)]
                    result = converter.run(sources=sources)

                    # Verify file was uploaded but NOT deleted
                    assert len(result["documents"]) == 1
                    converter.client.files.upload.assert_called_once()
                    mock_delete.assert_not_called()

    def test_run_cleanup_happens_on_ocr_failure(self, tmp_path):
        """Test that cleanup happens even when OCR processing fails"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key"), cleanup_uploaded_files=True
        )

        # Create a temporary file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.id = "uploaded-file-123"

        with patch.object(
            converter.client.files, "upload", return_value=mock_uploaded_file
        ):
            with patch.object(
                converter.client.ocr, "process", side_effect=Exception("OCR failed")
            ):
                with patch.object(converter.client.files, "delete") as mock_delete:
                    sources = [str(test_file)]
                    result = converter.run(sources=sources)

                    # Verify no documents returned due to failure
                    assert len(result["documents"]) == 0
                    # But file should still be deleted
                    mock_delete.assert_called_once_with(file_id="uploaded-file-123")

    def test_run_cleanup_failure_does_not_break_flow(self, mock_ocr_response, tmp_path):
        """Test that cleanup failures don't break the main flow"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key"), cleanup_uploaded_files=True
        )

        # Create a temporary file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.id = "uploaded-file-123"

        with patch.object(
            converter.client.files, "upload", return_value=mock_uploaded_file
        ):
            with patch.object(
                converter.client.ocr, "process", return_value=mock_ocr_response
            ):
                with patch.object(
                    converter.client.files,
                    "delete",
                    side_effect=Exception("Delete failed"),
                ):
                    sources = [str(test_file)]
                    # Should not raise an exception
                    result = converter.run(sources=sources)

                    # Verify document was still processed successfully
                    assert len(result["documents"]) == 1
                    assert isinstance(result["documents"][0], Document)

    def test_run_mixed_sources_only_uploaded_files_deleted(
        self, mock_ocr_response, tmp_path
    ):
        """Test that only uploaded files are deleted, not user-provided chunks"""
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key"), cleanup_uploaded_files=True
        )

        # Create a temporary file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        mock_uploaded_file = MagicMock()
        mock_uploaded_file.id = "uploaded-file-123"

        with patch.object(
            converter.client.files, "upload", return_value=mock_uploaded_file
        ):
            with patch.object(
                converter.client.ocr, "process", return_value=mock_ocr_response
            ):
                with patch.object(converter.client.files, "delete") as mock_delete:
                    sources = [
                        str(test_file),  # This will be uploaded
                        FileChunk(file_id="user-file-123"),  # User-provided
                        DocumentURLChunk(
                            document_url="https://example.com/doc.pdf"
                        ),  # URL
                    ]
                    result = converter.run(sources=sources)

                    # Verify all sources processed
                    assert len(result["documents"]) == 3
                    # Only the uploaded file should be deleted
                    mock_delete.assert_called_once_with(file_id="uploaded-file-123")

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY"),
        reason="Export an env var called MISTRAL_API_KEY containing the Mistral API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_run_with_document_url(self):
        """Integration test with real API call using arxiv PDF"""
        converter = MistralOCRDocumentConverter()

        sources = [DocumentURLChunk(document_url="https://arxiv.org/pdf/1706.03762")]
        result = converter.run(sources=sources)

        assert len(result["documents"]) == 1
        assert isinstance(result["documents"][0], Document)
        assert len(result["documents"][0].content) > 0
        assert result["documents"][0].meta["source_page_count"] > 0
        assert "raw_mistral_response" in result
        assert len(result["raw_mistral_response"]) == 1

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY"),
        reason="Export an env var called MISTRAL_API_KEY containing the Mistral API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_run_with_annotations(self):
        """Integration test with real API call using annotation schemas"""
        converter = MistralOCRDocumentConverter(
            pages=[0]
        )  # Only process first page for speed

        # Define simple annotation schemas
        class ImageAnnotation(BaseModel):
            image_type: str = Field(
                ...,
                description="The type of image content (e.g., diagram, chart, photo)",
            )

        class DocumentAnnotation(BaseModel):
            language: str = Field(
                ..., description="The primary language of the document"
            )

        sources = [DocumentURLChunk(document_url="https://arxiv.org/pdf/1706.03762")]
        result = converter.run(
            sources=sources,
            bbox_annotation_schema=ImageAnnotation,
            document_annotation_schema=DocumentAnnotation,
        )

        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert isinstance(doc, Document)
        assert len(doc.content) > 0
        # Check if document annotation was added to metadata
        assert "source_language" in doc.meta

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY"),
        reason="Export an env var called MISTRAL_API_KEY containing the Mistral API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_multiple_sources(self):
        """Integration test with real API call using multiple sources"""
        converter = MistralOCRDocumentConverter(pages=[0])  # Only first page for speed

        sources = [
            DocumentURLChunk(document_url="https://arxiv.org/pdf/1706.03762"),
            DocumentURLChunk(document_url="https://arxiv.org/pdf/2309.06180"),
        ]
        result = converter.run(sources=sources)

        assert len(result["documents"]) == 2
        assert all(isinstance(doc, Document) for doc in result["documents"])
        assert all(len(doc.content) > 0 for doc in result["documents"])
        assert len(result["raw_mistral_response"]) == 2
