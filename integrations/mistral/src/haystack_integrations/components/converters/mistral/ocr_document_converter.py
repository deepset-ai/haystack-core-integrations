"""Mistral OCR Document Converter for Haystack.

A custom Haystack component that uses Mistral's OCR API to extract text from documents.
Takes a signed/public URL to a document and returns extracted text as Haystack Documents.

API Reference:
https://docs.mistral.ai/capabilities/document_ai/basic_ocr/
https://docs.mistral.ai/capabilities/document_ai/annotations/

Usage Example:
    ```python
    from haystack.utils import Secret
    from haystack_integrations.mistral import MistralOCRDocumentConverter
    from mistralai.models import DocumentURLChunk, ImageURLChunk, FileChunk

    converter = MistralOCRDocumentConverter(
        api_key=Secret.from_env_var("MISTRAL_API_KEY"),
        model="mistral-ocr-2505"
    )

    # Option 1: Process a document URL
    doc_source = DocumentURLChunk(document_url="https://example.com/document.pdf")
    result = converter.run(source=doc_source)

    # Option 2: Process an image URL
    img_source = ImageURLChunk(image_url="https://example.com/receipt.jpg")
    result = converter.run(source=img_source)

    # Option 3: Process a Mistral file ID (See: https://docs.mistral.ai/api/#tag/files)
    file_source = FileChunk(file_id="file-abc123")
    result = converter.run(source=file_source)

    documents = result["documents"]
    raw_response = result["raw_mistral_response"]
    ```

Structured Output Example:
    ```python
    from pydantic import BaseModel, Field
    from haystack_integrations.mistral import MistralOCRDocumentConverter

    # Define schema for structured image annotations
    class ImageAnnotation(BaseModel):
        image_type: str = Field(..., description="The type of image content")
        short_description: str = Field(..., description="Short natural-language description")
        summary: str = Field(..., description="Detailed summary of the image content")

    # Define schema for structured document annotations
    class DocumentAnnotation(BaseModel):
        language: str = Field(..., description="Primary language of the document")
        chapter_titles: list[str] = Field(..., description="Detected chapter or section titles")
        urls: list[str] = Field(..., description="URLs found in the text")

    converter = MistralOCRDocumentConverter(
        api_key=Secret.from_env_var("MISTRAL_API_KEY"),
        model="mistral-ocr-2505",
        bbox_annotation_schema=ImageAnnotation,
        document_annotation_schema=DocumentAnnotation,
    )

    doc_source = DocumentURLChunk(document_url="https://example.com/report.pdf")
    result = converter.run(source=doc_source)
    ```
"""

import json
import re
from typing import Type

from haystack import Document, component
from haystack.utils import Secret
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from mistralai.models import (
    DocumentURLChunk,
    FileChunk,
    ImageURLChunk,
    OCRResponse,
)
from pydantic import BaseModel


@component
class MistralOCRDocumentConverter:
    """
    Extracts text from documents using Mistral's OCR API, with optional structured
    annotations for both individual image regions (bounding boxes) and full documents.

    Accepts a document URL (public or signed) and retrieves the recognized text
    via Mistral's OCR service. Returns a single Haystack Document containing all
    pages concatenated with form feed characters (\f), ensuring compatibility with e.g
    Haystac's DocumentSplitter for accurate page-wise splitting and overlap handling.
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
        model: str = "mistral-ocr-2505",
        include_image_base64: bool = False,
        pages: list[int] | None = None,
        image_limit: int | None = None,
        image_min_size: int | None = None,
        bbox_annotation_schema: Type[BaseModel] | None = None,
        document_annotation_schema: Type[BaseModel] | None = None,
    ):
        """
        Initialize the MistralOCRDocumentConverter.

        Args:
            api_key: Mistral API key (defaults to MISTRAL_API_KEY env var)
            model: OCR model to use (default: "mistral-ocr-2505")
            include_image_base64: Include base64 encoded images in response
                (may significantly increase response size and time)
            pages: Specific pages to process (0-indexed), Defaults to all pages
            image_limit: Maximum number of images to extract
            image_min_size: Minimum height and width of images to extract
            bbox_annotation_schema: Pydantic model for structured annotations per bounding box
            document_annotation_schema: Pydantic model for structured annotations for the full document
        """
        self.api_key = api_key
        self.model = model
        self.include_image_base64 = include_image_base64
        self.pages = pages
        self.image_limit = image_limit
        self.image_min_size = image_min_size

        # Keep schemas accessible for filtering downstream
        self.bbox_annotation_schema = bbox_annotation_schema
        self.document_annotation_schema = document_annotation_schema

        # Automatically convert provided Pydantic models into Mistral ResponseFormat schemas
        self.bbox_annotation_format = (
            response_format_from_pydantic_model(bbox_annotation_schema) if bbox_annotation_schema else None
        )
        self.document_annotation_format = (
            response_format_from_pydantic_model(document_annotation_schema) if document_annotation_schema else None
        )

        # Initialize Mistral client
        self.client = Mistral(api_key=self.api_key.resolve_value())

    @component.output_types(documents=list[Document], raw_mistral_response=OCRResponse)
    def run(self, source: DocumentURLChunk | FileChunk | ImageURLChunk) -> dict:
        """
        Extract text from a document using Mistral OCR.

        Args:
            source: Document source to process. Can be one of:
                - DocumentURLChunk: For document URLs (signed or public URLs to PDFs, etc.)
                - ImageURLChunk: For image URLs (signed or public URLs to images)
                - FileChunk: For Mistral file IDs (files previously uploaded to Mistral)

        Returns:
            Dictionary with two keys:
                - "documents": List containing a single Haystack Document
                  The Document contains:
                    - content: All pages joined with form feed (\f) separators in markdown format.
                      When using bbox_annotation in any format, image tags will be enriched by your defined descriptions
                    - meta: Aggregated metadata with structure:
                      {"source_page_count": int, "source_total_images": int, "source_*": any}
                      If document_annotation_format was provided, all annotation fields are unpacked
                      with 'source_' prefix (e.g., source_language, source_chapter_titles, source_urls)
                - "raw_mistral_response": Raw OCRResponse object from Mistral API
                  Contains complete response including per-page details, images, annotations, and usage info
        """
        # Call Mistral OCR API with the provided source
        ocr_response: OCRResponse = self.client.ocr.process(
            model=self.model,
            document=source,
            include_image_base64=self.include_image_base64,
            pages=self.pages,
            image_limit=self.image_limit,
            image_min_size=self.image_min_size,
            bbox_annotation_format=self.bbox_annotation_format,
            document_annotation_format=self.document_annotation_format,
        )

        # Convert OCR pages to a single Haystack Document
        # We add "\f" separators between pages to differentiate them and make them usable across other components
        page_contents = []
        total_images = 0

        for page in ocr_response.pages:
            # Enrich markdown content with structured image annotations inline
            enriched_content = page.markdown
            for img in page.images:
                if img.image_annotation:
                    # Regex pattern to find ![img-id](img-id) and insert annotation after it
                    pattern = f"!\\[{re.escape(img.id)}\\]\\({re.escape(img.id)}\\)"
                    replacement = f"![{img.id}]({img.id})\n\n**Image Annotation:** {img.image_annotation}\n"
                    enriched_content = re.sub(pattern, replacement, enriched_content)

            page_contents.append(enriched_content)
            total_images += len(page.images)

        # Join all pages with form feed character (\f) as separator
        all_content = "\f".join(page_contents)

        # Parse and filter document-level annotations to schema-defined fields
        try:
            parsed = json.loads(ocr_response.document_annotation or "{}")
            if self.document_annotation_schema:
                allowed = self.document_annotation_schema.model_fields.keys()
                parsed = {k: v for k, v in parsed.items() if k in allowed}
            doc_annotation_meta = {f"source_{k}": v for k, v in parsed.items()}
        except Exception:
            doc_annotation_meta = {}

        # Create a single Document with aggregated metadata
        document = Document(
            content=all_content,
            meta={
                "source_page_count": len(ocr_response.pages),
                "source_total_images": total_images,
                # Unpack document annotation
                **doc_annotation_meta,
            },
        )

        # Return single document and raw API response for flexibility
        return {"documents": [document], "raw_mistral_response": ocr_response}
