import json
import re
from typing import Any, Dict, List, Optional, Type, Union

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
    This component extracts text from documents using Mistral's OCR API, with optional structured
    annotations for both individual image regions (bounding boxes) and full documents.

    Accepts document sources (DocumentURLChunk for document URLs, ImageURLChunk for image URLs,
    or FileChunk for Mistral file IDs) and retrieves the recognized text via Mistral's OCR service.
    Returns a single Haystack Document containing all pages concatenated with form feed characters (\\f),
    ensuring compatibility with Haystack's DocumentSplitter for accurate page-wise splitting and overlap handling.

    **How Annotations Work:**
    When annotation schemas (`bbox_annotation_schema` or `document_annotation_schema`) are provided,
    the OCR model first extracts text and structure from the document. Then, a Vision LLM is called
    to analyze the content and generate structured annotations according to your defined schemas.
    For more details, see: https://docs.mistral.ai/capabilities/document_ai/annotations/#how-it-works

    **API Reference:**
    - Basic OCR: https://docs.mistral.ai/capabilities/document_ai/basic_ocr/
    - Annotations: https://docs.mistral.ai/capabilities/document_ai/annotations/

    **Usage Example:**
    ```python
    from haystack.utils import Secret
    from haystack_integrations.mistral import MistralOCRDocumentConverter
    from mistralai.models import DocumentURLChunk, ImageURLChunk, FileChunk

    converter = MistralOCRDocumentConverter(
        api_key=Secret.from_env_var("MISTRAL_API_KEY"),
        model="mistral-ocr-2505"
    )

    # Process multiple sources
    sources = [
        DocumentURLChunk(document_url="https://example.com/document.pdf"),
        ImageURLChunk(image_url="https://example.com/receipt.jpg"),
        FileChunk(file_id="file-abc123"),
    ]
    result = converter.run(sources=sources)

    documents = result["documents"]  # List of 3 Documents
    raw_responses = result["raw_mistral_response"]  # List of 3 raw responses
    ```

    **Structured Output Example:**
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
        chapter_titles: List[str] = Field(..., description="Detected chapter or section titles")
        urls: List[str] = Field(..., description="URLs found in the text")

    converter = MistralOCRDocumentConverter(
        api_key=Secret.from_env_var("MISTRAL_API_KEY"),
        model="mistral-ocr-2505",
    )

    sources = [DocumentURLChunk(document_url="https://example.com/report.pdf")]
    result = converter.run(
        sources=sources,
        bbox_annotation_schema=ImageAnnotation,
        document_annotation_schema=DocumentAnnotation,
    )

    documents = result["documents"]
    raw_responses = result["raw_mistral_response"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
        model: str = "mistral-ocr-2505",
        include_image_base64: bool = False,
        pages: Optional[List[int]] = None,
        image_limit: Optional[int] = None,
        image_min_size: Optional[int] = None,
    ):
        """
        Creates a MistralOCRDocumentConverter component.

        :param api_key:
            The Mistral API key. Defaults to the MISTRAL_API_KEY environment variable.
        :param model:
            The OCR model to use. Default is "mistral-ocr-2505".
        :param include_image_base64:
            If True, includes base64 encoded images in the response.
            This may significantly increase response size and processing time.
        :param pages:
            Specific page numbers to process (0-indexed). If None, processes all pages.
        :param image_limit:
            Maximum number of images to extract from the document.
        :param image_min_size:
            Minimum height and width (in pixels) for images to be extracted.
        """
        self.api_key = api_key
        self.model = model
        self.include_image_base64 = include_image_base64
        self.pages = pages
        self.image_limit = image_limit
        self.image_min_size = image_min_size

        # Initialize Mistral client
        self.client = Mistral(api_key=self.api_key.resolve_value())

    @component.output_types(documents=List[Document], raw_mistral_response=List[Dict[str, Any]])
    def run(
        self,
        sources: List[Union[DocumentURLChunk, FileChunk, ImageURLChunk]],
        bbox_annotation_schema: Optional[Type[BaseModel]] = None,
        document_annotation_schema: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """
        Extract text from documents using Mistral OCR.

        :param sources:
            List of document sources to process. Each source can be one of:
            - DocumentURLChunk: For document URLs (signed or public URLs to PDFs, etc.)
            - ImageURLChunk: For image URLs (signed or public URLs to images)
            - FileChunk: For Mistral file IDs (files previously uploaded to Mistral)
        :param bbox_annotation_schema:
            Optional Pydantic model for structured annotations per bounding box.
            When provided, a Vision LLM analyzes each image region and returns structured data.
        :param document_annotation_schema:
            Optional Pydantic model for structured annotations for the full document.
            When provided, a Vision LLM analyzes the entire document and returns structured data.

        :returns:
            A dictionary with the following keys:
            - `documents`: List of Haystack Documents (one per source). Each Document has the following structure:
                - `content`: All pages joined with form feed (\\f) separators in markdown format.
                  When using bbox_annotation_schema, image tags will be enriched with your defined descriptions.
                - `meta`: Aggregated metadata dictionary with structure:
                  {"source_page_count": int, "source_total_images": int, "source_*": any}.
                  If document_annotation_schema was provided, all annotation fields are unpacked
                  with 'source_' prefix (e.g., source_language, source_chapter_titles, source_urls).
            - `raw_mistral_response`: List of dictionaries containing raw OCR responses from Mistral API (one per source).
              Each response includes per-page details, images, annotations, and usage info.
        """
        # Convert Pydantic models to Mistral ResponseFormat schemas
        bbox_annotation_format = (
            response_format_from_pydantic_model(bbox_annotation_schema) if bbox_annotation_schema else None
        )
        document_annotation_format = (
            response_format_from_pydantic_model(document_annotation_schema) if document_annotation_schema else None
        )

        # Process each source
        documents = []
        raw_responses = []

        for source in sources:
            # Call Mistral OCR API with the provided source
            ocr_response: OCRResponse = self.client.ocr.process(
                model=self.model,
                document=source,
                include_image_base64=self.include_image_base64,
                pages=self.pages,
                image_limit=self.image_limit,
                image_min_size=self.image_min_size,
                bbox_annotation_format=bbox_annotation_format,
                document_annotation_format=document_annotation_format,
            )

            # Process the OCR response into a Document
            document = self._process_ocr_response(ocr_response, document_annotation_schema)
            documents.append(document)
            raw_responses.append(ocr_response.to_dict())

        return {"documents": documents, "raw_mistral_response": raw_responses}

    def _process_ocr_response(
        self,
        ocr_response: OCRResponse,
        document_annotation_schema: Optional[Type[BaseModel]],
    ) -> Document:
        """
        Convert an OCR response from Mistral API into a single Haystack Document.

        :param ocr_response:
            The OCR response object from Mistral API.
        :param document_annotation_schema:
            Optional Pydantic model for document-level annotations.

        :returns:
            A single Haystack Document containing the processed OCR content.
        """
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
            if document_annotation_schema:
                allowed = document_annotation_schema.model_fields.keys()
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

        return document
