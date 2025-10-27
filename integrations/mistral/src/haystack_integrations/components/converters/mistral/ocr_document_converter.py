import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from mistralai.models import (
    DocumentURLChunk,
    FileChunk,
    ImageURLChunk,
    OCRResponse,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@component
class MistralOCRDocumentConverter:
    """
    This component extracts text from documents using Mistral's OCR API, with optional structured
    annotations for both individual image regions (bounding boxes) and full documents.

    Accepts document sources in various formats (str/Path for local files, ByteStream for in-memory data,
    DocumentURLChunk for document URLs, ImageURLChunk for image URLs, or FileChunk for Mistral file IDs)
    and retrieves the recognized text via Mistral's OCR service. Local files are automatically uploaded
    to Mistral's storage.
    Returns Haystack Documents (one per source) containing all pages concatenated with form feed characters (\\f),
    ensuring compatibility with Haystack's DocumentSplitter for accurate page-wise splitting and overlap handling.

    **How Annotations Work:**
    When annotation schemas (`bbox_annotation_schema` or `document_annotation_schema`) are provided,
    the OCR model first extracts text and structure from the document. Then, a Vision LLM is called
    to analyze the content and generate structured annotations according to your defined schemas.
    For more details, see: https://docs.mistral.ai/capabilities/document_ai/annotations/#how-it-works

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
        cleanup_uploaded_files: bool = True,
    ):
        """
        Creates a MistralOCRDocumentConverter component.

        :param api_key:
            The Mistral API key. Defaults to the MISTRAL_API_KEY environment variable.
        :param model:
            The OCR model to use. Default is "mistral-ocr-2505".
            See more: https://docs.mistral.ai/getting-started/models/models_overview/
        :param include_image_base64:
            If True, includes base64 encoded images in the response.
            This may significantly increase response size and processing time.
        :param pages:
            Specific page numbers to process (0-indexed). If None, processes all pages.
        :param image_limit:
            Maximum number of images to extract from the document.
        :param image_min_size:
            Minimum height and width (in pixels) for images to be extracted.
        :param cleanup_uploaded_files:
            If True, automatically deletes files uploaded to Mistral after processing.
            Only affects files uploaded from local sources (str, Path, ByteStream).
            Files provided as FileChunk are not deleted. Default is True.
        """
        self.api_key = api_key
        self.model = model
        self.include_image_base64 = include_image_base64
        self.pages = pages
        self.image_limit = image_limit
        self.image_min_size = image_min_size
        self.cleanup_uploaded_files = cleanup_uploaded_files

        # Initialize Mistral client
        self.client = Mistral(api_key=self.api_key.resolve_value())

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            include_image_base64=self.include_image_base64,
            pages=self.pages,
            image_limit=self.image_limit,
            image_min_size=self.image_min_size,
            cleanup_uploaded_files=self.cleanup_uploaded_files,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MistralOCRDocumentConverter":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document], raw_mistral_response=List[Dict[str, Any]])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream, DocumentURLChunk, FileChunk, ImageURLChunk]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        bbox_annotation_schema: Optional[Type[BaseModel]] = None,
        document_annotation_schema: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """
        Extract text from documents using Mistral OCR.

        :param sources:
            List of document sources to process. Each source can be one of:
            - str: File path to a local document
            - Path: Path object to a local document
            - ByteStream: Haystack ByteStream object containing document data
            - DocumentURLChunk: Mistral chunk for document URLs (signed or public URLs to PDFs, etc.)
            - ImageURLChunk: Mistral chunk for image URLs (signed or public URLs to images)
            - FileChunk: Mistral chunk for file IDs (files previously uploaded to Mistral)
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because they will be zipped.
        :param bbox_annotation_schema:
            Optional Pydantic model for structured annotations per bounding box.
            When provided, a Vision LLM analyzes each image region and returns structured data.
        :param document_annotation_schema:
            Optional Pydantic model for structured annotations for the full document.
            When provided, a Vision LLM analyzes the entire document and returns structured data.
            Note: Document annotation is limited to a maximum of 8 pages. Documents exceeding
            this limit will not be processed for document annotation.

        :returns:
            A dictionary with the following keys:
            - `documents`: List of Haystack Documents (one per source). Each Document has the following structure:
                - `content`: All pages joined with form feed (\\f) separators in markdown format.
                  When using bbox_annotation_schema, image tags will be enriched with your defined descriptions.
                - `meta`: Aggregated metadata dictionary with structure:
                  `{"source_page_count": int, "source_total_images": int, "source_*": any}`.
                  If document_annotation_schema was provided, all annotation fields are unpacked
                  with 'source_' prefix (e.g., source_language, source_chapter_titles, source_urls).
            - `raw_mistral_response`:
                List of dictionaries containing raw OCR responses from Mistral API (one per source).
                Each response includes per-page details, images, annotations, and usage info.
        """
        # Convert Pydantic models to Mistral ResponseFormat schemas
        bbox_annotation_format = (
            response_format_from_pydantic_model(bbox_annotation_schema) if bbox_annotation_schema else None
        )
        document_annotation_format = (
            response_format_from_pydantic_model(document_annotation_schema) if document_annotation_schema else None
        )

        # Normalize metadata
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        # Process each source
        documents = []
        raw_responses = []
        uploaded_file_ids = []

        for source, user_metadata in zip(sources, meta_list):
            document, raw_response, uploaded_file_id = self._process_single_source(
                source,
                user_metadata,
                bbox_annotation_format,
                document_annotation_format,
                document_annotation_schema,
            )

            # Add results if processing succeeded
            if document is not None and raw_response is not None:
                documents.append(document)
                raw_responses.append(raw_response)

            # Track uploaded file for cleanup even if processing failed
            if uploaded_file_id:
                uploaded_file_ids.append(uploaded_file_id)

        # Cleanup uploaded files
        self._cleanup_uploaded_files(uploaded_file_ids)

        return {"documents": documents, "raw_mistral_response": raw_responses}

    def _process_single_source(
        self,
        source: Union[str, Path, ByteStream, DocumentURLChunk, FileChunk, ImageURLChunk],
        user_metadata: Dict[str, Any],
        bbox_annotation_format: Optional[Any],
        document_annotation_format: Optional[Any],
        document_annotation_schema: Optional[Type[BaseModel]],
    ) -> tuple[Optional[Document], Optional[Dict[str, Any]], Optional[str]]:
        """
        Process a single source and return the document, raw response, and file_id if uploaded.

        :param source:
            The source to process.
        :param user_metadata:
            User-provided metadata to attach to the document.
        :param bbox_annotation_format:
            Optional response format for bounding box annotations.
        :param document_annotation_format:
            Optional response format for document annotations.
        :param document_annotation_schema:
            Optional Pydantic model for document-level annotations.

        :returns:
            A tuple of (Document|None, raw_response_dict|None, uploaded_file_id|None).
            Returns (None, None, uploaded_file_id) if processing fails but file was uploaded.
        """
        uploaded_file_id = None
        try:
            chunk = self._convert_source_to_chunk(source)

            # Track if we uploaded this file
            if isinstance(source, (str, Path, ByteStream)) and isinstance(chunk, FileChunk):
                uploaded_file_id = chunk.file_id

            ocr_response: OCRResponse = self.client.ocr.process(
                model=self.model,
                document=chunk,
                include_image_base64=self.include_image_base64,
                pages=self.pages,
                image_limit=self.image_limit,
                image_min_size=self.image_min_size,
                bbox_annotation_format=bbox_annotation_format,
                document_annotation_format=document_annotation_format,
            )

            document = self._process_ocr_response(ocr_response, user_metadata, document_annotation_schema)
            return (document, ocr_response.model_dump(), uploaded_file_id)
        except Exception as e:
            logger.warning(
                "Could not process source {source}. Skipping it. Error: {error}",
                source=source,
                error=e,
            )
            return (None, None, uploaded_file_id)

    def _cleanup_uploaded_files(self, file_ids: List[str]) -> None:
        """
        Delete uploaded files from Mistral storage.

        :param file_ids:
            List of file IDs to delete.
        """
        if not self.cleanup_uploaded_files or not file_ids:
            return

        for file_id in file_ids:
            try:
                self.client.files.delete(file_id=file_id)
            except Exception as e:
                logger.warning(
                    "Failed to delete uploaded file {file_id}. Error: {error}",
                    file_id=file_id,
                    error=e,
                )

    def _convert_source_to_chunk(
        self,
        source: Union[str, Path, ByteStream, DocumentURLChunk, FileChunk, ImageURLChunk],
    ) -> Union[DocumentURLChunk, FileChunk, ImageURLChunk]:
        """
        Convert various source types to Mistral-compatible chunk format.

        Local sources (str, Path, ByteStream) are uploaded to Mistral's storage and returned
        as FileChunk. Remote sources (DocumentURLChunk, ImageURLChunk, FileChunk) are returned as-is.

        :param source:
            The source to convert. Can be a file path (str/Path), ByteStream, or Mistral chunk type.

        :returns:
            A Mistral chunk type (DocumentURLChunk, FileChunk, or ImageURLChunk).
        """
        # If already a Mistral chunk type, return as-is
        if isinstance(source, (DocumentURLChunk, FileChunk, ImageURLChunk)):
            return source

        # Convert str/Path/ByteStream to ByteStream
        bytestream = get_bytestream_from_source(source=source)

        # Upload file to Mistral and get file ID
        uploaded_file = self.client.files.upload(
            file={
                "file_name": bytestream.meta.get("file_path", "document"),
                "content": bytestream.data,
            },
            purpose="ocr",
        )

        # Return FileChunk with the uploaded file ID
        return FileChunk(file_id=uploaded_file.id)

    def _process_ocr_response(
        self,
        ocr_response: OCRResponse,
        user_metadata: Dict[str, Any],
        document_annotation_schema: Optional[Type[BaseModel]],
    ) -> Document:
        """
        Convert an OCR response from Mistral API into a single Haystack Document.

        :param ocr_response:
            The OCR response object from Mistral API.
        :param user_metadata:
            User-provided metadata to attach to the document.
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
                # User metadata (lowest priority - can be overridden)
                **user_metadata,
                # Automatic metadata (medium priority)
                "source_page_count": len(ocr_response.pages),
                "source_total_images": total_images,
                # Document annotation (highest priority - overrides all)
                **doc_annotation_meta,
            },
        )

        return document
