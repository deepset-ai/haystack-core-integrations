# To run this example, you will need to set a `MISTRAL_API_KEY` environment variable.
# This example demonstrates OCR document processing with structured annotations.

from mistralai.models import DocumentURLChunk
from pydantic import BaseModel, Field

from haystack_integrations.components.converters.mistral.ocr_document_converter import (
    MistralOCRDocumentConverter,
)


# Define schema for structured image annotations (bbox)
class ImageAnnotation(BaseModel):
    image_type: str = Field(..., description="The type of image content")
    description: str = Field(..., description="Brief description of the image")


# Define schema for structured document annotations
class DocumentAnnotation(BaseModel):
    language: str = Field(..., description="Primary language of the document")
    urls: list[str] = Field(..., description="URLs found in the document")
    topics: list[str] = Field(..., description="Main topics covered in the document")


# Initialize the converter with annotation schemas
converter = MistralOCRDocumentConverter(
    pages=[2, 3],
    bbox_annotation_schema=ImageAnnotation,
    document_annotation_schema=DocumentAnnotation,
)

# Process a document URL (you can use any public or signed URL to a PDF or image)
doc_source = DocumentURLChunk(document_url="https://arxiv.org/pdf/1706.03762")

# Run OCR
result = converter.run(source=doc_source)

# Extract results
documents = result["documents"]
raw_mistral_response = result["raw_mistral_response"]
