# To run this example, you will need to set a `MISTRAL_API_KEY` environment variable.
# This example demonstrates OCR document processing with structured annotations.

from typing import List

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
    urls: List[str] = Field(..., description="URLs found in the document")
    topics: List[str] = Field(..., description="Main topics covered in the document")


# Initialize the converter
converter = MistralOCRDocumentConverter(
    pages=[2, 3],
)

# Process document URLs (you can use any public or signed URL to a PDF or image)
sources = [DocumentURLChunk(document_url="https://arxiv.org/pdf/1706.03762")]

# Run OCR with annotation schemas
result = converter.run(
    sources=sources,
    bbox_annotation_schema=ImageAnnotation,
    document_annotation_schema=DocumentAnnotation,
)

# Extract results
documents = result["documents"]
raw_mistral_response = result["raw_mistral_response"]
