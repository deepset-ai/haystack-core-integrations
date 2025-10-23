# To run this example, you will need to:
# 1. Set a `MISTRAL_API_KEY` environment variable
# 2. Place a PDF file named `sample.pdf` in the same directory as this script
#
# This example demonstrates OCR document processing with structured annotations,
# embedding the extracted documents using Mistral embeddings, and storing them
# in an InMemoryDocumentStore for later retrieval.
#
# You can customize the ImageAnnotation and DocumentAnnotation schemas below
# to extract different structured information from your documents.

from typing import List

from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from mistralai.models import DocumentURLChunk
from pydantic import BaseModel, Field

from haystack_integrations.components.converters.mistral.ocr_document_converter import (
    MistralOCRDocumentConverter,
)
from haystack_integrations.components.embedders.mistral.document_embedder import (
    MistralDocumentEmbedder,
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


# Initialize document store
document_store = InMemoryDocumentStore()

# Create indexing pipeline
indexing_pipeline = Pipeline()

# Add components to the pipeline
indexing_pipeline.add_component(
    "converter",
    MistralOCRDocumentConverter(pages=[0, 1]),
)
indexing_pipeline.add_component(
    "embedder",
    MistralDocumentEmbedder(),
)
indexing_pipeline.add_component(
    "writer",
    DocumentWriter(document_store=document_store),
)

# Connect components
indexing_pipeline.connect("converter.documents", "embedder.documents")
indexing_pipeline.connect("embedder.documents", "writer.documents")

# Prepare sources: URL and local file
sources = [
    DocumentURLChunk(document_url="https://arxiv.org/pdf/1706.03762"),
    "./sample.pdf",  # Local PDF file
]

# Run the pipeline with annotation schemas
result = indexing_pipeline.run(
    {
        "converter": {
            "sources": sources,
            "bbox_annotation_schema": ImageAnnotation,
            "document_annotation_schema": DocumentAnnotation,
        }
    }
)


# Check out documents processed by OCR.
# Optional with enriched content (from bbox annotation) and semantic meta data (from document annotation)
documents = document_store.storage
# Check out mistral api response for unprocessed data and with usage_info
raw_mistral_response = result["converter"]["raw_mistral_response"]
