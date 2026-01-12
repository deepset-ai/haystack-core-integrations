# azure-doc-intelligence-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/azure-doc-intelligence-haystack.svg)](https://pypi.org/project/azure-doc-intelligence-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/azure-doc-intelligence-haystack.svg)](https://pypi.org/project/azure-doc-intelligence-haystack)

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

```console
pip install azure-doc-intelligence-haystack
```

For CSV table output support, install with the optional dependency:

```console
pip install "azure-doc-intelligence-haystack[csv]"
```

## Overview

This integration provides `AzureDocumentIntelligenceConverter`, a Haystack component that converts documents (PDF, images, Office files) to Haystack Documents using Azure's Document Intelligence service.

### Key Features

- **Markdown output** (recommended for RAG): Returns GitHub Flavored Markdown with inline tables, preserved headings, and document structure
- **Text output** with optional CSV tables for backward compatibility
- **Multiple model support**: `prebuilt-read` (fast OCR), `prebuilt-layout` (enhanced structure), `prebuilt-document` (general), or custom models
- **Supported formats**: PDF, JPEG, PNG, BMP, TIFF, DOCX, XLSX, PPTX, HTML

### Prerequisites

You need an active Azure account and a Document Intelligence or Cognitive Services resource. See the [Azure documentation](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api) for setup instructions.

## Usage

### Basic Usage (Markdown Output - Recommended for RAG)

```python
import os
from haystack_integrations.components.converters.azure_doc_intelligence import (
    AzureDocumentIntelligenceConverter,
)
from haystack.utils import Secret

converter = AzureDocumentIntelligenceConverter(
    endpoint=os.environ["AZURE_DI_ENDPOINT"],
    api_key=Secret.from_env_var("AZURE_AI_API_KEY"),
    output_format="markdown",  # Default
)

results = converter.run(sources=["invoice.pdf", "contract.docx"])
documents = results["documents"]

# Documents contain markdown with inline tables
print(documents[0].content)
# Output:
# # Invoice
#
# | Item | Quantity | Price |
# |------|----------|-------|
# | Widget | 10 | $50.00 |
#
# Total: $500.00
```

### Text Output with CSV Tables

For backward compatibility or when you need tables as separate documents:

```python
converter = AzureDocumentIntelligenceConverter(
    endpoint=os.environ["AZURE_DI_ENDPOINT"],
    api_key=Secret.from_env_var("AZURE_AI_API_KEY"),
    output_format="text",
    table_format="csv",
)

results = converter.run(sources=["document_with_tables.pdf"])

# Returns separate documents for text and each table
for doc in results["documents"]:
    if doc.meta.get("content_format") == "table":
        print(f"Table on page {doc.meta.get('page')}")
        print(doc.content)  # CSV content
    else:
        print("Text content:", doc.content)
```

### Enhanced Layout Analysis

Use `prebuilt-layout` model for better table and structure detection:

```python
converter = AzureDocumentIntelligenceConverter(
    endpoint=os.environ["AZURE_DI_ENDPOINT"],
    api_key=Secret.from_env_var("AZURE_AI_API_KEY"),
    model_id="prebuilt-layout",  # Better structure detection
    output_format="markdown",
)

results = converter.run(sources=["complex_document.pdf"])
```

### Using in a Haystack Pipeline

```python
from haystack import Pipeline
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.converters.azure_doc_intelligence import (
    AzureDocumentIntelligenceConverter,
)
from haystack.utils import Secret
import os

document_store = InMemoryDocumentStore()

pipeline = Pipeline()
pipeline.add_component(
    "converter",
    AzureDocumentIntelligenceConverter(
        endpoint=os.environ["AZURE_DI_ENDPOINT"],
        api_key=Secret.from_env_var("AZURE_AI_API_KEY"),
    ),
)
pipeline.add_component(
    "splitter",
    DocumentSplitter(split_by="sentence", split_length=5),
)
pipeline.add_component(
    "writer",
    DocumentWriter(document_store=document_store),
)

pipeline.connect("converter", "splitter")
pipeline.connect("splitter", "writer")

pipeline.run({"converter": {"sources": ["document.pdf"]}})
```

### Component Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endpoint` | `str` | Required | Azure Document Intelligence endpoint URL |
| `api_key` | `Secret` | `Secret.from_env_var("AZURE_AI_API_KEY")` | API key for authentication |
| `model_id` | `str` | `"prebuilt-read"` | Azure model ID (`prebuilt-read`, `prebuilt-layout`, `prebuilt-document`, or custom) |
| `output_format` | `Literal["text", "markdown"]` | `"markdown"` | Output format for document content |
| `table_format` | `Literal["csv", "markdown"]` | `"markdown"` | How to format tables when `output_format="text"` |
| `store_full_path` | `bool` | `False` | Store full file path in metadata (vs. just filename) |

### Output

The `run()` method returns a dictionary with:
- `documents`: List of Haystack `Document` objects
- `raw_azure_response`: List of raw Azure API responses (as dicts)

Each document includes metadata:
- `content_format`: `"markdown"`, `"text"`, or `"table"`
- `model_id`: The Azure model used
- `page_count`: Number of pages (for markdown output)
- `file_path`: Source file path/name
- `page`: Page number (for table documents)

## License

`azure-doc-intelligence-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
