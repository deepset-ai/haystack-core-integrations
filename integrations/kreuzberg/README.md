# kreuzberg-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/kreuzberg-haystack.svg)](https://pypi.org/project/kreuzberg-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/kreuzberg-haystack.svg)](https://pypi.org/project/kreuzberg-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/kreuzberg/CHANGELOG.md)

---

## Overview

A [Haystack](https://haystack.deepset.ai/) integration for [Kreuzberg](https://docs.kreuzberg.dev/), a document intelligence framework that extracts text from PDFs, Office documents, images, and 75+ other file formats. All processing is performed locally — no external API calls.

## Installation

```console
pip install kreuzberg-haystack
```

Kreuzberg requires system OCR libraries for image-based extraction. See the [Kreuzberg installation docs](https://docs.kreuzberg.dev/) for platform-specific setup (e.g. Tesseract, EasyOCR).

## Quick Start

```python
from haystack_integrations.components.converters.kreuzberg import KreuzbergConverter

converter = KreuzbergConverter()
result = converter.run(sources=["document.pdf", "report.docx"])

for doc in result["documents"]:
    print(doc.content[:200])
```

## Features

- **75+ file formats** — PDF, DOCX, PPTX, XLSX, HTML, images, email, ebooks, and more
- **Local processing** — No external API calls; fully offline capable
- **Batch extraction** — Parallel processing via Rust rayon thread pool
- **Per-page splitting** — One Document per page for fine-grained retrieval
- **Built-in chunking** — Server-side chunking via kreuzberg's `ChunkingConfig`
- **Multiple output formats** — Plain text, Markdown, or HTML
- **OCR backends** — Tesseract and EasyOCR with configurable language and preprocessing
- **Token reduction** — Reduce output size for LLM consumption (5 levels)
- **Rich metadata** — Tables, images, annotations, keywords, quality scores, detected languages
- **Configuration files** — Load settings from TOML, YAML, or JSON files

## Usage Examples

### Basic Conversion

```python
from haystack_integrations.components.converters.kreuzberg import KreuzbergConverter

converter = KreuzbergConverter()
result = converter.run(sources=["report.pdf"])
documents = result["documents"]
```

### Markdown Output with OCR

```python
from kreuzberg import ExtractionConfig, OcrConfig

converter = KreuzbergConverter(
    config=ExtractionConfig(
        output_format="markdown",
        ocr=OcrConfig(backend="tesseract", language="eng"),
    ),
)
result = converter.run(sources=["scanned_document.pdf"])
```

### Per-Page Extraction

```python
from kreuzberg import ExtractionConfig, PageConfig

converter = KreuzbergConverter(
    config=ExtractionConfig(
        pages=PageConfig(extract_pages=True),
    ),
)
result = converter.run(sources=["multi_page.pdf"])
# One Document per page, with page_number in metadata
```

### Token Reduction for LLMs

```python
from kreuzberg import ExtractionConfig, TokenReductionConfig

converter = KreuzbergConverter(
    config=ExtractionConfig(
        token_reduction=TokenReductionConfig(mode="moderate"),
    ),
)
```

### Directory Input

```python
converter = KreuzbergConverter()
# Expands to all files in the directory (non-recursive, sorted)
result = converter.run(sources=["./documents/"])
```

### In a Haystack Pipeline

```python
from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.converters.kreuzberg import KreuzbergConverter

document_store = InMemoryDocumentStore()

pipeline = Pipeline()
pipeline.add_component("converter", KreuzbergConverter())
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.add_component("writer", DocumentWriter(document_store=document_store))

pipeline.connect("converter.documents", "cleaner")
pipeline.connect("cleaner", "writer")

pipeline.run({"converter": {"sources": ["report.pdf", "notes.docx"]}})
```

### Configuration File

```python
converter = KreuzbergConverter(config_path="kreuzberg.toml")
```

Where `kreuzberg.toml` might contain:

```toml
output_format = "markdown"

[ocr]
backend = "tesseract"
language = "eng+deu"
```

## API Reference

### `KreuzbergConverter.__init__`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `ExtractionConfig \| None` | `None` | Kreuzberg extraction configuration object. Controls output format, OCR, chunking, keyword extraction, and more. |
| `config_path` | `str \| Path \| None` | `None` | Path to a kreuzberg config file (TOML/YAML/JSON). When both `config` and `config_path` are given, `config` takes precedence. |
| `store_full_path` | `bool` | `False` | If `True`, store full file paths in metadata. If `False`, store only the file name. |
| `batch` | `bool` | `True` | Use kreuzberg's batch APIs for parallel extraction via Rust rayon thread pool. |
| `easyocr_kwargs` | `dict \| None` | `None` | Extra keyword arguments for EasyOCR (GPU, beam width, model storage, etc.). |

### `KreuzbergConverter.run`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sources` | `list[str \| Path \| ByteStream]` | *(required)* | File paths, directory paths, or ByteStream objects to convert. |
| `meta` | `dict \| list[dict] \| None` | `None` | Metadata to attach to Documents. A single dict applies to all; a list is zipped with sources. |

**Returns** a dict with:
- `documents` — `list[Document]`: Converted documents with content and metadata.
- `raw_extraction` — `list[dict]`: Serialized kreuzberg `ExtractionResult` for each source (useful for debugging).

## Metadata Fields

Each Document's `meta` dict may include the following fields (depending on source format and configuration):

| Field | Type | Description |
|---|---|---|
| `file_path` | `str` | Source file name (or full path if `store_full_path=True`) |
| `mime_type` | `str` | Detected MIME type of the source |
| `file_extensions` | `list[str]` | Known extensions for the MIME type |
| `output_format` | `str` | Output format used (plain/markdown/html) |
| `result_format` | `str` | Result format from kreuzberg |
| `quality_score` | `float` | Extraction quality score (0.0–1.0) |
| `detected_languages` | `list[str]` | Languages detected in the content |
| `extracted_keywords` | `list[dict]` | Keywords with text, score, and algorithm |
| `table_count` | `int` | Number of tables extracted |
| `tables` | `list[dict]` | Table data (cells, markdown, page_number) |
| `image_count` | `int` | Number of images found |
| `images` | `list[dict]` | Image metadata (format, dimensions, page, description) |
| `annotations` | `list[dict]` | PDF annotations (type, content, page_number) |
| `processing_warnings` | `list[dict]` | Warnings from extraction (source, message) |
| `page_number` | `int` | Page number (per-page mode only) |
| `is_blank` | `bool` | Whether the page is blank (per-page mode only) |
| `chunk_index` | `int` | Chunk index (chunking mode only) |
| `total_chunks` | `int` | Total chunks (chunking mode only) |

Format-specific metadata from kreuzberg (e.g. PDF title, author, page count) is also flattened into `meta`.

## Supported Formats

Kreuzberg supports 75+ file formats. You can query the available extractors at runtime:

```python
KreuzbergConverter.supported_extractors()
KreuzbergConverter.supported_ocr_backends()
```

Common supported formats include:

| Category | Formats |
|---|---|
| Documents | PDF, DOCX, DOC, ODT, RTF, EPUB |
| Spreadsheets | XLSX, XLS, ODS, CSV, TSV |
| Presentations | PPTX, PPT, ODP |
| Images | PNG, JPEG, TIFF, BMP, WebP (via OCR) |
| Web | HTML, XHTML, XML, Markdown |
| Email | EML, MSG |
| Code | Plain text, source code files |
| Archives | Extracts from contained documents |

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run tests locally:

```console
# Install the integration in development mode
pip install -e ".[dev]"

# Run unit tests
pytest tests/
```

No external services or API keys are required — kreuzberg processes everything locally. For OCR tests, ensure Tesseract is installed on your system.

## License

`kreuzberg-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
