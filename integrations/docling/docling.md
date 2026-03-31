---
title: "Docling"
id: integrations-docling
description: "Docling integration for Haystack"
slug: "/integrations-docling"
---


## haystack_integrations.components.converters.docling.converter

Docling Haystack converter module.

### ExportType

Bases: <code>str</code>, <code>Enum</code>

Enumeration of available export types.

### BaseMetaExtractor

Bases: <code>ABC</code>

BaseMetaExtractor.

#### extract_chunk_meta

```python
extract_chunk_meta(chunk: BaseChunk) -> dict[str, Any]
```

Extract chunk meta.

#### extract_dl_doc_meta

```python
extract_dl_doc_meta(dl_doc: DoclingDocument) -> dict[str, Any]
```

Extract Docling document meta.

### MetaExtractor

Bases: <code>BaseMetaExtractor</code>

MetaExtractor.

#### extract_chunk_meta

```python
extract_chunk_meta(chunk: BaseChunk) -> dict[str, Any]
```

Extract chunk meta.

#### extract_dl_doc_meta

```python
extract_dl_doc_meta(dl_doc: DoclingDocument) -> dict[str, Any]
```

Extract Docling document meta.

### DoclingConverter

Docling Haystack converter.

#### __init__

```python
__init__(
    converter: DocumentConverter | None = None,
    convert_kwargs: dict[str, Any] | None = None,
    export_type: ExportType = ExportType.DOC_CHUNKS,
    md_export_kwargs: dict[str, Any] | None = None,
    chunker: BaseChunker | None = None,
    meta_extractor: BaseMetaExtractor | None = None,
) -> None
```

Create a Docling Haystack converter.

Args:
converter: The Docling `DocumentConverter` to use; if not set, a system
default is used.
convert_kwargs: Any parameters to pass to Docling conversion; if not set, a
system default is used.
export_type: The export mode to use:
\* `ExportType.MARKDOWN` captures each input document as a single
markdown `Document`.
\* `ExportType.DOC_CHUNKS` (default) first chunks each input document
and then returns one `Document` per chunk.
\* `ExportType.JSON` serializes the full Docling document to a JSON string.
md_export_kwargs: Any parameters to pass to Markdown export (applicable in
case of `ExportType.MARKDOWN`).
chunker: The Docling chunker instance to use; if not set, a system default
is used.
meta_extractor: The extractor instance to use for populating the output
document metadata; if not set, a system default is used.

#### run

```python
run(paths: Iterable[Path | str]) -> dict[str, list[Document]]
```

Run the DoclingConverter.

Args:
paths: The input document locations, either as local paths or URLs.

Returns:
list\[Document\]: The output Haystack Documents.
