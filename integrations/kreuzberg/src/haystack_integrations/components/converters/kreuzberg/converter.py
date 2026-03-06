# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import copy
import tempfile
from pathlib import Path
from typing import Any, cast

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream

from kreuzberg import (  # type: ignore[attr-defined]
    ExtractedImage,
    ExtractedTable,
    ExtractionConfig,
    ExtractionResult,
    LanguageDetectionConfig,
    batch_extract_bytes_sync,
    batch_extract_files_sync,
    config_merge,
    config_to_json,
    detect_mime_type,
    error_code_name,
    extract_bytes_sync,
    extract_file_sync,
    get_error_details,
    get_extensions_for_mime,
    get_last_error_code,
    list_document_extractors,
    list_ocr_backends,
)

logger = logging.getLogger(__name__)

# Metadata keys that duplicate top-level ExtractionResult fields.
# These are excluded when flattening result.metadata into Document.meta
# because we handle them from the top-level fields instead.
_METADATA_OVERLAP_KEYS = frozenset({"quality_score", "output_format", "keywords"})


@component
class KreuzbergConverter:
    """
    Converts files to Documents using [Kreuzberg](https://docs.kreuzberg.dev/).

    Kreuzberg is a document intelligence framework that extracts text from
    PDFs, Office documents, images, and 75+ other formats. All processing
    is performed locally with no external API calls.

    **Usage Example:**

    ```python
    from haystack_integrations.components.converters.kreuzberg import (
        KreuzbergConverter,
    )

    converter = KreuzbergConverter()
    result = converter.run(sources=["document.pdf", "report.docx"])
    documents = result["documents"]
    ```

    You can also pass kreuzberg's `ExtractionConfig` to customize extraction:

    ```python
    from kreuzberg import ExtractionConfig, OcrConfig

    converter = KreuzbergConverter(
        config=ExtractionConfig(
            output_format="markdown",
            ocr=OcrConfig(backend="tesseract", language="eng"),
        ),
    )
    ```

    The converter exposes two output sockets: `documents` and
    `raw_extraction`. The `raw_extraction` output contains the serialized
    kreuzberg `ExtractionResult` for each source, useful for debugging or
    advanced downstream processing.

    **Token reduction** can be configured via
    ``ExtractionConfig(token_reduction=TokenReductionConfig(mode="moderate"))``
    to reduce output size for LLM consumption. Five levels are available:
    ``"off"``, ``"light"``, ``"moderate"``, ``"aggressive"``, ``"maximum"``.
    The reduced text appears directly in ``Document.content``.

    **Image preprocessing for OCR** can be tuned via
    ``OcrConfig(tesseract_config=TesseractConfig(preprocessing=ImagePreprocessingConfig(...)))``
    with options for target DPI, auto-rotate, deskew, denoise,
    contrast enhancement, and binarization method.
    """

    def __init__(
        self,
        *,
        config: ExtractionConfig | None = None,
        config_path: str | Path | None = None,
        store_full_path: bool = False,
        batch: bool = True,
        easyocr_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a `KreuzbergConverter` component.

        :param config:
            An optional `kreuzberg.ExtractionConfig` object to customize
            extraction behavior. Use this to set output format, OCR backend
            and language, force-OCR mode, per-page extraction, chunking,
            keyword extraction, and other kreuzberg options. If not provided,
            kreuzberg's defaults are used.
        :param config_path:
            Path to a kreuzberg configuration file (`.toml`, `.yaml`, or
            `.json`). When both `config` and `config_path` are provided,
            `config` takes precedence (is merged on top of the file config).
        :param store_full_path:
            If `True`, the full file path is stored in the Document metadata.
            If `False`, only the file name is stored.
        :param batch:
            If `True`, use kreuzberg's batch extraction APIs, which leverage
            Rust's rayon thread pool for parallel processing. If `False`,
            sources are extracted one at a time.
        :param easyocr_kwargs:
            Optional keyword arguments to pass to EasyOCR when using the
            `"easyocr"` backend. Supports GPU, beam width, model storage,
            and other EasyOCR-specific options.
        """
        self.config = config
        self.config_path = Path(config_path).as_posix() if config_path is not None else None
        self.store_full_path = store_full_path
        self.batch = batch
        self.easyocr_kwargs = easyocr_kwargs

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        config_json = config_to_json(self.config) if self.config else None
        return default_to_dict(
            self,
            config=config_json,
            config_path=self.config_path,
            store_full_path=self.store_full_path,
            batch=self.batch,
            easyocr_kwargs=self.easyocr_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KreuzbergConverter":
        """
        Deserialize this component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        data = {**data, "init_parameters": dict(data.get("init_parameters", {}))}
        init_params = data["init_parameters"]
        config_data = init_params.get("config")
        if isinstance(config_data, str):
            init_params["config"] = _config_from_json_str(config_data)
        return default_from_dict(cls, data)

    def _build_config(self) -> ExtractionConfig:
        """
        Build the effective `ExtractionConfig` by merging all configuration
        sources in priority order: file config < explicit config.

        Always returns a *fresh* config object — never mutates ``self.config``.
        """
        # Determine base config (never mutate self.config)
        if self.config is not None:
            # Deep copy via JSON round-trip to avoid mutating the user's object
            config = _copy_config(self.config)
        elif self.config_path is not None:
            config = ExtractionConfig.from_file(self.config_path)
        else:
            config = ExtractionConfig()

        # When both config and config_path are provided, fill gaps in
        # config from the file config.  config_merge(base, source) fills
        # unset fields in *base* from *source* — base values always win.
        if self.config is not None and self.config_path is not None:
            file_config = ExtractionConfig.from_file(self.config_path)
            config_merge(config, file_config)

        # Auto-enable language detection if not explicitly configured
        if config.language_detection is None:
            config.language_detection = LanguageDetectionConfig(enabled=True)

        return config

    @staticmethod
    def _expand_sources(sources: list[str | Path | ByteStream]) -> list[str | Path | ByteStream]:
        """
        Expand directory paths in the sources list to their direct children.

        Directories are expanded non-recursively (only direct file children).
        Files are sorted alphabetically for deterministic ordering.
        """
        expanded: list[str | Path | ByteStream] = []
        for source in sources:
            if isinstance(source, ByteStream):
                expanded.append(source)
            else:
                path = Path(source)
                if path.is_dir():
                    expanded.extend(sorted(path.glob("*.*")))
                else:
                    expanded.append(source)
        return expanded

    def _extract_single(
        self,
        source: str | Path | ByteStream,
        config: ExtractionConfig,
        mime_type: str | None = None,
    ) -> ExtractionResult:
        """
        Extract content from a single source using kreuzberg.

        :param mime_type:
            Optional MIME type override. When provided, skips auto-detection.
        """
        if isinstance(source, ByteStream):
            effective_mime = mime_type or source.mime_type or detect_mime_type(source.data)
            return extract_bytes_sync(
                source.data,
                mime_type=effective_mime,
                config=config,
                easyocr_kwargs=self.easyocr_kwargs,
            )
        return extract_file_sync(source, mime_type=mime_type, config=config, easyocr_kwargs=self.easyocr_kwargs)

    def _extract_batch(
        self,
        sources: list[str | Path | ByteStream],
        config: ExtractionConfig,
    ) -> list[ExtractionResult | None]:
        """
        Extract content from multiple sources using kreuzberg's batch APIs.

        The result list is indexed to match ``sources``.  Slots for
        file-based sources are filled by ``batch_extract_files_sync``;
        slots for bytes-based sources by ``batch_extract_bytes_sync``.
        A slot remains ``None`` only if neither batch API populated it
        (i.e. the source was not dispatched to either batch call).
        """
        file_indices: list[int] = []
        file_paths: list[str | Path] = []
        bytes_indices: list[int] = []
        bytes_data: list[bytes | bytearray] = []
        bytes_mimes: list[str] = []

        for i, source in enumerate(sources):
            if isinstance(source, ByteStream):
                bytes_indices.append(i)
                bytes_data.append(source.data)
                bytes_mimes.append(source.mime_type or detect_mime_type(source.data))
            else:
                file_indices.append(i)
                file_paths.append(source)

        results: list[ExtractionResult | None] = [None] * len(sources)

        # Batch-extract file paths
        if file_paths:
            file_results = batch_extract_files_sync(file_paths, config=config, easyocr_kwargs=self.easyocr_kwargs)
            for idx, result in zip(file_indices, file_results, strict=True):
                results[idx] = result

        # Batch-extract byte streams
        if bytes_data:
            bytes_results = batch_extract_bytes_sync(
                bytes_data, bytes_mimes, config=config, easyocr_kwargs=self.easyocr_kwargs
            )
            for idx, result in zip(bytes_indices, bytes_results, strict=True):
                results[idx] = result

        return results

    def _build_extraction_metadata(self, result: ExtractionResult) -> dict[str, Any]:
        """
        Build metadata dict from an ``ExtractionResult``, flattening kreuzberg's
        metadata fields and enriching with top-level result attributes.

        None values are filtered out.
        """
        # Flatten kreuzberg document metadata (format-specific TypedDict)
        meta: dict[str, Any] = {
            k: v for k, v in result.metadata.items() if v is not None and k not in _METADATA_OVERLAP_KEYS
        }

        # Quality score
        if result.quality_score is not None:
            meta["quality_score"] = result.quality_score

        # Processing warnings
        if result.processing_warnings:
            meta["processing_warnings"] = _serialize_warnings(result.processing_warnings)

        # Detected languages
        if result.detected_languages:
            meta["detected_languages"] = list(result.detected_languages)

        # Extracted keywords
        if result.extracted_keywords:
            meta["extracted_keywords"] = _serialize_keywords(result.extracted_keywords)

        # Output format tracking
        meta["output_format"] = str(result.output_format)
        meta["result_format"] = str(result.result_format)
        meta["mime_type"] = result.mime_type
        try:
            extensions = get_extensions_for_mime(result.mime_type)
            if extensions:
                meta["file_extensions"] = extensions
        except RuntimeError:
            pass

        # Tables metadata
        if result.tables:
            meta["table_count"] = len(result.tables)
            meta["tables"] = _serialize_tables(result.tables)

        # Image extraction metadata (no binary data)
        if result.images:
            meta["image_count"] = len(result.images)
            meta["images"] = _serialize_images(result.images)

        # PDF annotations
        if result.annotations:
            meta["annotations"] = _serialize_annotations(result.annotations)

        return meta

    @staticmethod
    def _assemble_content(
        text: str, tables: list[ExtractedTable | dict[str, Any]] | None, output_format: str | None
    ) -> str:
        """
        Assemble document content, appending table markdown for plain-text output.

        When ``output_format`` is ``"markdown"`` or ``"html"``, kreuzberg already
        inlines tables into the content, so appending is skipped to avoid duplicates.

        Tables may be ``ExtractedTable`` objects or plain dicts (page context).
        """
        if not tables or output_format in ("markdown", "html"):
            return text

        table_blocks = [md for t in tables if (md := _get_table_markdown(t))]
        if not table_blocks:
            return text

        return text + "\n\n" + "\n\n".join(table_blocks)

    def _create_documents(
        self,
        result: ExtractionResult,
        bytestream: ByteStream,
        user_meta: dict[str, Any],
    ) -> list[Document]:
        """
        Create one or more ``Document`` objects from an ``ExtractionResult``.

        Output mode depends on what kreuzberg returns:
        - **Default:** one Document per source (unified content).
        - **Pages present:** one Document per page (when
          ``PageConfig(extract_pages=True)`` is in the config).
        - **Chunks present:** one Document per chunk (when
          ``ChunkingConfig`` is in the config).
        """
        base_meta = self._build_extraction_metadata(result)

        # Source metadata from bytestream
        source_meta = dict(bytestream.meta)
        if not self.store_full_path and "file_path" in source_meta:
            source_meta["file_path"] = Path(source_meta["file_path"]).name

        # Chunking mode — one Document per chunk
        if result.chunks:
            return self._create_chunked_documents(result, base_meta, source_meta, user_meta)

        # Per-page mode — one Document per page
        if result.pages:
            return self._create_per_page_documents(result, base_meta, source_meta, user_meta)

        # Default: unified mode — one Document per source
        content = self._assemble_content(result.content, result.tables, result.output_format)
        merged = {**base_meta, **source_meta, **copy.deepcopy(user_meta)}
        return [Document(content=content, meta=merged)]

    def _create_per_page_documents(
        self,
        result: ExtractionResult,
        base_meta: dict[str, Any],
        source_meta: dict[str, Any],
        user_meta: dict[str, Any],
    ) -> list[Document]:
        """Create one Document per page."""
        documents: list[Document] = []
        for page in result.pages:
            page_content = page.get("content", "")
            page_tables = page.get("tables", [])
            page_images = page.get("images", [])

            page_content = self._assemble_content(page_content, page_tables, result.output_format)

            page_meta: dict[str, Any] = {
                **copy.deepcopy(base_meta),
                **source_meta,
                "page_number": page.get("page_number"),
                "is_blank": page.get("is_blank", False),
            }

            if page_tables:
                page_meta["table_count"] = len(page_tables)
                page_meta["tables"] = _serialize_page_tables(page_tables)

            if page_images:
                page_meta["image_count"] = len(page_images)
                page_meta["images"] = _serialize_images(page_images)

            # Remove document-level table/image info to avoid confusion
            # (page-level info is more specific)
            if not page_tables:
                page_meta.pop("table_count", None)
                page_meta.pop("tables", None)

            page_meta.update(copy.deepcopy(user_meta))
            documents.append(Document(content=page_content, meta=page_meta))

        return documents

    def _create_chunked_documents(
        self,
        result: ExtractionResult,
        base_meta: dict[str, Any],
        source_meta: dict[str, Any],
        user_meta: dict[str, Any],
    ) -> list[Document]:
        """Create one Document per chunk, with optional embeddings."""
        documents: list[Document] = []
        total_chunks = len(result.chunks)

        for i, chunk in enumerate(result.chunks):
            chunk_meta = {
                **copy.deepcopy(base_meta),
                **source_meta,
                "chunk_index": i,
                "total_chunks": total_chunks,
                **copy.deepcopy(user_meta),
            }
            documents.append(
                Document(
                    content=chunk.content,
                    embedding=chunk.embedding,
                    meta=chunk_meta,
                )
            )

        return documents

    @staticmethod
    def _serialize_result(result: ExtractionResult) -> dict[str, Any]:
        """
        Serialize an ``ExtractionResult`` to a plain dict for the
        ``raw_extraction`` output socket.
        """
        raw: dict[str, Any] = {
            "content": result.content,
            "mime_type": result.mime_type,
            "output_format": str(result.output_format),
            "result_format": str(result.result_format),
            "metadata": dict(result.metadata),
        }

        if result.tables:
            raw["tables"] = _serialize_tables(result.tables)

        if result.quality_score is not None:
            raw["quality_score"] = result.quality_score

        if result.detected_languages:
            raw["detected_languages"] = list(result.detected_languages)

        if result.processing_warnings:
            raw["processing_warnings"] = _serialize_warnings(result.processing_warnings)

        if result.extracted_keywords:
            raw["extracted_keywords"] = _serialize_keywords(result.extracted_keywords)

        if result.annotations:
            raw["annotations"] = _serialize_annotations(result.annotations)

        if result.pages:
            raw["pages"] = [
                {
                    "page_number": p.get("page_number"),
                    "content": p.get("content"),
                    "is_blank": p.get("is_blank"),
                    "tables": _serialize_page_tables(p.get("tables", [])),
                }
                for p in result.pages
            ]

        if result.chunks:
            raw["chunks"] = [{"content": c.content, "metadata": c.metadata} for c in result.chunks]

        if result.images:
            raw["images"] = _serialize_images(result.images)

        return raw

    @staticmethod
    def _log_extraction_error(source: str | Path | ByteStream, error: Exception) -> None:
        """
        Log a structured extraction error using kreuzberg's error
        diagnostics when available.
        """
        try:
            error_code = get_last_error_code()
            details = get_error_details()
            code_name = error_code_name(error_code) if error_code is not None else "UNKNOWN"
            logger.warning(
                "Could not convert {source} to Document. Error code: {code} ({name}). Details: {details}. Skipping it.",
                source=source,
                code=error_code,
                name=code_name,
                details=details.get("message", str(error)),
            )
        except Exception as diag_err:
            logger.debug("Failed to get error diagnostics: {err}", err=diag_err)
            logger.warning(
                "Could not convert {source} to Document. Skipping it. Error: {error}",
                source=source,
                error=error,
            )

    @component.output_types(documents=list[Document], raw_extraction=list[dict[str, Any]])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document] | list[dict[str, Any]]]:
        """
        Convert files to Documents using Kreuzberg.

        :param sources:
            List of file paths, directory paths, or ByteStream objects to
            convert. Directory paths are expanded to their direct file children
            (non-recursive, sorted alphabetically).
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single
            dictionary. If it's a single dictionary, its content is added to
            the metadata of all produced Documents. If it's a list, the length
            of the list must match the number of sources, because the two
            lists will be zipped. If `sources` contains ByteStream objects,
            their `meta` will be added to the output Documents.

            **Note:** When directories are present in `sources`, `meta` must
            be a single dictionary (not a list), since the number of files in
            a directory is not known in advance.

        :returns:
            A dictionary with the following keys:

            - `documents`: A list of created Documents.
            - `raw_extraction`: A list of serialized kreuzberg
              ExtractionResult dicts, one per successfully processed source.
        """
        # Expand directories
        has_dirs = any(isinstance(s, (str, Path)) and Path(s).is_dir() for s in sources)
        if has_dirs and isinstance(meta, list):
            msg = (
                "When directories are present in 'sources', 'meta' must be a "
                "single dictionary, not a list, since the number of files in "
                "a directory is not known in advance."
            )
            raise ValueError(msg)

        expanded_sources = self._expand_sources(sources)
        meta_list = normalize_metadata(meta, sources_count=len(expanded_sources))

        config = self._build_config()

        if self.batch and len(expanded_sources) > 1:
            documents, raw_extractions = self._run_batch(expanded_sources, meta_list, config)
        else:
            documents, raw_extractions = self._run_sequential(expanded_sources, meta_list, config)

        return {"documents": documents, "raw_extraction": raw_extractions}

    def _run_sequential(
        self,
        sources: list[str | Path | ByteStream],
        meta_list: list[dict[str, Any]],
        config: ExtractionConfig,
    ) -> tuple[list[Document], list[dict[str, Any]]]:
        """Process sources one at a time.

        :returns: Tuple of (documents, raw_extractions).
        """
        documents: list[Document] = []
        raw_extractions: list[dict[str, Any]] = []

        for source, user_meta in zip(sources, meta_list, strict=True):
            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning(
                    "Could not read {source}. Skipping it. Error: {error}",
                    source=source,
                    error=e,
                )
                continue

            try:
                result = self._extract_single(source, config)
            except Exception as e:
                self._log_extraction_error(source, e)
                continue

            docs = self._create_documents(result, bytestream, user_meta)
            documents.extend(docs)
            raw_extractions.append(self._serialize_result(result))

        return documents, raw_extractions

    def _run_batch(
        self,
        sources: list[str | Path | ByteStream],
        meta_list: list[dict[str, Any]],
        config: ExtractionConfig,
    ) -> tuple[list[Document], list[dict[str, Any]]]:
        """Process sources using batch extraction.

        :returns: Tuple of (documents, raw_extractions).
        """
        documents: list[Document] = []
        raw_extractions: list[dict[str, Any]] = []

        # Pre-validate sources (get bytestreams for metadata)
        bytestreams: list[ByteStream | None] = []
        for source in sources:
            try:
                bytestreams.append(get_bytestream_from_source(source))
            except Exception as e:
                logger.warning(
                    "Could not read {source}. Skipping it. Error: {error}",
                    source=source,
                    error=e,
                )
                bytestreams.append(None)

        results = self._extract_batch(sources, config)

        for _source, result, bytestream, user_meta in zip(sources, results, bytestreams, meta_list, strict=True):
            if result is None or bytestream is None:
                continue

            docs = self._create_documents(result, bytestream, user_meta)
            documents.extend(docs)
            raw_extractions.append(self._serialize_result(result))

        return documents, raw_extractions

    @staticmethod
    def supported_extractors() -> list[str]:
        """
        List all document extractors registered in kreuzberg.

        :returns:
            List of extractor names.
        """
        return cast(list[str], list_document_extractors())

    @staticmethod
    def supported_ocr_backends() -> list[str]:
        """
        List all OCR backends registered in kreuzberg.

        :returns:
            List of OCR backend names.
        """
        return cast(list[str], list_ocr_backends())


def _get_table_markdown(table: ExtractedTable | dict[str, Any]) -> str | None:
    """Get markdown string from a table (``ExtractedTable`` object or dict)."""
    if isinstance(table, dict):
        return table.get("markdown") or None
    md = getattr(table, "markdown", None)
    return md or None


def _serialize_keywords(keywords: list[Any]) -> list[dict[str, Any]]:
    """Serialize ExtractedKeyword objects to plain dicts."""
    return [{"text": kw.text, "score": kw.score, "algorithm": kw.algorithm} for kw in keywords]


def _serialize_images(images: list[ExtractedImage]) -> list[dict[str, Any]]:
    """Serialize image metadata dicts, excluding binary data."""
    return [{k: v for k, v in img.items() if k != "data"} for img in images]


def _serialize_tables(tables: list[ExtractedTable]) -> list[dict[str, Any]]:
    """Serialize ExtractedTable objects to plain dicts."""
    return [
        {
            "cells": t.cells,
            "markdown": t.markdown,
            "page_number": t.page_number,
        }
        for t in tables
    ]


def _serialize_page_tables(tables: list[ExtractedTable | dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize tables from a page context (may be objects or dicts)."""
    serialized = []
    for t in tables:
        if isinstance(t, dict):
            serialized.append(
                {
                    "cells": t.get("cells"),
                    "markdown": t.get("markdown"),
                    "page_number": t.get("page_number"),
                }
            )
        else:
            serialized.append(
                {
                    "cells": t.cells,
                    "markdown": t.markdown,
                    "page_number": t.page_number,
                }
            )
    return serialized


def _serialize_warnings(warnings: list[Any]) -> list[dict[str, str]]:
    """Serialize processing warnings to plain dicts."""
    serialized = []
    for w in warnings:
        if isinstance(w, dict):
            serialized.append({"source": w.get("source", ""), "message": w.get("message", "")})
        else:
            serialized.append({"source": getattr(w, "source", ""), "message": getattr(w, "message", "")})
    return serialized


def _serialize_annotations(annotations: list[Any]) -> list[dict[str, Any]]:
    """Serialize PDF annotations to plain dicts."""
    serialized = []
    for ann in annotations:
        if isinstance(ann, dict):
            serialized.append(dict(ann))
        else:
            serialized.append(
                {
                    "type": getattr(ann, "annotation_type", None),
                    "content": getattr(ann, "content", None),
                    "page_number": getattr(ann, "page_number", None),
                }
            )
    return serialized


def _config_from_json_str(json_str: str) -> ExtractionConfig:
    """
    Load an ``ExtractionConfig`` from a JSON string via a temporary file.

    This is necessary because kreuzberg's PyO3 config objects don't expose a
    ``from_json()`` classmethod — only ``from_file()``.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
        f.write(json_str)
    try:
        return ExtractionConfig.from_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _copy_config(config: ExtractionConfig) -> ExtractionConfig:
    """
    Deep copy an ``ExtractionConfig`` by round-tripping through JSON.

    This is necessary because kreuzberg's PyO3 config objects don't support
    Python's ``copy.deepcopy()`` protocol.
    """
    return _config_from_json_str(config_to_json(config))
