# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import copy
from pathlib import Path
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream

from kreuzberg import (
    ExtractedTable,
    ExtractionConfig,
    ExtractionResult,
    LanguageDetectionConfig,
    batch_extract_bytes_sync,
    batch_extract_files_sync,
    classify_error,
    config_to_json,
    detect_mime_type,
    error_code_name,
    extract_bytes_sync,
    extract_file_sync,
    get_error_details,
    get_extensions_for_mime,
    get_last_error_code,
)

from .utils import (
    _config_from_json_str,
    _copy_config,
    _get_table_markdown,
    _is_batch_error,
    _serialize_annotations,
    _serialize_images,
    _serialize_keywords,
    _serialize_warnings,
)

logger = logging.getLogger(__name__)


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

    **Token reduction** can be configured via
    `ExtractionConfig(token_reduction=TokenReductionConfig(mode="moderate"))`
    to reduce output size for LLM consumption. Five levels are available:
    `"off"`, `"light"`, `"moderate"`, `"aggressive"`, `"maximum"`.
    The reduced text appears directly in `Document.content`.

    **Image preprocessing for OCR** can be tuned via
    `OcrConfig(tesseract_config=TesseractConfig(preprocessing=ImagePreprocessingConfig(...)))`
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
            See the [kreuzberg API reference](https://docs.kreuzberg.dev/reference/api-python/)
            for the full list of configuration options.
        :param config_path:
            Path to a kreuzberg configuration file (`.toml`, `.yaml`, or
            `.json`). Cannot be used together with `config`.
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
            See the [EasyOCR documentation](https://www.jaided.ai/easyocr/documentation/)
            for the full list of supported arguments.
        """
        if config is not None and config_path is not None:
            msg = "Cannot specify both 'config' and 'config_path'. Use one or the other."
            raise ValueError(msg)

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
        config_data = data["init_parameters"].get("config")
        if isinstance(config_data, str):
            data["init_parameters"]["config"] = _config_from_json_str(config_data)
        return default_from_dict(cls, data)

    def _build_config(self) -> ExtractionConfig:
        """
        Build the effective `ExtractionConfig`.

        Always returns a *fresh* config object — never mutates `self.config`.
        """
        if self.config is not None:
            config = _copy_config(self.config)
        elif self.config_path is not None:
            config = ExtractionConfig.from_file(self.config_path)
        else:
            config = ExtractionConfig()

        # Auto-enable language detection if not explicitly configured
        if config.language_detection is None:
            config.language_detection = LanguageDetectionConfig(enabled=True)

        return config

    def _extract_batch(
        self,
        sources: list[str | Path | ByteStream],
        config: ExtractionConfig,
    ) -> list[ExtractionResult | None]:
        """
        Extract content from multiple sources using kreuzberg's batch APIs.

        The result list is indexed to match `sources`.  Slots for
        file-based sources are filled by `batch_extract_files_sync`;
        slots for bytes-based sources by `batch_extract_bytes_sync`.
        A slot remains `None` only if neither batch API populated it
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
            self._collect_batch_results(
                file_indices,
                batch_extract_files_sync(file_paths, config=config, easyocr_kwargs=self.easyocr_kwargs),
                sources,
                results,
            )

        # Batch-extract byte streams
        if bytes_data:
            self._collect_batch_results(
                bytes_indices,
                batch_extract_bytes_sync(bytes_data, bytes_mimes, config=config, easyocr_kwargs=self.easyocr_kwargs),
                sources,
                results,
            )

        return results

    @staticmethod
    def _collect_batch_results(
        indices: list[int],
        batch_results: list[ExtractionResult],
        sources: list[str | Path | ByteStream],
        results: list[ExtractionResult | None],
    ) -> None:
        """Filter batch results, logging and skipping error entries."""
        for idx, result in zip(indices, batch_results, strict=True):
            if _is_batch_error(result):
                err_code = classify_error(result.content)
                logger.warning(
                    "Could not convert {source} to Document. Error code: {code} ({code_name}). "
                    "Details: {details}. Skipping it.",
                    source=sources[idx],
                    code=err_code,
                    code_name=error_code_name(err_code),
                    details=result.content,
                )
                continue
            results[idx] = result

    @staticmethod
    def _build_extraction_metadata(result: ExtractionResult) -> dict[str, Any]:
        """
        Build metadata dict from an `ExtractionResult`.

        Flattens kreuzberg's metadata fields and enriches with top-level result attributes.
        """
        meta: dict[str, Any] = {k: v for k, v in result.metadata.items() if v is not None}

        if result.output_format:
            meta["output_format"] = result.output_format
        if result.quality_score:
            meta["quality_score"] = result.quality_score
        if result.extracted_keywords:
            meta["keywords"] = _serialize_keywords(result.extracted_keywords)
        if result.processing_warnings:
            meta["processing_warnings"] = _serialize_warnings(result.processing_warnings)
        if result.detected_languages:
            meta["detected_languages"] = list(result.detected_languages)
        meta["result_format"] = str(result.result_format)
        meta["mime_type"] = result.mime_type
        try:
            extensions = get_extensions_for_mime(result.mime_type)
            if extensions:
                meta["file_extensions"] = extensions
        except RuntimeError:
            pass

        if result.images:
            meta["image_count"] = len(result.images)
            meta["images"] = _serialize_images(result.images)

        if result.annotations:
            meta["annotations"] = _serialize_annotations(result.annotations)

        return meta

    @staticmethod
    def _assemble_content(
        text: str, tables: list[ExtractedTable | dict[str, Any]] | None, output_format: str | None
    ) -> str:
        """
        Assemble document content, appending table markdown for plain-text output.

        When `output_format` is `"markdown"` or `"html"`, kreuzberg already
        inlines tables into the content, so appending is skipped to avoid duplicates.

        Tables may be `ExtractedTable` objects or plain dicts (page context).
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
        Create one or more `Document` objects from an `ExtractionResult`.

        Output mode depends on what kreuzberg returns:
        - **Default:** one Document per source (unified content).
        - **Pages present:** one Document per page (when
          `PageConfig(extract_pages=True)` is in the config).
        - **Chunks present:** one Document per chunk (when
          `ChunkingConfig` is in the config).
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

    @staticmethod
    def _create_per_page_documents(
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

            page_content = KreuzbergConverter._assemble_content(page_content, page_tables, result.output_format)

            page_meta: dict[str, Any] = {
                **copy.deepcopy(base_meta),
                **source_meta,
                "page_number": page.get("page_number"),
                "is_blank": page.get("is_blank", False),
            }

            if page_images:
                page_meta["image_count"] = len(page_images)
                page_meta["images"] = _serialize_images(page_images)

            page_meta.update(copy.deepcopy(user_meta))
            documents.append(Document(content=page_content, meta=page_meta))

        return documents

    @staticmethod
    def _create_chunked_documents(
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

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
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
            A dictionary with the following key:

            - `documents`: A list of created Documents.
        """
        # Expand directories
        has_dirs = any(isinstance(s, (str, Path)) and Path(s).is_dir() for s in sources)
        if has_dirs and isinstance(meta, list):
            msg = (
                "When directories are present in 'sources', 'meta' must be a "
                "single dictionary, not a list, since the number of files in "
                "a directory is not known in advance."
            )
            raise TypeError(msg)

        # Expand directory paths to their direct file children
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

        meta_list = normalize_metadata(meta, sources_count=len(expanded))

        config = self._build_config()

        # Resolve each source to a ByteStream
        bytestreams: list[ByteStream | None] = []
        for source in expanded:
            try:
                bytestreams.append(get_bytestream_from_source(source))
            except Exception as e:
                logger.warning(
                    "Could not read {source}. Skipping it. Error: {error}",
                    source=source,
                    error=e,
                )
                bytestreams.append(None)

        if self.batch and len(expanded) > 1:
            results = self._extract_batch(expanded, config)
        else:
            results = self._extract_sequential(expanded, bytestreams, config)

        documents: list[Document] = []
        for result, bytestream, user_meta in zip(results, bytestreams, meta_list, strict=True):
            if result is None or bytestream is None:
                continue
            documents.extend(self._create_documents(result, bytestream, user_meta))

        return {"documents": documents}

    def _extract_sequential(
        self,
        sources: list[str | Path | ByteStream],
        bytestreams: list[ByteStream | None],
        config: ExtractionConfig,
    ) -> list[ExtractionResult | None]:
        """Extract sources one at a time, returning results aligned with sources."""
        results: list[ExtractionResult | None] = []
        for source, bytestream in zip(sources, bytestreams, strict=True):
            if bytestream is None:
                results.append(None)
                continue
            try:
                if isinstance(source, ByteStream):
                    effective_mime = source.mime_type or detect_mime_type(source.data)
                    result = extract_bytes_sync(
                        source.data,
                        mime_type=effective_mime,
                        config=config,
                        easyocr_kwargs=self.easyocr_kwargs,
                    )
                else:
                    result = extract_file_sync(
                        source,
                        config=config,
                        easyocr_kwargs=self.easyocr_kwargs,
                    )
                results.append(result)
            except Exception as e:
                try:
                    err_code = get_last_error_code()
                    details = get_error_details()
                    code_name = error_code_name(err_code) if err_code is not None else "UNKNOWN"
                    logger.warning(
                        "Could not convert {source} to Document. Error code: {code} ({code_name}). "
                        "Details: {details}. Skipping it.",
                        source=source,
                        code=err_code,
                        code_name=code_name,
                        details=details.get("message", str(e)),
                    )
                except Exception as diag_err:
                    logger.debug("Failed to get error diagnostics: {err}", err=diag_err)
                    logger.warning(
                        "Could not convert {source} to Document. Skipping it. Error: {error}",
                        source=source,
                        error=e,
                    )
                results.append(None)
        return results
