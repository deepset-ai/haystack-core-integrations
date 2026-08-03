# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""OpenDataLoader PDF Haystack converter module."""

import json
import mimetypes
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import opendataloader_pdf
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)

# Internal separator used to recover per-page boundaries from text/markdown/html output.
_PAGE_SPLIT_SEPARATOR = "\n<<<ODL_PAGE_BREAK_%page-number%>>>\n"
_PAGE_SPLIT_PATTERN = re.compile(re.escape(_PAGE_SPLIT_SEPARATOR).replace(re.escape("%page-number%"), r"(\d+)"))

_EXT_MAP = {"json": "json", "text": "txt", "markdown": "md"}


def _resolve_source_name(source: ByteStream) -> str:
    """
    Resolve a display/staging name for a ByteStream source.

    Checks common metadata keys (`file_path`, `file_name`, `name`) and fall back to
    MIME-type extension guessing so that opendataloader_pdf can reliably detect the
    input format.
    """
    meta = source.meta or {}
    raw_name = meta.get("file_path") or meta.get("file_name") or meta.get("name")
    name = Path(raw_name).name if raw_name else "document"

    if not Path(name).suffix and source.mime_type:
        ext = mimetypes.guess_extension(source.mime_type)
        if ext:
            name = f"{name}{ext}"

    return name


def _is_valid_pdf_source(bytestream: ByteStream, name: str) -> bool:
    """
    Check if ByteStream is a valid PDF.
    """
    if bytestream.mime_type and "pdf" in bytestream.mime_type.lower():
        return True
    if name.lower().endswith(".pdf"):
        return True
    return bytestream.data[:5] == b"%PDF-"


@component
class OpenDataLoaderConverter:
    """
    Converts PDF files to Documents using OpenDataLoader PDF.

    This component wraps the `opendataloader-pdf` Python package, which runs a local
    Java engine (Java 11+ required, no external API calls) to extract structured
    content -- text, tables, headings, and reading order -- from PDF files.

    Accepts file paths and Haystack `ByteStream` objects. Sources that are not PDFs are
    skipped with a warning rather than raising -- see `run()` below for exactly how this
    is detected.

    ### Usage example

    ```python
    from haystack_integrations.components.converters.opendataloader_pdf import OpenDataLoaderConverter

    converter = OpenDataLoaderConverter(split_pages = True)
    results = converter.run(sources=["invoice.pdf", "haystack.pdf"])
    documents = results["documents"]
    ```
    """

    def __init__(
        self,
        *,
        output_format: str = "text",
        split_pages: bool = False,
        convert_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Create an OpenDataLoaderConverter component.

        :param output_format:
            Output format produced by the engine before being wrapped into Documents.
            One of "text", "markdown", "html", "json". Default: "text".
        :param split_pages:
            If True, yields one Document per PDF page.
            If False (default), yields a single Document per PDF file, with `page_count`
            stored in `meta`.
        :param convert_kwargs:
            Any parameters to pass to OpenDataLoader conversion; if not set,
            the engine's own defaults are used.
        """
        self.output_format = output_format.lower()
        if self.output_format not in _EXT_MAP:
            msg = f"Invalid format '{format}'. Valid options are: {', '.join(_EXT_MAP)}"
            raise ValueError(msg)

        self.split_pages = split_pages
        self.convert_kwargs = convert_kwargs if convert_kwargs is not None else {}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            output_format=self.output_format,
            split_pages=self.split_pages,
            convert_kwargs=self.convert_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenDataLoaderConverter":
        """
        Deserialize this component from a dictionary.

        :param data:
            Dictionary with keys `type` and `init_parameters`, as produced by `to_dict`.
        :returns:
            A new `OpenDataLoaderConverter` instance.
        """
        return default_from_dict(cls, data)

    @staticmethod
    def _is_java_available() -> bool:
        """Check if Java is installed and executable."""
        java = shutil.which("java")
        if java is None:
            return False

        try:
            subprocess.run(  # noqa: S603
                [java, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return False

        return True

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Run the OpenDataLoaderConvertor.

        :param sources:
            List of PDF file paths or `ByteStream` objects.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists
            will be zipped.
            If a source is a ByteStream, its own metadata is also merged into the output.
         :returns:
            A dictionary with key `"documents"` containing the output Haystack Documents.
        :raises ValueError: If `meta` is a list whose length does not match the number of sources.
        """
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))

        # Stage valid PDF sources into one temp input dir so the engine can batch them into a
        # single JVM invocation -- each opendataloader_pdf.convert() call spawns a new JVM process.
        input_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        documents: list[Document] = []

        try:
            staged: list[dict[str, Any]] = []

            for idx, (source, source_meta) in enumerate(zip(sources, meta_list, strict=True)):
                if isinstance(source, ByteStream):
                    bytestream = source
                    original_name = _resolve_source_name(source)
                else:
                    try:
                        bytestream = get_bytestream_from_source(source=source)
                    except Exception as e:
                        logger.warning(
                            "Could not read from {source}. Skipping it. Error: {error}", source=source, error=e
                        )
                        continue
                    original_name = Path(bytestream.meta.get("file_path") or f"source_{idx}.pdf").name

                if not _is_valid_pdf_source(bytestream, original_name):
                    logger.warning(
                        "{source} is not a PDF file (checked MIME type, extension, and file signature). "
                        "Skipping it -- OpenDataLoaderConverter only supports PDF input.",
                        source=source,
                    )
                    continue

                stem = f"{idx}_{Path(original_name).stem}"
                (Path(input_dir) / f"{stem}.pdf").write_bytes(bytestream.data)

                merged_meta = {**(bytestream.meta or {}), **source_meta, "file_path": original_name}
                staged.append({"stem": stem, "source_name": original_name, "meta": merged_meta})

            if not staged:
                return {"documents": documents}

            self._convert(input_dir, output_dir)

            ext = _EXT_MAP[self.output_format]
            for entry in staged:
                output_file = Path(output_dir) / f"{entry['stem']}.{ext}"
                if not output_file.exists():
                    logger.warning(
                        "OpenDataLoader PDF produced no output for {source}. Skipping it.",
                        source=entry["source_name"],
                    )
                    continue

                content = output_file.read_text(encoding="utf-8")
                if self.output_format == "json":
                    documents.extend(self._documents_from_json(content, entry["source_name"], entry["meta"]))
                else:
                    documents.extend(self._documents_from_text(content, entry["source_name"], entry["meta"]))

        finally:
            shutil.rmtree(input_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

        return {"documents": documents}

    def _get_page_separator(self) -> str | None:
        if self.split_pages:
            return _PAGE_SPLIT_SEPARATOR
        return None

    def _convert(self, input_dir: str, output_dir: str) -> None:
        if not self._is_java_available():
            msg = (
                "Java must be installed in the system ``PATH`` to use "
                "``OpenDataLoaderConverter``.\n"
                "Check the requirements here: "
                "https://opendataloader.org/docs/quick-start-python#requirements"
            )
            raise RuntimeError(msg)

        kwargs = dict(self.convert_kwargs)
        kwargs.pop("text_page_separator", None)
        kwargs.pop("markdown_page_separator", None)
        kwargs.pop("html_page_separator", None)

        page_sep = self._get_page_separator()

        opendataloader_pdf.convert(
            input_path=input_dir,
            output_dir=output_dir,
            format=[self.output_format],
            **kwargs,
            markdown_page_separator=page_sep,
            text_page_separator=page_sep,
            html_page_separator=page_sep,
        )

    def _documents_from_text(self, content: str, source_name: str, meta: dict[str, Any]) -> list[Document]:
        parts = _PAGE_SPLIT_PATTERN.split(content)

        if len(parts) == 1:
            pages = [(1, parts[0].strip())] if parts[0].strip() else []
        else:
            pages = [
                (int(parts[i]), parts[i + 1].strip())
                for i in range(1, len(parts), 2)
                if i + 1 < len(parts) and parts[i + 1].strip()
            ]

        if not pages:
            return []

        if self.split_pages:
            return [
                Document(
                    content=page_content,
                    meta={
                        **meta,
                        "source": source_name,
                        "format": self.output_format,
                        "page_number": page_num,
                    },
                )
                for page_num, page_content in pages
            ]

        return [
            Document(
                content="\n\n".join(page_content for _, page_content in pages),
                meta={
                    **meta,
                    "source": source_name,
                    "format": self.output_format,
                    "page_count": len(pages),
                },
            )
        ]

    def _documents_from_json(self, content: str, source_name: str, meta: dict[str, Any]) -> list[Document]:
        data = json.loads(content)

        pages: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for element in data.get("kids", []):
            page_num = element.get("page number", 1)
            pages[page_num].append(element)

        if not pages:
            return []

        if self.split_pages:
            return [
                Document(
                    content=json.dumps(
                        {"page number": page_num, "kids": pages[page_num]},
                        ensure_ascii=False,
                    ),
                    meta={
                        **meta,
                        "source": source_name,
                        "format": self.output_format,
                        "page_number": page_num,
                    },
                )
                for page_num in sorted(pages)
            ]

        return [
            Document(
                content=content,
                meta={
                    **meta,
                    "source": source_name,
                    "format": self.output_format,
                    "page_count": len(pages),
                },
            )
        ]
