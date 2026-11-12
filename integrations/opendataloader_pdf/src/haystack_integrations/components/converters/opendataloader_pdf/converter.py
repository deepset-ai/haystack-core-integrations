# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
import shutil
import subprocess
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal
from uuid import uuid4

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import normalize_metadata
from haystack.dataclasses import ByteStream

import opendataloader_pdf  # type: ignore[import-untyped]

OutputFormat = Literal["markdown", "text", "json", "html"]

logger = logging.getLogger(__name__)

_EXTENSIONS = {"markdown": "md", "text": "txt", "html": "html", "json": "json"}

# OpenDataLoader root JSON fields mapped to Document.meta keys
_ROOT_FIELDS = {
    "title": "title",
    "author": "author",
    "creation date": "creation_date",
    "modification date": "modification_date",
    "number of pages": "page_count",
}

# Keys under which OpenDataLoader nests child elements: lists use "list items", tables "rows" -> "cells"
_CHILD_KEYS = ("kids", "list items", "rows", "cells")

# PDF date string, e.g. "D:20160319061844Z00'00'" or "D:20240101120000+02'00'" (PDF 32000-1, section 7.9.4)
_PDF_DATE = re.compile(r"^(?:D:)?(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?(?:([Z+-])(?:(\d{2})'?(\d{2})?'?)?)?$")

# OpenDataLoader writes the page separator before every page, including the first one
_PAGE_SEPARATOR = "\f"


@component
class OpenDataLoaderConverter:
    """
    OpenDataLoader PDF converter component.

    The component accepts PDF file paths and Haystack ByteStream objects, runs OpenDataLoader PDF extraction, and
    returns Haystack Document objects. It can also extract images to a persistent directory and return one image
    Document per extracted file.

    With `output_format="json"`, the Document content is Markdown with pages separated by form feeds (`\\f`), so
    `DocumentSplitter` can compute page numbers, and the PDF metadata from the OpenDataLoader JSON output is added to
    `Document.meta`. Set `raw_json=True` to get the raw JSON string as content instead.

    Java 11 or newer must be installed and available on PATH.

    ### Usage example
    ```python
    from haystack_integrations.components.converters.opendataloader_pdf import OpenDataLoaderConverter

    converter = OpenDataLoaderConverter(
        output_format="markdown", extract_images=True, image_output_dir="extracted_images"
    )
    result = converter.run(sources=["report.pdf"], meta={"source": "annual-report"})

    documents = result["documents"]
    image_documents = result["image_documents"]
    print(documents[0].content)
    print(documents[0].meta["file_path"])
    ```
    """

    def __init__(
        self,
        *,
        output_format: OutputFormat = "markdown",
        convert_kwargs: dict[str, Any] | None = None,
        extract_images: bool = False,
        image_output_dir: str | Path | None = None,
        skip_failed_pdfs: bool = False,
        raw_json: bool = False,
        include_layout: bool = False,
    ) -> None:
        """
        Initialize the OpenDataLoader converter.

        :param output_format: Format OpenDataLoader should produce. With `"json"`, the Document content is Markdown
            with pages separated by `\\f`, and `Document.meta` gets the PDF's `title`, `author`, `creation_date`,
            `modification_date` (ISO 8601 when the PDF date can be parsed, otherwise as found in the PDF), and
            `page_count`. Fields missing from the PDF are omitted. User-provided metadata takes precedence on key
            conflicts.
        :param convert_kwargs: Additional arguments passed to `opendataloader_pdf.convert`. See the
            [OpenDataLoader PDF Python options](https://opendataloader.org/docs/quick-start-python#convert-options).
            The `image_output` and `image_dir` arguments are managed by this component; supplied values are ignored.
            With `output_format="json"`, passing `markdown_page_separator` replaces the default `\\f` page separator.
        :param extract_images: Whether to extract images and return them through the `image_documents` output.
        :param image_output_dir: Persistent root directory for extracted image files. Each `run()` stores its images
            in a unique subdirectory of this directory. Required when `extract_images` is `True`.
        :param skip_failed_pdfs: Bisect failed batches and skip failing PDFs with a warning. Java and source
            preparation errors still propagate.
        :param raw_json: If `True` with `output_format="json"`, store the raw OpenDataLoader JSON string in
            `Document.content` and add no PDF metadata. Has no effect with other output formats.
        :param include_layout: If `True` with `output_format="json"`, add a `layout` list to `Document.meta` with one
            entry per positioned element in reading order: `type`, `page_number`, `bounding_box`, and
            `heading_level` for headings. The list covers the whole PDF and is copied to every chunk when the Document
            is split, so it can be large for long PDFs. Has no effect with other output formats or with `raw_json`.
        :raises ValueError: If image extraction is enabled without an output directory.
        """
        conversion_options = convert_kwargs.copy() if convert_kwargs else {}
        if managed_options := {"image_dir", "image_output"}.intersection(conversion_options):
            message = (
                f"Ignoring component-managed image options in convert_kwargs: {', '.join(sorted(managed_options))}"
            )
            logger.warning(message)
            for option in managed_options:
                conversion_options.pop(option)
        if extract_images and image_output_dir is None:
            message = "image_output_dir is required when extract_images is enabled"
            raise ValueError(message)

        self.output_format = output_format
        self.convert_kwargs = conversion_options
        self.extract_images = extract_images
        self.image_output_dir = Path(image_output_dir) if image_output_dir is not None else None
        self.skip_failed_pdfs = skip_failed_pdfs
        self.raw_json = raw_json
        self.include_layout = include_layout

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component.

        :returns:
            Dictionary representation of the converter.
        """
        return default_to_dict(
            self,
            output_format=self.output_format,
            convert_kwargs=self.convert_kwargs,
            extract_images=self.extract_images,
            image_output_dir=str(self.image_output_dir) if self.image_output_dir is not None else None,
            skip_failed_pdfs=self.skip_failed_pdfs,
            raw_json=self.raw_json,
            include_layout=self.include_layout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenDataLoaderConverter":
        """
        Deserialize the component.

        :param data: Serialized component dictionary.
        :returns:
            Reconstructed OpenDataLoaderConverter.
        """
        return default_from_dict(cls=cls, data=data)

    def _prepare_sources(
        self, sources: list[str | Path | ByteStream], metadata: list[dict[str, Any]], input_dir: Path
    ) -> list[tuple[Path, dict[str, Any]]]:
        """
        Stage PDF sources under unique names and preserve their metadata.

        :param sources: PDF file paths or Haystack ByteStream objects.
        :param metadata: User-provided metadata aligned with `sources`.
        :param input_dir: Temporary directory where sources are staged.
        :raises ValueError: If a ByteStream has a non-PDF MIME type, or a file path does not have a `.pdf` extension.
        :returns:
            Staged PDF paths paired with their merged metadata.
        """
        prepared_sources: list[tuple[Path, dict[str, Any]]] = []
        for index, (source, user_meta) in enumerate(zip(sources, metadata, strict=True)):
            staged_path = input_dir / f"document_{index}.pdf"
            if isinstance(source, ByteStream):
                if source.mime_type:
                    mime_type = source.mime_type.split(sep=";", maxsplit=1)[0].strip().lower()
                    if mime_type != "application/pdf":
                        message = (
                            "OpenDataLoaderConverter only supports PDF ByteStreams. "
                            f"Received MIME type: {source.mime_type}"
                        )
                        raise ValueError(message)
                staged_path.write_bytes(source.data)
                source_meta = dict(source.meta)
                if not source_meta.get("file_path"):
                    source_meta["file_path"] = f"document_{index}.pdf"
            else:
                source_path = Path(source)
                if source_path.suffix.lower() != ".pdf":
                    message = f"OpenDataLoaderConverter only supports PDFs: {source_path}"
                    raise ValueError(message)
                shutil.copyfile(src=source_path, dst=staged_path)
                source_meta = {"file_path": source_path.name}
            prepared_sources.append((staged_path, {**source_meta, **user_meta}))
        return prepared_sources

    @staticmethod
    def _check_java_available() -> None:
        """
        Check whether a usable Java runtime is available.

        :raises RuntimeError: If `java` is not available on PATH or cannot be executed.
        """
        message = (
            "Java 11 or newer is required to use OpenDataLoaderConverter. "
            "Install Java and ensure `java` is available on PATH."
        )
        java = shutil.which("java")
        if java is None:
            raise RuntimeError(message)

        try:
            subprocess.run(  # noqa: S603
                args=[java, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(message) from exc

    @staticmethod
    def _read_output(output_dir: Path, pdf_path: Path, output_format: OutputFormat) -> str:
        """
        Read an OpenDataLoader output file.

        :param output_dir: Directory containing converted files.
        :param pdf_path: Staged PDF path used to determine the output filename.
        :param output_format: Format of the output file to read.
        :raises RuntimeError: If OpenDataLoader did not create the expected output file.
        :returns:
            Extracted document content.
        """
        output_file = output_dir / f"{pdf_path.stem}.{_EXTENSIONS[output_format]}"
        if not output_file.exists():
            message = f"OpenDataLoader did not create expected file: {output_file}"
            raise RuntimeError(message)
        return output_file.read_text(encoding="utf-8")

    @staticmethod
    def _parse_pdf_date(value: str) -> str:
        """
        Convert a PDF date string to ISO 8601.

        :param value: Date as found in the PDF, e.g. `D:20160319061844Z00'00'`.
        :returns:
            The ISO 8601 date, or `value` unchanged if it is not a valid PDF date.
        """
        match = _PDF_DATE.match(value.strip())
        if match is None:
            logger.warning("Could not parse PDF date {value}; keeping it unchanged", value=value)
            return value
        year, month, day, hour, minute, second, sign, tz_hour, tz_minute = match.groups()
        tzinfo = None
        if sign == "Z":
            tzinfo = UTC
        elif sign is not None:
            offset = timedelta(hours=int(tz_hour or 0), minutes=int(tz_minute or 0))
            tzinfo = timezone(offset if sign == "+" else -offset)
        try:
            parsed = datetime(
                int(year),
                int(month or 1),
                int(day or 1),
                int(hour or 0),
                int(minute or 0),
                int(second or 0),
                tzinfo=tzinfo,
            )
        except ValueError:
            logger.warning("Could not parse PDF date {value}; keeping it unchanged", value=value)
            return value
        return parsed.isoformat()

    def _meta_from_json(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Map OpenDataLoader JSON fields, and optionally the element layout, into Haystack Document metadata.

        :param data: Parsed OpenDataLoader JSON document.
        :returns:
            Root PDF fields, omitting null ones, plus a `layout` list when `include_layout` is enabled.
        """
        meta: dict[str, Any] = {}
        for source, dest in _ROOT_FIELDS.items():
            value = data.get(source)
            if value is None:
                continue
            if dest in ("creation_date", "modification_date") and isinstance(value, str):
                value = self._parse_pdf_date(value)
            meta[dest] = value

        if not self.include_layout:
            return meta

        layout: list[dict[str, Any]] = []
        # Depth-first so entries follow reading order; children are pushed reversed to pop in document order
        stack: list[Any] = list(reversed(data.get("kids") or []))
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue
            page = node.get("page number")
            bbox = node.get("bounding box")
            if page is not None and bbox is not None:
                item: dict[str, Any] = {"type": node.get("type"), "page_number": page, "bounding_box": bbox}
                if "heading level" in node:
                    item["heading_level"] = node["heading level"]
                layout.append(item)
            for key in reversed(_CHILD_KEYS):
                stack.extend(reversed(node.get(key) or []))
        meta["layout"] = layout
        return meta

    def _convert_batch(
        self,
        sources: list[tuple[Path, dict[str, Any]]],
        output_dir: Path,
        image_dir: Path | None,
    ) -> list[Document]:
        """
        Convert a non-empty batch, bisecting failures into isolated output directories when enabled.

        :param sources: Non-empty list of staged PDF paths paired with their merged metadata.
        :param output_dir: Temporary output directory for this batch. Partial outputs are removed before retrying.
        :param image_dir: Persistent image directory for this batch, or `None` to disable image extraction.
            Images from failed attempts are removed.
        :returns:
            Converted Documents in source order, omitting failed PDFs when `skip_failed_pdfs` is enabled.
            Returns an empty list when a single PDF fails and is skipped.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        conversion_kwargs = {"image_output": "off", **self.convert_kwargs}
        if image_dir is not None:
            image_dir.mkdir(parents=True, exist_ok=True)
            conversion_kwargs.update(image_output="external", image_dir=str(image_dir))

        parse_json = self.output_format == "json" and not self.raw_json
        convert_format: str | list[str] = self.output_format
        if parse_json:
            convert_format = ["json", "markdown"]
            conversion_kwargs.setdefault("markdown_page_separator", _PAGE_SEPARATOR)

        documents: list[Document] = []
        try:
            opendataloader_pdf.convert(
                input_path=[str(pdf_path) for pdf_path, _ in sources],
                output_dir=str(output_dir),
                format=convert_format,
                **conversion_kwargs,
            )
            for pdf_path, document_meta in sources:
                extra_meta: dict[str, Any] = {}
                if parse_json:
                    raw = self._read_output(output_dir=output_dir, pdf_path=pdf_path, output_format="json")
                    extra_meta = self._meta_from_json(json.loads(raw))
                    content = self._read_output(output_dir=output_dir, pdf_path=pdf_path, output_format="markdown")
                    # Drop the separator OpenDataLoader puts before page 1; otherwise DocumentSplitter numbers every
                    # page one too high
                    if conversion_kwargs["markdown_page_separator"] == _PAGE_SEPARATOR:
                        content = content.removeprefix(_PAGE_SEPARATOR)
                else:
                    content = self._read_output(
                        output_dir=output_dir, pdf_path=pdf_path, output_format=self.output_format
                    )
                documents.append(
                    Document(content=content, meta={**extra_meta, **document_meta, "output_format": self.output_format})
                )
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            if not self.skip_failed_pdfs:
                raise
            # Failed attempts may leave seemingly valid text and images. Never reuse their outputs.
            shutil.rmtree(output_dir)
            if image_dir is not None:
                shutil.rmtree(image_dir)
            if len(sources) == 1:
                pdf_path, document_meta = sources[0]
                logger.warning(
                    "Skipping PDF {file_path} after conversion failure: {error}",
                    file_path=document_meta.get("file_path", str(pdf_path)),
                    error=exc,
                )
                return []

            logger.warning(
                "Batch of {count} PDFs failed; splitting into smaller batches: {error}", count=len(sources), error=exc
            )
            midpoint = len(sources) // 2
            documents = []
            for name, batch in (("left", sources[:midpoint]), ("right", sources[midpoint:])):
                documents.extend(
                    self._convert_batch(
                        sources=batch,
                        output_dir=output_dir / name,
                        image_dir=image_dir / name if image_dir is not None else None,
                    )
                )
        return documents

    @component.output_types(documents=list[Document], image_documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Convert PDF sources into Haystack Documents.

        :param sources: PDF file paths or Haystack ByteStream objects.
        :param meta: Optional metadata attached to the generated Documents. A single dictionary is applied to every
            source. A list must contain one dictionary per source. ByteStream metadata is also preserved.
        :returns:
            Dictionary containing the converted text Documents and image Documents. Each image Document has the
            persistent extracted image path in its `file_path` metadata field. With `skip_failed_pdfs=True`, PDFs
            that fail individual conversion are omitted from both outputs.
        """
        if not sources:
            return {"documents": [], "image_documents": []}

        self._check_java_available()
        metadata = normalize_metadata(meta=meta, sources_count=len(sources))

        documents: list[Document] = []
        image_documents: list[Document] = []
        image_dir: Path | None = None
        if self.image_output_dir is not None:
            self.image_output_dir.mkdir(parents=True, exist_ok=True)
            image_dir = self.image_output_dir / uuid4().hex
            image_dir.mkdir()

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            input_dir.mkdir()
            output_dir.mkdir()
            prepared_sources = self._prepare_sources(sources=sources, metadata=metadata, input_dir=input_dir)

            documents = self._convert_batch(sources=prepared_sources, output_dir=output_dir, image_dir=image_dir)

        if image_dir is not None:
            image_documents = [
                Document(meta={"file_path": str(path)}) for path in image_dir.rglob("*") if path.is_file()
            ]

        return {"documents": documents, "image_documents": image_documents}
