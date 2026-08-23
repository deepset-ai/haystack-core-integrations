# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
import re
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, NamedTuple, TypedDict
from urllib.parse import urlparse

from gotenberg_client import AsyncGotenbergClient, SyncGotenbergClient
from gotenberg_client.responses import SingleFileResponse, ZipFileResponse
from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ByteStream
from typing_extensions import Self

ConversionType = Literal["libreoffice", "html", "markdown", "url"]
Source = str | Path | ByteStream

MIME_TYPE_EXTENSIONS = {
    "application/msword": ".doc",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.oasis.opendocument.presentation": ".odp",
    "application/vnd.oasis.opendocument.spreadsheet": ".ods",
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


class GotenbergFileConverterOutput(TypedDict):
    """The output returned by `GotenbergFileConverter`."""

    output: list[ByteStream]


class _PreparedSource(NamedTuple):
    kind: ConversionType
    value: str | Path
    markdown_path: Path | None = None


def _route(client: Any, source: _PreparedSource, resources: list[Path]) -> Any:
    """Build an SDK route from a validated, normalized source."""
    if source.kind == "libreoffice":
        return client.libre_office.to_pdf().convert(Path(source.value))
    if source.kind == "html":
        route = client.chromium.html_to_pdf().string_index(str(source.value))
    elif source.kind == "markdown":
        route = client.chromium.markdown_to_pdf().string_index(str(source.value))
        route.markdown_file(source.markdown_path)
    else:
        return client.chromium.url_to_pdf().url(str(source.value))
    if resources:
        route.resources(resources)
    return route


def _resources(resources: list[Path] | None) -> list[Path]:
    paths = resources or []
    names: set[str] = set()
    for resource in paths:
        if not isinstance(resource, Path):
            msg = f"Unsupported resource type: {type(resource).__name__}; expected Path"
            raise TypeError(msg)
        if not resource.is_file():
            msg = f"Resource does not exist or is not a file: {resource}"
            raise FileNotFoundError(msg)
        if resource.name in names:
            msg = f"Resource filenames must be unique: {resource.name!r}"
            raise ValueError(msg)
        if resource.name == "index.html":
            msg = "A resource cannot be named 'index.html'; that name is reserved by Gotenberg"
            raise ValueError(msg)
        names.add(resource.name)
    return paths


def _text(source: Source, kind: str) -> tuple[str, str | None]:
    if isinstance(source, str):
        return source, None
    if isinstance(source, Path):
        if not source.is_file():
            msg = f"{kind} source file does not exist or is not a file: {source}"
            raise FileNotFoundError(msg)
        data = source.read_bytes()
        filename: str | None = source.name
    elif isinstance(source, ByteStream):
        data = source.data
        file_path = source.meta.get("file_path")
        filename = Path(str(file_path)).name if file_path else None
    else:
        msg = f"Unsupported {kind} source type: {type(source).__name__}; expected literal str, Path, or ByteStream"
        raise TypeError(msg)
    try:
        return data.decode("utf-8"), filename
    except UnicodeDecodeError as error:
        msg = f"{kind} sources must contain UTF-8 text"
        raise ValueError(msg) from error


@component
class GotenbergFileConverter:
    """
    Convert files, UTF-8 HTML or Markdown, and web URLs to ordered PDFs with Gotenberg.

    Resources are supported only for HTML and Markdown conversions.
    """

    def __init__(self, url: str = "http://localhost:3000", timeout: float = 30.0) -> None:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            msg = "url must be a valid HTTP(S) Gotenberg service URL"
            raise ValueError(msg)
        if timeout <= 0:
            msg = "timeout must be greater than zero"
            raise ValueError(msg)
        self.url = url.rstrip("/")
        self.timeout = timeout

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(self, url=self.url, timeout=self.timeout)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize this component from a dictionary."""
        return default_from_dict(cls, data)

    @staticmethod
    def _pdf(response: SingleFileResponse | ZipFileResponse) -> ByteStream:
        if response.is_zip:
            msg = "Gotenberg unexpectedly returned a ZIP archive instead of one PDF"
            raise RuntimeError(msg)
        return ByteStream(data=bytes(response.content), mime_type="application/pdf")

    @staticmethod
    def _libreoffice_source(source: Source, directory: Path, index: int) -> Path:
        if isinstance(source, ByteStream):
            file_path = source.meta.get("file_path")
            filename = Path(str(file_path)).name if file_path else ""
            suffix = Path(filename).suffix
            if not suffix and source.mime_type:
                suffix = (
                    MIME_TYPE_EXTENSIONS.get(source.mime_type)
                    or mimetypes.guess_extension(source.mime_type, strict=False)
                    or ""
                )
            if not suffix:
                msg = (
                    "LibreOffice ByteStream sources require either a filename with an extension in "
                    'meta["file_path"] or a recognized mime_type'
                )
                raise ValueError(msg)
            stem = Path(filename).stem if filename else "document"
            path = directory / f"{index}-{stem}{suffix}"
            path.write_bytes(source.data)
            return path
        if not isinstance(source, (str, Path)):
            msg = f"Unsupported LibreOffice source type: {type(source).__name__}; expected str, Path, or ByteStream"
            raise TypeError(msg)
        path = Path(source)
        if not path.is_file():
            msg = f"LibreOffice source file does not exist or is not a file: {path}"
            raise FileNotFoundError(msg)
        if not path.suffix:
            msg = f"LibreOffice source filename must include a file extension: {path.name!r}"
            raise ValueError(msg)
        return path

    @staticmethod
    def _markdown_filename(filename: str | None) -> str:
        name = filename or "document.md"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(name).name)
        if not safe_name or safe_name in {".", ".."}:
            safe_name = "document.md"
        if not Path(safe_name).suffix:
            safe_name += ".md"
        if safe_name == "index.html":
            safe_name = "document.md"
        return safe_name

    @contextmanager
    def _prepare(
        self, sources: list[Source], conversion_type: ConversionType, resources: list[Path] | None
    ) -> Generator[tuple[list[_PreparedSource], list[Path]], None, None]:
        if conversion_type not in {"libreoffice", "html", "markdown", "url"}:
            msg = f"Unsupported conversion_type: {conversion_type!r}; expected libreoffice, html, markdown, or url"
            raise ValueError(msg)
        if not sources:
            msg = f"sources must contain at least one source for {conversion_type} conversion"
            raise ValueError(msg)
        if conversion_type in {"libreoffice", "url"} and resources is not None:
            msg = f"resources are not supported for {conversion_type} conversion"
            raise ValueError(msg)
        resource_paths = _resources(resources) if conversion_type in {"html", "markdown"} else []
        with ExitStack() as stack:
            prepared: list[_PreparedSource]
            if conversion_type == "libreoffice":
                directory = Path(stack.enter_context(TemporaryDirectory(prefix="haystack-gotenberg-")))
                prepared = [
                    _PreparedSource("libreoffice", self._libreoffice_source(source, directory, index))
                    for index, source in enumerate(sources)
                ]
            elif conversion_type == "html":
                prepared = [_PreparedSource("html", _text(source, "HTML")[0]) for source in sources]
            elif conversion_type == "markdown":
                texts = [_text(source, "Markdown") for source in sources]
                filenames = [self._markdown_filename(filename) for _, filename in texts]
                resource_names = {resource.name for resource in resource_paths}
                for filename in filenames:
                    if filename in resource_names:
                        msg = f"A resource conflicts with the Markdown source filename: {filename!r}"
                        raise ValueError(msg)
                prepared = []
                for (text, _), filename in zip(texts, filenames, strict=True):
                    directory = Path(stack.enter_context(TemporaryDirectory(prefix="haystack-gotenberg-")))
                    path = directory / filename
                    path.write_text(text, encoding="utf-8")
                    head = '<!doctype html><html><head><meta charset="utf-8"></head>'
                    template = f'{head}<body>{{{{ toHTML "{filename}" }}}}</body></html>'
                    prepared.append(_PreparedSource("markdown", template, path))
            else:
                prepared = []
                for source in sources:
                    if not isinstance(source, str):
                        msg = f"Unsupported URL source type: {type(source).__name__}; expected an HTTP(S) string"
                        raise TypeError(msg)
                    parsed = urlparse(source)
                    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                        msg = f"URL source must be a valid HTTP(S) URL: {source!r}"
                        raise ValueError(msg)
                    prepared.append(_PreparedSource("url", source))
            yield prepared, resource_paths

    @component.output_types(output=list[ByteStream])
    def run(
        self, sources: list[Source], conversion_type: ConversionType, resources: list[Path] | None = None
    ) -> GotenbergFileConverterOutput:
        """
        Convert sources to PDF using the selected Gotenberg route.

        :param sources: Sources interpreted according to `conversion_type`.
        :param conversion_type: Route to use: `libreoffice`, `html`, `markdown`, or `url`.
        :param resources: Optional resource files for HTML or Markdown conversion only.
        :returns: One PDF byte stream per source, in input order.
        """
        with self._prepare(sources, conversion_type, resources) as (prepared, resource_paths):
            with SyncGotenbergClient(self.url, timeout=self.timeout, backend="httpx") as client:
                output = [self._pdf(_route(client, source, resource_paths).run()) for source in prepared]
        return {"output": output}

    @component.output_types(output=list[ByteStream])
    async def run_async(
        self, sources: list[Source], conversion_type: ConversionType, resources: list[Path] | None = None
    ) -> GotenbergFileConverterOutput:
        """
        Asynchronously convert sources using the selected Gotenberg route.

        :param sources: Sources interpreted according to `conversion_type`.
        :param conversion_type: Route to use: `libreoffice`, `html`, `markdown`, or `url`.
        :param resources: Optional resource files for HTML or Markdown conversion only.
        :returns: One PDF byte stream per source, in input order.
        """
        with self._prepare(sources, conversion_type, resources) as (prepared, resource_paths):
            async with AsyncGotenbergClient(self.url, timeout=self.timeout, backend="httpx") as client:
                output = [self._pdf(await _route(client, source, resource_paths).run()) for source in prepared]
        return {"output": output}
