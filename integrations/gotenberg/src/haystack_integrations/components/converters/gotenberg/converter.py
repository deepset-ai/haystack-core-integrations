# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
import re
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, overload
from urllib.parse import urlparse

import httpx
from gotenberg_client import AsyncGotenbergClient, SyncGotenbergClient
from gotenberg_client._base import AsyncBaseRoute, SyncBaseRoute
from gotenberg_client.responses import SingleFileResponse, ZipFileResponse
from haystack import component, default_from_dict, default_to_dict
from haystack.components.converters.utils import normalize_metadata
from haystack.dataclasses import ByteStream
from typing_extensions import Self

_Route = Literal["libreoffice", "html", "markdown", "url"]
_LIBREOFFICE_EXTENSIONS = frozenset(
    {
        ".123",
        ".602",
        ".abw",
        ".bib",
        ".bmp",
        ".cdr",
        ".cgm",
        ".cmx",
        ".csv",
        ".cwk",
        ".dbf",
        ".dif",
        ".doc",
        ".docm",
        ".docx",
        ".dot",
        ".dotm",
        ".dotx",
        ".dxf",
        ".emf",
        ".eps",
        ".epub",
        ".fodg",
        ".fodp",
        ".fods",
        ".fodt",
        ".fopd",
        ".gif",
        ".htm",
        ".html",
        ".hwp",
        ".jpeg",
        ".jpg",
        ".key",
        ".ltx",
        ".lwp",
        ".mcw",
        ".met",
        ".mml",
        ".mw",
        ".numbers",
        ".odd",
        ".odg",
        ".odm",
        ".odp",
        ".ods",
        ".odt",
        ".otg",
        ".oth",
        ".otp",
        ".ots",
        ".ott",
        ".pages",
        ".pbm",
        ".pcd",
        ".pct",
        ".pcx",
        ".pdb",
        ".pdf",
        ".pgm",
        ".png",
        ".pot",
        ".potm",
        ".potx",
        ".ppm",
        ".pps",
        ".ppsm",
        ".ppsx",
        ".ppt",
        ".pptm",
        ".pptx",
        ".psd",
        ".psw",
        ".pub",
        ".pwp",
        ".pxl",
        ".ras",
        ".rtf",
        ".sda",
        ".sdc",
        ".sdd",
        ".sdp",
        ".sdw",
        ".sgl",
        ".slk",
        ".smf",
        ".stc",
        ".std",
        ".sti",
        ".stw",
        ".svg",
        ".svm",
        ".swf",
        ".sxc",
        ".sxd",
        ".sxg",
        ".sxi",
        ".sxm",
        ".sxw",
        ".tga",
        ".tif",
        ".tiff",
        ".txt",
        ".uof",
        ".uop",
        ".uos",
        ".uot",
        ".vdx",
        ".vor",
        ".vsd",
        ".vsdm",
        ".vsdx",
        ".wb2",
        ".wk1",
        ".wks",
        ".wmf",
        ".wpd",
        ".wpg",
        ".wps",
        ".xbm",
        ".xhtml",
        ".xls",
        ".xlsb",
        ".xlsm",
        ".xlsx",
        ".xlt",
        ".xltm",
        ".xltx",
        ".xlw",
        ".xml",
        ".xpm",
        ".zabw",
    }
)
MIME_TYPE_EXTENSIONS = {
    "application/epub+zip": ".epub",
    "application/msword": ".doc",
    "application/pdf": ".pdf",
    "application/postscript": ".eps",
    "application/rtf": ".rtf",
    "application/vnd.apple.keynote": ".key",
    "application/vnd.apple.numbers": ".numbers",
    "application/vnd.apple.pages": ".pages",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.ms-excel.sheet.binary.macroenabled.12": ".xlsb",
    "application/vnd.ms-excel.sheet.macroenabled.12": ".xlsm",
    "application/vnd.ms-excel.template.macroenabled.12": ".xltm",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.ms-powerpoint.presentation.macroenabled.12": ".pptm",
    "application/vnd.ms-powerpoint.slideshow.macroenabled.12": ".ppsm",
    "application/vnd.ms-powerpoint.template.macroenabled.12": ".potm",
    "application/vnd.ms-word.document.macroenabled.12": ".docm",
    "application/vnd.ms-word.template.macroenabled.12": ".dotm",
    "application/vnd.oasis.opendocument.graphics": ".odg",
    "application/vnd.oasis.opendocument.graphics-template": ".otg",
    "application/vnd.oasis.opendocument.presentation": ".odp",
    "application/vnd.oasis.opendocument.presentation-template": ".otp",
    "application/vnd.oasis.opendocument.spreadsheet": ".ods",
    "application/vnd.oasis.opendocument.spreadsheet-template": ".ots",
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/vnd.oasis.opendocument.text-master": ".odm",
    "application/vnd.oasis.opendocument.text-template": ".ott",
    "application/vnd.oasis.opendocument.text-web": ".oth",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.presentationml.slideshow": ".ppsx",
    "application/vnd.openxmlformats-officedocument.presentationml.template": ".potx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.template": ".xltx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.template": ".dotx",
    "application/vnd.stardivision.calc": ".sdc",
    "application/vnd.stardivision.draw": ".sda",
    "application/vnd.stardivision.impress": ".sdd",
    "application/vnd.stardivision.math": ".smf",
    "application/vnd.stardivision.writer": ".sdw",
    "application/vnd.sun.xml.calc": ".sxc",
    "application/vnd.sun.xml.draw": ".sxd",
    "application/vnd.sun.xml.impress": ".sxi",
    "application/vnd.sun.xml.math": ".sxm",
    "application/vnd.sun.xml.writer": ".sxw",
    "application/vnd.sun.xml.writer.global": ".sxg",
    "application/vnd.uof.presentation": ".uop",
    "application/vnd.uof.spreadsheet": ".uos",
    "application/vnd.uof.text": ".uot",
    "application/vnd.wordperfect": ".wpd",
    "application/xml": ".xml",
    "image/pict": ".pct",
    "image/svg+xml": ".svg",
    "text/csv": ".csv",
    "text/plain": ".txt",
    "text/rtf": ".rtf",
    "text/xml": ".xml",
}


def _resources(resources: list[Path] | None) -> list[Path]:
    """Validate resource paths once and return the normalized list."""
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


def _is_url_source(source: str) -> bool:
    """Return whether a string is an HTTP(S) URL, rejecting malformed or unsupported URLs."""
    if "://" not in source:
        return False
    try:
        parsed = httpx.URL(url=source)
    except httpx.InvalidURL as error:
        msg = f"Malformed HTTP(S) URL source: {source!r}"
        raise ValueError(msg) from error
    if parsed.scheme not in {"http", "https"}:
        msg = f"Unsupported URL scheme {parsed.scheme!r}; only HTTP(S) URLs are supported"
        raise ValueError(msg)

    if not parsed.host:
        msg = f"Malformed HTTP(S) URL source: {source!r}; expected a valid host"
        raise ValueError(msg)
    return True


@component
class GotenbergFileConverter:
    """
    Automatically route local files, typed byte streams, and web URLs to Gotenberg and return ordered PDFs.

    Local Markdown files use Gotenberg's Markdown route, local HTML files use its Chromium HTML route, and every
    other supported local file uses its LibreOffice route. HTTP(S) strings use the Chromium URL route. `ByteStream`
    sources are routed by their MIME type. Resources are validated for every batch but uploaded only with HTML and
    Markdown sources.

    ### Usage example

    ```python
    from pathlib import Path

    from haystack_integrations.components.converters.gotenberg import GotenbergFileConverter

    converter = GotenbergFileConverter()
    result = converter.run(sources=[Path("report.docx"), "https://haystack.deepset.ai"])
    pdfs = result["output"]
    ```
    """

    def __init__(self, url: str = "http://localhost:3000", timeout: float = 30.0) -> None:
        """
        Create a Gotenberg file converter.

        :param url: The URL of the Gotenberg service.
        :param timeout: The request timeout in seconds.
        :raises ValueError: If `url` is not a valid HTTP(S) URL or `timeout` is not greater than zero.
        """
        parsed_url = urlparse(url=url)
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
        return default_to_dict(obj=self, url=self.url, timeout=self.timeout)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize this component from a dictionary."""
        return default_from_dict(cls=cls, data=data)

    @staticmethod
    def _pdf(response: SingleFileResponse | ZipFileResponse, meta: dict[str, Any]) -> ByteStream:
        """Convert a Gotenberg response to a PDF byte stream."""
        if response.is_zip:
            msg = "Gotenberg unexpectedly returned a ZIP archive instead of one PDF"
            raise RuntimeError(msg)
        return ByteStream(data=bytes(response.content), meta=meta, mime_type="application/pdf")

    @staticmethod
    def _local_route(source: Path) -> _Route:
        """Validate a local source and return its route from the file suffix."""
        if not source.is_file():
            msg = f"Local source file does not exist or is not a file: {source}"
            raise FileNotFoundError(msg)
        suffix = source.suffix.lower()
        if not suffix:
            msg = f"Local source filename is missing a file extension: {source.name!r}"
            raise ValueError(msg)
        if suffix in {".md", ".markdown"}:
            return "markdown"
        if suffix in {".html", ".htm", ".xhtml"}:
            return "html"
        if suffix in _LIBREOFFICE_EXTENSIONS:
            return "libreoffice"
        msg = f"Unsupported local source suffix: {suffix!r}"
        raise ValueError(msg)

    @staticmethod
    def _byte_stream_route(source: ByteStream) -> tuple[_Route, str]:
        """Classify a byte stream by its normalized MIME type and return its authoritative suffix."""
        mime_type = (source.mime_type or "").split(";", maxsplit=1)[0].strip().lower()
        if not mime_type:
            msg = "ByteStream sources require a non-empty mime_type"
            raise ValueError(msg)
        if mime_type in {"text/html", "application/xhtml+xml"}:
            return "html", ".html"
        if mime_type in {"text/markdown", "text/x-markdown"}:
            return "markdown", ".md"

        suffix = MIME_TYPE_EXTENSIONS.get(mime_type) or mimetypes.guess_extension(type=mime_type, strict=False)
        if suffix is None or suffix.lower() not in _LIBREOFFICE_EXTENSIONS:
            msg = f"Unsupported ByteStream MIME type: {mime_type!r}"
            raise ValueError(msg)
        return "libreoffice", suffix.lower()

    @staticmethod
    def _safe_stem(filename: str | None) -> str:
        """Return a filesystem-safe source filename stem."""
        stem = re.sub(pattern=r"[^A-Za-z0-9_.-]", repl="_", string=Path(filename or "document").stem)
        return "document" if stem in {"", ".", ".."} else stem

    @overload
    def _prepare(
        self,
        sources: list[str | Path | ByteStream],
        resources: list[Path] | None,
        directory: Path,
        client: SyncGotenbergClient,
    ) -> Sequence[SyncBaseRoute]: ...

    @overload
    def _prepare(
        self,
        sources: list[str | Path | ByteStream],
        resources: list[Path] | None,
        directory: Path,
        client: AsyncGotenbergClient,
    ) -> Sequence[AsyncBaseRoute]: ...

    def _prepare(
        self,
        sources: list[str | Path | ByteStream],
        resources: list[Path] | None,
        directory: Path,
        client: SyncGotenbergClient | AsyncGotenbergClient,
    ) -> Sequence[SyncBaseRoute | AsyncBaseRoute]:
        """Validate sources and resources, then build SDK routes in input order."""
        if not sources:
            msg = "sources must contain at least one source"
            raise ValueError(msg)
        resource_paths = _resources(resources=resources)
        resource_names = {resource.name for resource in resource_paths}
        routes: list[SyncBaseRoute | AsyncBaseRoute] = []
        for index, source in enumerate(sources):
            if isinstance(source, str) and _is_url_source(source=source):
                routes.append(client.chromium.url_to_pdf().url(url=source))
                continue

            filename: str | None
            if isinstance(source, (Path, str)):
                source_path = Path(source)
                kind = self._local_route(source_path)
                if kind == "libreoffice":
                    routes.append(client.libre_office.to_pdf().convert(input_file_path=source_path))
                    continue
                data, filename = source_path.read_bytes(), source_path.name
            elif isinstance(source, ByteStream):
                kind, suffix = self._byte_stream_route(source)
                file_path = source.meta.get("file_path")
                filename = Path(file_path).name if file_path else None
                data = source.data
            else:
                msg = f"Unsupported source type: {type(source).__name__}; expected str, Path, or ByteStream"
                raise TypeError(msg)

            if kind == "libreoffice":
                path = directory / f"{index}-{self._safe_stem(filename)}{suffix}"
                path.write_bytes(data=data)
                routes.append(client.libre_office.to_pdf().convert(input_file_path=path))
                continue
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as error:
                msg = f"{kind.upper() if kind == 'html' else kind.title()} sources must contain UTF-8 text"
                raise ValueError(msg) from error
            if kind == "html":
                html_route = client.chromium.html_to_pdf().string_index(index=text)
                if resource_paths:
                    html_route.resources(resources=resource_paths)
                routes.append(html_route)
                continue

            filename = f"{self._safe_stem(filename=filename)}.md"
            if filename in resource_names:
                msg = f"A resource conflicts with the Markdown source filename: {filename!r}"
                raise ValueError(msg)
            path = directory / str(index) / filename
            path.parent.mkdir()
            path.write_text(data=text, encoding="utf-8")
            head = '<!doctype html><html><head><meta charset="utf-8"></head>'
            template = f'{head}<body>{{{{ toHTML "{filename}" }}}}</body></html>'
            markdown_route = client.chromium.markdown_to_pdf().string_index(index=template)
            markdown_route.markdown_file(markdown_file=path)
            if resource_paths:
                markdown_route.resources(resources=resource_paths)
            routes.append(markdown_route)
        return routes

    @component.output_types(output=list[ByteStream])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
        *,
        resources: list[Path] | None = None,
    ) -> dict[str, list[ByteStream]]:
        """
        Convert automatically routed sources to PDF.

        Each source is classified independently and converted using the corresponding Gotenberg route:

        - A string containing `://` is treated as a URL. It must be an HTTP(S) URL, and Gotenberg's Chromium URL
          route navigates to that URL and prints the resulting page to PDF.
        - A string without `://`, or a `Path`, is treated as a local file. Markdown files (`.md` and `.markdown`) use
          the Chromium Markdown route, HTML files (`.html`, `.htm`, and `.xhtml`) use the Chromium HTML route, and
          other supported file extensions use the LibreOffice route.
        - A `ByteStream` is classified by its MIME type. HTML MIME types use the Chromium HTML route, Markdown MIME
          types use the Chromium Markdown route, and other supported MIME types use the LibreOffice route. For
          LibreOffice conversion, the MIME type determines the staged file extension. MIME parameters such as
          `charset=utf-8` are ignored when classifying the stream.

        HTML and Markdown inputs are uploaded to Gotenberg together with `resources`, while URL inputs are fetched by
        Gotenberg and LibreOffice inputs are uploaded as files. A mixed batch can contain any combination of these
        source types. Routes are executed in input order, and one PDF is produced for each source. HTML and Markdown
        `ByteStream` content must be UTF-8 text.

        :param sources: Sources to convert. Strings, `Path` objects, and `ByteStream` objects are supported as
            described above.
        :param meta: Optional metadata to attach to the output PDFs. A single dictionary is applied to every output. A
            list of dictionaries must have the same length as `sources` and is applied to corresponding outputs.
            Metadata on a source `ByteStream` is preserved, with values from this parameter taking precedence.
        :param resources: Optional local resources for HTML and Markdown conversion. Resources are validated once and
            uploaded only with those routes. Filenames must be unique and cannot be `index.html`; a resource also
            cannot have the same filename as a staged Markdown source.
        :returns: A dictionary containing an `"output"` list with one PDF `ByteStream` per source, in input order.
        :raises TypeError: If a source or resource has an unsupported type.
        :raises FileNotFoundError: If a local source or resource does not exist or is not a file.
        :raises ValueError: If sources, metadata, URLs, suffixes, MIME types, resources, or text contents are invalid.
        :raises RuntimeError: If Gotenberg returns a ZIP archive instead of a PDF.
        """
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))
        with TemporaryDirectory(prefix="haystack-gotenberg-") as temporary_directory:
            with SyncGotenbergClient(host=self.url, timeout=self.timeout, backend="httpx") as client:
                routes = self._prepare(
                    sources=sources, resources=resources, directory=Path(temporary_directory), client=client
                )
                output = [
                    self._pdf(
                        response=route.run(),
                        meta={**(source.meta if isinstance(source, ByteStream) else {}), **source_meta},
                    )
                    for source, route, source_meta in zip(sources, routes, meta_list, strict=True)
                ]
        return {"output": output}

    @component.output_types(output=list[ByteStream])
    async def run_async(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
        *,
        resources: list[Path] | None = None,
    ) -> dict[str, list[ByteStream]]:
        """
        Asynchronously convert automatically routed sources to PDF.

        This is the asynchronous equivalent of `run()` and uses the same source classification and routing rules. Each
        source is classified independently and converted using the corresponding Gotenberg route:

        - A string containing `://` is treated as a valid HTTP(S) URL and converted by Gotenberg's Chromium URL route.
        - A string without `://`, or a `Path`, is treated as a local file. Markdown files (`.md` and `.markdown`) use
          the Chromium Markdown route, HTML files (`.html`, `.htm`, and `.xhtml`) use the Chromium HTML route, and
          other supported file extensions use the LibreOffice route.
        - A `ByteStream` is classified by its MIME type: HTML MIME types use the Chromium HTML route, Markdown MIME
          types use the Chromium Markdown route, and other supported MIME types use the LibreOffice route. The MIME
          type determines the staged file extension for LibreOffice conversion, without its parameters (for example,
          `charset=utf-8`).

        HTML and Markdown inputs are uploaded with `resources`, URL inputs are fetched by Gotenberg, and LibreOffice
        inputs are uploaded as files. A mixed batch can contain any combination of these source types. Routes are
        executed in input order, and one PDF is produced for each source. HTML and Markdown `ByteStream` content must
        be UTF-8 text.

        :param sources: Sources to convert. Strings, `Path` objects, and `ByteStream` objects are supported as
            described above.
        :param meta: Optional metadata to attach to the output PDFs. A single dictionary is applied to every output. A
            list of dictionaries must have the same length as `sources` and is applied to corresponding outputs.
            Metadata on a source `ByteStream` is preserved, with values from this parameter taking precedence.
        :param resources: Optional local resources for HTML and Markdown conversion. Resources are validated once and
            uploaded only with those routes. Filenames must be unique and cannot be `index.html`; a resource also
            cannot have the same filename as a staged Markdown source.
        :returns: A dictionary containing an `"output"` list with one PDF `ByteStream` per source, in input order.
        :raises TypeError: If a source or resource has an unsupported type.
        :raises FileNotFoundError: If a local source or resource does not exist or is not a file.
        :raises ValueError: If sources, metadata, URLs, suffixes, MIME types, resources, or text contents are invalid.
        :raises RuntimeError: If Gotenberg returns a ZIP archive instead of a PDF.
        """
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))
        with TemporaryDirectory(prefix="haystack-gotenberg-") as temporary_directory:
            async with AsyncGotenbergClient(host=self.url, timeout=self.timeout, backend="httpx") as client:
                routes = self._prepare(
                    sources=sources, resources=resources, directory=Path(temporary_directory), client=client
                )
                output = [
                    self._pdf(
                        response=await route.run(),
                        meta={**(source.meta if isinstance(source, ByteStream) else {}), **source_meta},
                    )
                    for source, route, source_meta in zip(sources, routes, meta_list, strict=True)
                ]
        return {"output": output}
