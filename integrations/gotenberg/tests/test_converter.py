# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from gotenberg_client.responses import SingleFileResponse
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.gotenberg import ConversionType, GotenbergFileConverter

CONVERTER_MODULE = "haystack_integrations.components.converters.gotenberg.converter"


def _response(content: bytes = b"%PDF-mocked", *, is_zip: bool = False) -> Mock:
    response = Mock(spec=SingleFileResponse)
    response.content = content
    response.is_zip = is_zip
    return response


def _assert_pdf(result: dict, content: bytes = b"%PDF-mocked") -> None:
    assert result == {"output": [ByteStream(data=content, mime_type="application/pdf")]}


@pytest.fixture(autouse=True)
def mock_gotenberg_clients():
    """Patch both SDK clients so unit tests can never contact Gotenberg or the network."""
    with (
        patch(f"{CONVERTER_MODULE}.SyncGotenbergClient", autospec=True) as sync_factory,
        patch(f"{CONVERTER_MODULE}.AsyncGotenbergClient", autospec=True) as async_factory,
    ):
        sync_client = sync_factory.return_value.__enter__.return_value
        async_client = MagicMock()
        async_factory.return_value.__aenter__.return_value = async_client
        yield SimpleNamespace(
            sync_factory=sync_factory,
            sync_client=sync_client,
            async_factory=async_factory,
            async_client=async_client,
        )


def test_init_serialization_and_conversion_type_export() -> None:
    conversion_type: ConversionType = "html"
    converter = GotenbergFileConverter(url="https://gotenberg.example/", timeout=12.5)

    assert conversion_type == "html"
    assert converter.url == "https://gotenberg.example"
    assert converter.timeout == 12.5
    data = converter.to_dict()
    assert data == {
        "type": "haystack_integrations.components.converters.gotenberg.converter.GotenbergFileConverter",
        "init_parameters": {"url": "https://gotenberg.example", "timeout": 12.5},
    }

    restored = GotenbergFileConverter.from_dict(data)
    assert restored.url == converter.url
    assert restored.timeout == converter.timeout


@pytest.mark.parametrize("url", ["", "localhost:3000", "ftp://example.com", "http:///missing-host"])
def test_init_rejects_invalid_url(url: str) -> None:
    with pytest.raises(ValueError, match=r"valid HTTP\(S\)"):
        GotenbergFileConverter(url=url)


@pytest.mark.parametrize("timeout", [0, -1])
def test_init_rejects_invalid_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        GotenbergFileConverter(timeout=timeout)


def test_run_requires_conversion_type() -> None:
    with pytest.raises(TypeError, match="conversion_type"):
        GotenbergFileConverter().run(["source"])


@pytest.mark.asyncio
async def test_run_async_requires_conversion_type() -> None:
    with pytest.raises(TypeError, match="conversion_type"):
        await GotenbergFileConverter().run_async(["source"])


def test_run_rejects_invalid_conversion_type() -> None:
    with pytest.raises(ValueError, match="Unsupported conversion_type"):
        GotenbergFileConverter().run(["source"], "pdf")


@pytest.mark.asyncio
async def test_run_async_rejects_invalid_conversion_type() -> None:
    with pytest.raises(ValueError, match="Unsupported conversion_type"):
        await GotenbergFileConverter().run_async(["source"], "pdf")


@pytest.mark.parametrize("conversion_type", ["libreoffice", "html", "markdown", "url"])
def test_all_routes_reject_empty_sources(conversion_type: ConversionType) -> None:
    with pytest.raises(ValueError, match="at least one"):
        GotenbergFileConverter().run([], conversion_type=conversion_type)


def test_libreoffice_run_converts_path_and_bytestreams_in_order(tmp_path: Path, mock_gotenberg_clients) -> None:
    path = tmp_path / "quarterly report.docx"
    path.write_bytes(b"docx bytes")
    metadata_stream = ByteStream(
        data=b"spreadsheet bytes",
        meta={"file_path": "/uploads/budget.xlsx"},
        mime_type="application/custom-spreadsheet",
    )
    mime_stream = ByteStream(data=b"plain text bytes", mime_type="text/plain")
    captured_uploads: list[tuple[str, bytes]] = []

    path_route = MagicMock()
    path_route.convert.return_value = path_route
    path_route.run.return_value = _response(b"%PDF-path")
    metadata_route = MagicMock()
    metadata_route.convert.side_effect = lambda upload: (
        captured_uploads.append((upload.suffix, upload.read_bytes())) or metadata_route
    )
    metadata_route.run.return_value = _response(b"%PDF-metadata")
    mime_route = MagicMock()
    mime_route.convert.side_effect = lambda upload: (
        captured_uploads.append((upload.suffix, upload.read_bytes())) or mime_route
    )
    mime_route.run.return_value = _response(b"%PDF-mime")
    mock_gotenberg_clients.sync_client.libre_office.to_pdf.side_effect = [
        path_route,
        metadata_route,
        mime_route,
    ]

    result = GotenbergFileConverter(url="http://gotenberg:3000", timeout=7).run(
        [path, metadata_stream, mime_stream], conversion_type="libreoffice"
    )

    mock_gotenberg_clients.sync_factory.assert_called_once_with("http://gotenberg:3000", timeout=7, backend="httpx")
    path_route.convert.assert_called_once_with(path)
    (metadata_upload,) = metadata_route.convert.call_args.args
    (mime_upload,) = mime_route.convert.call_args.args
    assert isinstance(metadata_upload, Path)
    assert isinstance(mime_upload, Path)
    assert metadata_upload.name == "1-budget.xlsx"
    assert mime_upload.name == "2-document.txt"
    assert metadata_upload.parent == mime_upload.parent
    assert metadata_upload.parent.name.startswith("haystack-gotenberg-")
    assert captured_uploads == [(".xlsx", b"spreadsheet bytes"), (".txt", b"plain text bytes")]
    assert not metadata_upload.exists()
    assert not mime_upload.exists()
    assert [item.data for item in result["output"]] == [b"%PDF-path", b"%PDF-metadata", b"%PDF-mime"]
    assert all(item.mime_type == "application/pdf" for item in result["output"])


@pytest.mark.asyncio
async def test_libreoffice_run_async_converts_metadata_and_mime_bytestreams(mock_gotenberg_clients) -> None:
    metadata_route = MagicMock()
    metadata_route.run = AsyncMock(return_value=_response(b"%PDF-async-metadata"))
    mime_route = MagicMock()
    mime_route.run = AsyncMock(return_value=_response(b"%PDF-async-mime"))
    captured_uploads: list[tuple[str, bytes]] = []
    metadata_route.convert.side_effect = lambda upload: (
        captured_uploads.append((upload.suffix, upload.read_bytes())) or metadata_route
    )
    mime_route.convert.side_effect = lambda upload: (
        captured_uploads.append((upload.suffix, upload.read_bytes())) or mime_route
    )
    mock_gotenberg_clients.async_client.libre_office.to_pdf.side_effect = [metadata_route, mime_route]
    metadata_source = ByteStream(data=b"presentation", meta={"file_path": "slides.pptx"})
    mime_source = ByteStream(data=b"async text", mime_type="text/plain")

    result = await GotenbergFileConverter(timeout=3).run_async(
        [metadata_source, mime_source], conversion_type="libreoffice"
    )

    mock_gotenberg_clients.async_factory.assert_called_once_with("http://localhost:3000", timeout=3, backend="httpx")
    (metadata_upload,) = metadata_route.convert.call_args.args
    (mime_upload,) = mime_route.convert.call_args.args
    assert isinstance(metadata_upload, Path)
    assert isinstance(mime_upload, Path)
    assert metadata_upload.name == "0-slides.pptx"
    assert mime_upload.name == "1-document.txt"
    assert metadata_upload.parent == mime_upload.parent
    assert metadata_upload.parent.name.startswith("haystack-gotenberg-")
    assert captured_uploads == [(".pptx", b"presentation"), (".txt", b"async text")]
    assert not metadata_upload.exists()
    assert not mime_upload.exists()
    assert [item.data for item in result["output"]] == [b"%PDF-async-metadata", b"%PDF-async-mime"]
    assert all(item.mime_type == "application/pdf" for item in result["output"])


def test_html_run_reads_path_and_uploads_resources(tmp_path: Path, mock_gotenberg_clients) -> None:
    source = tmp_path / "page.html"
    source.write_text("<h1>Hello</h1>", encoding="utf-8")
    css = tmp_path / "style.css"
    css.write_text("body { color: black; }", encoding="utf-8")
    route = MagicMock()
    route.string_index.return_value = route
    route.run.return_value = _response()
    mock_gotenberg_clients.sync_client.chromium.html_to_pdf.return_value = route

    result = GotenbergFileConverter().run([source], conversion_type="html", resources=[css])

    mock_gotenberg_clients.sync_client.chromium.html_to_pdf.assert_called_once_with()
    route.string_index.assert_called_once_with("<h1>Hello</h1>")
    route.resources.assert_called_once_with([css])
    _assert_pdf(result)


@pytest.mark.asyncio
async def test_html_run_async_reads_utf8_bytestream(mock_gotenberg_clients) -> None:
    route = MagicMock()
    route.string_index.return_value = route
    route.run = AsyncMock(return_value=_response(b"%PDF-async-html"))
    mock_gotenberg_clients.async_client.chromium.html_to_pdf.return_value = route

    result = await GotenbergFileConverter().run_async([ByteStream(data="<p>Hé</p>".encode())], conversion_type="html")

    route.string_index.assert_called_once_with("<p>Hé</p>")
    route.resources.assert_not_called()
    _assert_pdf(result, b"%PDF-async-html")


def test_markdown_run_builds_template_and_uploads_resources(tmp_path: Path, mock_gotenberg_clients) -> None:
    image = tmp_path / "logo.png"
    image.write_bytes(b"image")
    route = MagicMock()
    route.string_index.return_value = route
    route.run.return_value = _response()
    uploaded_markdown: dict[str, str] = {}

    def capture_markdown(path: Path) -> MagicMock:
        uploaded_markdown["name"] = path.name
        uploaded_markdown["text"] = path.read_text(encoding="utf-8")
        return route

    route.markdown_file.side_effect = capture_markdown
    mock_gotenberg_clients.sync_client.chromium.markdown_to_pdf.return_value = route
    source = ByteStream(data=b"# Heading", meta={"file_path": "/uploads/my notes.md"})

    result = GotenbergFileConverter().run([source], conversion_type="markdown", resources=[image])

    assert uploaded_markdown == {"name": "my_notes.md", "text": "# Heading"}
    assert '{{ toHTML "my_notes.md" }}' in route.string_index.call_args.args[0]
    route.resources.assert_called_once_with([image])
    _assert_pdf(result)


@pytest.mark.asyncio
async def test_markdown_run_async_uses_default_filename(mock_gotenberg_clients) -> None:
    route = MagicMock()
    route.string_index.return_value = route
    route.run = AsyncMock(return_value=_response(b"%PDF-async-markdown"))
    uploaded_names: list[str] = []
    route.markdown_file.side_effect = lambda path: uploaded_names.append(path.name)
    mock_gotenberg_clients.async_client.chromium.markdown_to_pdf.return_value = route

    result = await GotenbergFileConverter().run_async(["# Async"], conversion_type="markdown")

    assert uploaded_names == ["document.md"]
    assert '{{ toHTML "document.md" }}' in route.string_index.call_args.args[0]
    _assert_pdf(result, b"%PDF-async-markdown")


def test_url_run_converts_each_url_in_order(mock_gotenberg_clients) -> None:
    first = MagicMock()
    first.url.return_value = first
    first.run.return_value = _response(b"%PDF-one")
    second = MagicMock()
    second.url.return_value = second
    second.run.return_value = _response(b"%PDF-two")
    mock_gotenberg_clients.sync_client.chromium.url_to_pdf.side_effect = [first, second]

    result = GotenbergFileConverter().run(["https://one.example/page", "http://two.example"], conversion_type="url")

    assert mock_gotenberg_clients.sync_client.chromium.url_to_pdf.call_args_list == [call(), call()]
    first.url.assert_called_once_with("https://one.example/page")
    second.url.assert_called_once_with("http://two.example")
    assert [item.data for item in result["output"]] == [b"%PDF-one", b"%PDF-two"]


@pytest.mark.asyncio
async def test_url_run_async_calls_url_route(mock_gotenberg_clients) -> None:
    route = MagicMock()
    route.url.return_value = route
    route.run = AsyncMock(return_value=_response(b"%PDF-async-url"))
    mock_gotenberg_clients.async_client.chromium.url_to_pdf.return_value = route

    result = await GotenbergFileConverter().run_async(["https://example.test"], conversion_type="url")

    route.url.assert_called_once_with("https://example.test")
    _assert_pdf(result, b"%PDF-async-url")


@pytest.mark.parametrize(
    ("sources", "exception", "message"),
    [
        (
            [ByteStream(data=b"document", mime_type="application/x-unrecognized-gotenberg")],
            ValueError,
            "either a filename with an extension.*recognized mime_type",
        ),
        ([ByteStream(data=b"document", meta={"file_path": "document"})], ValueError, "recognized mime_type"),
        ([object()], TypeError, "Unsupported LibreOffice source type"),
    ],
)
def test_libreoffice_validates_source(sources: list, exception: type[Exception], message: str) -> None:
    with pytest.raises(exception, match=message):
        GotenbergFileConverter().run(sources, conversion_type="libreoffice")


def test_libreoffice_validates_path_and_rejects_resources(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GotenbergFileConverter().run([tmp_path / "missing.docx"], conversion_type="libreoffice")

    extensionless = tmp_path / "document"
    extensionless.write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match="extension"):
        GotenbergFileConverter().run([extensionless], conversion_type="libreoffice")
    with pytest.raises(ValueError, match="resources are not supported"):
        GotenbergFileConverter().run([extensionless], conversion_type="libreoffice", resources=[])


@pytest.mark.parametrize("conversion_type", ["html", "markdown"])
def test_text_routes_validate_source_type_existence_and_utf8(tmp_path: Path, conversion_type: ConversionType) -> None:
    source_name = conversion_type.upper() if conversion_type == "html" else "Markdown"
    with pytest.raises(TypeError, match=f"Unsupported {source_name}"):
        GotenbergFileConverter().run([object()], conversion_type=conversion_type)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GotenbergFileConverter().run([tmp_path / "missing.txt"], conversion_type=conversion_type)
    with pytest.raises(ValueError, match="UTF-8"):
        GotenbergFileConverter().run([ByteStream(data=b"\xff")], conversion_type=conversion_type)


@pytest.mark.parametrize("conversion_type", ["html", "markdown"])
def test_text_routes_validate_resources(tmp_path: Path, conversion_type: ConversionType) -> None:
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "style.css"
    duplicate = second_dir / "style.css"
    first.write_text("one", encoding="utf-8")
    duplicate.write_text("two", encoding="utf-8")
    index = tmp_path / "index.html"
    index.write_text("reserved", encoding="utf-8")

    with pytest.raises(TypeError, match="expected Path"):
        GotenbergFileConverter().run(["text"], conversion_type=conversion_type, resources=[str(first)])
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GotenbergFileConverter().run(["text"], conversion_type=conversion_type, resources=[tmp_path / "missing.css"])
    with pytest.raises(ValueError, match="unique"):
        GotenbergFileConverter().run(["text"], conversion_type=conversion_type, resources=[first, duplicate])
    with pytest.raises(ValueError, match="reserved"):
        GotenbergFileConverter().run(["text"], conversion_type=conversion_type, resources=[index])


def test_markdown_rejects_resource_conflicting_with_source(tmp_path: Path) -> None:
    resource = tmp_path / "notes.md"
    resource.write_text("resource", encoding="utf-8")
    source = ByteStream(data=b"source", meta={"file_path": "notes.md"})

    with pytest.raises(ValueError, match="conflicts"):
        GotenbergFileConverter().run([source], conversion_type="markdown", resources=[resource])


@pytest.mark.parametrize("sources", [["localhost:8080"], ["file:///tmp/page.html"]])
def test_url_route_rejects_invalid_urls(sources: list[str]) -> None:
    with pytest.raises(ValueError, match=r"valid HTTP\(S\)"):
        GotenbergFileConverter().run(sources, conversion_type="url")


def test_url_route_rejects_non_string_and_resources() -> None:
    with pytest.raises(TypeError, match=r"HTTP\(S\) string"):
        GotenbergFileConverter().run([Path("page.html")], conversion_type="url")
    with pytest.raises(ValueError, match="resources are not supported"):
        GotenbergFileConverter().run(["https://example.test"], conversion_type="url", resources=[])


def test_sdk_zip_response_is_rejected(mock_gotenberg_clients) -> None:
    route = MagicMock()
    route.url.return_value = route
    route.run.return_value = _response(b"PK archive", is_zip=True)
    mock_gotenberg_clients.sync_client.chromium.url_to_pdf.return_value = route

    with pytest.raises(RuntimeError, match="ZIP archive"):
        GotenbergFileConverter().run(["https://example.test"], conversion_type="url")
