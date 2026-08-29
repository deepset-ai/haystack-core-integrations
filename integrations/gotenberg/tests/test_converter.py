# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from gotenberg_client.responses import SingleFileResponse
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.gotenberg import GotenbergFileConverter

CONVERTER_MODULE = "haystack_integrations.components.converters.gotenberg.converter"


def _response(content: bytes = b"%PDF-mocked", *, is_zip: bool = False) -> Mock:
    response = Mock(spec=SingleFileResponse)
    response.content = content
    response.is_zip = is_zip
    return response


def _assert_pdf(result: dict, content: bytes = b"%PDF-mocked", meta: dict | None = None) -> None:
    assert result == {"output": [ByteStream(data=content, meta=meta or {}, mime_type="application/pdf")]}


@pytest.fixture(autouse=True)
def mock_gotenberg_clients(request: pytest.FixtureRequest):
    """Patch SDK clients for unit tests without affecting integration tests."""
    if request.node.get_closest_marker("integration"):
        yield None
        return
    with (
        patch(target=f"{CONVERTER_MODULE}.SyncGotenbergClient", autospec=True) as sync_factory,
        patch(target=f"{CONVERTER_MODULE}.AsyncGotenbergClient", autospec=True) as async_factory,
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


@pytest.fixture
def make_route(mock_gotenberg_clients):
    """Configure and return a synchronous mock route for a Gotenberg endpoint."""

    def make(kind: str, content: bytes = b"%PDF-mocked") -> MagicMock:
        route = MagicMock()
        route.run.return_value = _response(content=content)
        endpoint, method = {
            "url": (mock_gotenberg_clients.sync_client.chromium.url_to_pdf, "url"),
            "html": (mock_gotenberg_clients.sync_client.chromium.html_to_pdf, "string_index"),
            "markdown": (mock_gotenberg_clients.sync_client.chromium.markdown_to_pdf, "string_index"),
            "libreoffice": (mock_gotenberg_clients.sync_client.libre_office.to_pdf, "convert"),
        }[kind]
        endpoint.return_value = route
        getattr(route, method).return_value = route
        return route

    return make


def test_init_and_serialization_round_trip() -> None:
    converter = GotenbergFileConverter(url="https://gotenberg.example/", timeout=12.5)

    assert converter.url == "https://gotenberg.example"
    assert converter.timeout == 12.5
    data = converter.to_dict()
    assert data == {
        "type": "haystack_integrations.components.converters.gotenberg.converter.GotenbergFileConverter",
        "init_parameters": {"url": "https://gotenberg.example", "timeout": 12.5},
    }

    restored = GotenbergFileConverter.from_dict(data=data)
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


def test_run_rejects_empty_sources() -> None:
    with pytest.raises(ValueError, match="at least one"):
        GotenbergFileConverter().run(sources=[])


@pytest.mark.asyncio
async def test_run_async_rejects_empty_sources() -> None:
    with pytest.raises(ValueError, match="at least one"):
        await GotenbergFileConverter().run_async(sources=[])


@pytest.mark.parametrize(
    ("filename", "kind", "content"),
    [
        *[
            (f"REPORT{suffix.upper()}", "libreoffice", b"docx bytes")
            for suffix in ".docx .pages .xlsx .key .vsdx .png .pdf .uot".split()
        ],
        *[(f"page{suffix}", "html", "<h1>Local HTML</h1>") for suffix in ".html .htm .xhtml".split()],
        *[(f"release notes{suffix}", "markdown", "# Local Markdown") for suffix in ".md .markdown".split()],
    ],
)
@pytest.mark.parametrize("source_type", ["str", "path"])
def test_local_sources_route_by_suffix(
    tmp_path: Path, make_route, filename: str, kind: str, content: str | bytes, source_type: str
) -> None:
    path = tmp_path / filename
    if isinstance(content, bytes):
        path.write_bytes(data=content)
    else:
        path.write_text(data=content, encoding="utf-8")
    route = make_route(kind=kind)
    staged_paths: list[Path] = []

    def capture_markdown(markdown_file: Path) -> None:
        staged_paths.append(markdown_file)
        assert markdown_file.read_text(encoding="utf-8") == content

    if kind == "markdown":
        route.markdown_file.side_effect = capture_markdown

    result = GotenbergFileConverter().run(sources=[str(path) if source_type == "str" else path])

    route.run.assert_called_once_with()
    if kind == "libreoffice":
        route.convert.assert_called_once_with(input_file_path=path)
    elif kind == "html":
        route.string_index.assert_called_once_with(index=content)
    else:
        assert len(staged_paths) == 1
        staged_path = staged_paths[0]
        assert staged_path.name == "release_notes.md"
        assert staged_path.parent.parent.name.startswith("haystack-gotenberg-")
        assert not staged_path.exists()
        route.markdown_file.assert_called_once_with(markdown_file=staged_path)
        assert '{{ toHTML "release_notes.md" }}' in route.string_index.call_args.kwargs["index"]
    _assert_pdf(result=result)


@pytest.mark.parametrize("url", ["http://example.test/page", "https://example.test/secure?q=1"])
def test_http_and_https_strings_route_to_url(mock_gotenberg_clients, url: str) -> None:
    route = MagicMock()
    route.url.return_value = route
    route.run.return_value = _response()
    mock_gotenberg_clients.sync_client.chromium.url_to_pdf.return_value = route

    result = GotenbergFileConverter().run(sources=[url])

    mock_gotenberg_clients.sync_client.chromium.url_to_pdf.assert_called_once_with()
    route.url.assert_called_once_with(url=url)
    route.run.assert_called_once_with()
    _assert_pdf(result=result)


@pytest.mark.parametrize(
    "url",
    [
        "http:///missing-host",
        "https://",
        "https://example.test:not-a-port/page",
    ],
)
def test_malformed_http_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError, match=r"Malformed HTTP\(S\) URL"):
        GotenbergFileConverter().run(sources=[url])


@pytest.mark.parametrize("url", ["ftp://example.test/file", "file:///tmp/page.html"])
def test_unsupported_url_schemes_are_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        GotenbergFileConverter().run(sources=[url])


@pytest.mark.parametrize("source_type", ["str", "path"])
def test_missing_local_file_is_rejected(tmp_path: Path, source_type: str) -> None:
    path = tmp_path / "missing.docx"
    source = str(path) if source_type == "str" else path

    with pytest.raises(FileNotFoundError, match="does not exist"):
        GotenbergFileConverter().run(sources=[source])


def test_colon_in_local_string_path_is_not_treated_as_url(tmp_path: Path, mock_gotenberg_clients) -> None:
    source = tmp_path / "quarterly:report.docx"
    source.write_bytes(data=b"docx")
    route = MagicMock()
    route.convert.return_value = route
    route.run.return_value = _response()
    mock_gotenberg_clients.sync_client.libre_office.to_pdf.return_value = route

    GotenbergFileConverter().run(sources=[str(source)])

    route.convert.assert_called_once_with(input_file_path=source)


def test_extensionless_local_file_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "document"
    source.write_text(data="content", encoding="utf-8")

    with pytest.raises(ValueError, match="missing a file extension"):
        GotenbergFileConverter().run(sources=[source])


def test_unsupported_local_suffix_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "program.exe"
    source.write_bytes(data=b"binary")

    with pytest.raises(ValueError, match=r"Unsupported local source suffix: '\.exe'"):
        GotenbergFileConverter().run(sources=[source])


def test_unsupported_source_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="expected str, Path, or ByteStream"):
        GotenbergFileConverter().run(sources=[object()])


@pytest.mark.parametrize("mime_type", [None, "", "   ", " ; charset=UTF-8"])
def test_bytestream_requires_non_empty_mime_type(mime_type: str | None) -> None:
    source = ByteStream(data=b"document", mime_type=mime_type)

    with pytest.raises(ValueError, match="require a non-empty mime_type"):
        GotenbergFileConverter().run(sources=[source])


def test_bytestream_mime_type_is_normalized(mock_gotenberg_clients) -> None:
    route = MagicMock()
    route.convert.return_value = route
    route.run.return_value = _response()
    staged_paths: list[Path] = []

    def capture_upload(input_file_path: Path) -> MagicMock:
        staged_paths.append(input_file_path)
        assert input_file_path.read_bytes() == b"plain text"
        return route

    route.convert.side_effect = capture_upload
    mock_gotenberg_clients.sync_client.libre_office.to_pdf.return_value = route
    source = ByteStream(data=b"plain text", mime_type="  TEXT/PLAIN ; charset=UTF-8 ")

    result = GotenbergFileConverter().run(sources=[source])

    assert len(staged_paths) == 1
    staged_path = staged_paths[0]
    assert staged_path.name == "0-document.txt"
    assert not staged_path.exists()
    route.convert.assert_called_once_with(input_file_path=staged_path)
    _assert_pdf(result=result)


@pytest.mark.parametrize("mime_type", ["text/html", "application/xhtml+xml"])
def test_html_mime_aliases_route_to_html(mock_gotenberg_clients, mime_type: str) -> None:
    route = MagicMock()
    route.string_index.return_value = route
    route.run.return_value = _response()
    mock_gotenberg_clients.sync_client.chromium.html_to_pdf.return_value = route
    source = ByteStream(data=b"<p>Typed HTML</p>", mime_type=mime_type)

    result = GotenbergFileConverter().run(sources=[source])

    route.string_index.assert_called_once_with(index="<p>Typed HTML</p>")
    mock_gotenberg_clients.sync_client.libre_office.to_pdf.assert_not_called()
    _assert_pdf(result=result)


@pytest.mark.parametrize("mime_type", ["text/markdown", "text/x-markdown"])
def test_markdown_mime_aliases_route_to_markdown(mock_gotenberg_clients, mime_type: str) -> None:
    route = MagicMock()
    route.string_index.return_value = route
    route.run.return_value = _response()
    staged_paths: list[Path] = []

    def capture_markdown(markdown_file: Path) -> None:
        staged_paths.append(markdown_file)

    route.markdown_file.side_effect = capture_markdown
    mock_gotenberg_clients.sync_client.chromium.markdown_to_pdf.return_value = route
    source = ByteStream(data=b"# Typed Markdown", mime_type=mime_type)

    result = GotenbergFileConverter().run(sources=[source])

    assert len(staged_paths) == 1
    route.markdown_file.assert_called_once_with(markdown_file=staged_paths[0])
    assert '{{ toHTML "document.md" }}' in route.string_index.call_args.kwargs["index"]
    assert not staged_paths[0].exists()
    _assert_pdf(result=result)


@pytest.mark.parametrize(
    ("mime_type", "expected_suffix"),
    [
        ("application/msword", ".doc"),
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
        ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
        ("application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"),
        ("application/postscript", ".eps"),
        ("image/pict", ".pct"),
    ],
)
def test_common_office_mime_types_route_to_libreoffice(
    mock_gotenberg_clients, mime_type: str, expected_suffix: str
) -> None:
    route = MagicMock()
    route.convert.return_value = route
    route.run.return_value = _response()
    staged_paths: list[Path] = []
    route.convert.side_effect = lambda input_file_path: staged_paths.append(input_file_path) or route
    mock_gotenberg_clients.sync_client.libre_office.to_pdf.return_value = route
    source = ByteStream(data=b"office bytes", mime_type=mime_type)

    result = GotenbergFileConverter().run(sources=[source])

    assert len(staged_paths) == 1
    assert staged_paths[0].suffix == expected_suffix
    assert not staged_paths[0].exists()
    route.convert.assert_called_once_with(input_file_path=staged_paths[0])
    _assert_pdf(result=result)


def test_unsupported_bytestream_mime_type_is_rejected() -> None:
    source = ByteStream(data=b"unknown", mime_type="application/x-unrecognized-gotenberg")

    with pytest.raises(ValueError, match="Unsupported ByteStream MIME type"):
        GotenbergFileConverter().run(sources=[source])


def test_mime_derived_staged_extension_overrides_metadata_suffix(mock_gotenberg_clients) -> None:
    route = MagicMock()
    route.convert.return_value = route
    route.run.return_value = _response()
    captured: list[tuple[Path, bytes]] = []

    def capture_upload(input_file_path: Path) -> MagicMock:
        captured.append((input_file_path, input_file_path.read_bytes()))
        return route

    route.convert.side_effect = capture_upload
    mock_gotenberg_clients.sync_client.libre_office.to_pdf.return_value = route
    source = ByteStream(
        data=b"actual docx",
        meta={"file_path": "/uploads/misleading report:final.markdown", "source": "upload"},
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    result = GotenbergFileConverter().run(sources=[source])

    assert len(captured) == 1
    staged_path, staged_data = captured[0]
    assert staged_path.name == "0-misleading_report_final.docx"
    assert staged_data == b"actual docx"
    assert not staged_path.exists()
    route.convert.assert_called_once_with(input_file_path=staged_path)
    _assert_pdf(
        result=result,
        meta={"file_path": "/uploads/misleading report:final.markdown", "source": "upload"},
    )


def test_mixed_batch_routes_each_source_and_preserves_pdf_order(tmp_path: Path, mock_gotenberg_clients) -> None:
    office_path = tmp_path / "report.docx"
    office_path.write_bytes(data=b"docx")

    url_route = MagicMock()
    url_route.url.return_value = url_route
    url_route.run.return_value = _response(content=b"%PDF-url")
    html_route = MagicMock()
    html_route.string_index.return_value = html_route
    html_route.run.return_value = _response(content=b"%PDF-html")
    markdown_route = MagicMock()
    markdown_route.string_index.return_value = markdown_route
    markdown_route.run.return_value = _response(content=b"%PDF-markdown")
    staged_markdown_paths: list[Path] = []

    def capture_markdown(markdown_file: Path) -> None:
        staged_markdown_paths.append(markdown_file)
        assert markdown_file.read_text(encoding="utf-8") == "# Markdown"

    markdown_route.markdown_file.side_effect = capture_markdown
    office_route = MagicMock()
    office_route.convert.return_value = office_route
    office_route.run.return_value = _response(content=b"%PDF-office")
    mock_gotenberg_clients.sync_client.chromium.url_to_pdf.return_value = url_route
    mock_gotenberg_clients.sync_client.chromium.html_to_pdf.return_value = html_route
    mock_gotenberg_clients.sync_client.chromium.markdown_to_pdf.return_value = markdown_route
    mock_gotenberg_clients.sync_client.libre_office.to_pdf.return_value = office_route

    result = GotenbergFileConverter(url="http://gotenberg:3000", timeout=7).run(
        sources=[
            "https://example.test",
            ByteStream(data=b"<h1>HTML</h1>", mime_type="text/html"),
            ByteStream(data=b"# Markdown", mime_type="text/markdown"),
            office_path,
        ]
    )

    mock_gotenberg_clients.sync_factory.assert_called_once_with(
        host="http://gotenberg:3000", timeout=7, backend="httpx"
    )
    url_route.url.assert_called_once_with(url="https://example.test")
    html_route.string_index.assert_called_once_with(index="<h1>HTML</h1>")
    assert len(staged_markdown_paths) == 1
    staged_markdown_path = staged_markdown_paths[0]
    assert staged_markdown_path.name == "document.md"
    assert not staged_markdown_path.exists()
    markdown_route.markdown_file.assert_called_once_with(markdown_file=staged_markdown_path)
    office_route.convert.assert_called_once_with(input_file_path=office_path)
    assert [item.data for item in result["output"]] == [
        b"%PDF-url",
        b"%PDF-html",
        b"%PDF-markdown",
        b"%PDF-office",
    ]
    assert all(item.mime_type == "application/pdf" for item in result["output"])


def test_resources_are_only_passed_to_text_routes_in_mixed_batch(tmp_path: Path, mock_gotenberg_clients) -> None:
    resource = tmp_path / "styles.css"
    resource.write_text(data="body {}", encoding="utf-8")
    office_path = tmp_path / "report.docx"
    office_path.write_bytes(data=b"docx")

    url_route = MagicMock()
    url_route.url.return_value = url_route
    url_route.run.return_value = _response(content=b"%PDF-url")
    office_route = MagicMock()
    office_route.convert.return_value = office_route
    office_route.run.return_value = _response(content=b"%PDF-office")
    html_route = MagicMock()
    html_route.string_index.return_value = html_route
    html_route.run.return_value = _response(content=b"%PDF-html")
    markdown_route = MagicMock()
    markdown_route.string_index.return_value = markdown_route
    markdown_route.run.return_value = _response(content=b"%PDF-markdown")
    mock_gotenberg_clients.sync_client.chromium.url_to_pdf.return_value = url_route
    mock_gotenberg_clients.sync_client.libre_office.to_pdf.return_value = office_route
    mock_gotenberg_clients.sync_client.chromium.html_to_pdf.return_value = html_route
    mock_gotenberg_clients.sync_client.chromium.markdown_to_pdf.return_value = markdown_route

    result = GotenbergFileConverter().run(
        sources=[
            "https://example.test",
            office_path,
            ByteStream(data=b"<p>HTML</p>", mime_type="text/html"),
            ByteStream(data=b"# Markdown", mime_type="text/markdown"),
        ],
        resources=[resource],
    )

    url_route.resources.assert_not_called()
    office_route.resources.assert_not_called()
    html_route.resources.assert_called_once_with(resources=[resource])
    markdown_route.resources.assert_called_once_with(resources=[resource])
    assert [item.data for item in result["output"]] == [
        b"%PDF-url",
        b"%PDF-office",
        b"%PDF-html",
        b"%PDF-markdown",
    ]


def test_run_applies_shared_metadata_to_every_source(mock_gotenberg_clients) -> None:
    first = MagicMock()
    first.url.return_value = first
    first.run.return_value = _response(content=b"%PDF-one")
    second = MagicMock()
    second.url.return_value = second
    second.run.return_value = _response(content=b"%PDF-two")
    mock_gotenberg_clients.sync_client.chromium.url_to_pdf.side_effect = [first, second]

    result = GotenbergFileConverter().run(
        sources=["https://one.example", "https://two.example"],
        meta={"batch": "shared"},
    )

    assert [item.meta for item in result["output"]] == [{"batch": "shared"}, {"batch": "shared"}]


@pytest.mark.asyncio
async def test_run_async_uses_async_client_and_aligns_metadata_with_pdf_order(mock_gotenberg_clients) -> None:
    first = MagicMock()
    first.url.return_value = first
    first.run = AsyncMock(return_value=_response(content=b"%PDF-one"))
    second = MagicMock()
    second.url.return_value = second
    second.run = AsyncMock(return_value=_response(content=b"%PDF-two"))
    mock_gotenberg_clients.async_client.chromium.url_to_pdf.side_effect = [first, second]

    result = await GotenbergFileConverter(timeout=3).run_async(
        sources=["https://one.example", "http://two.example"],
        meta=[{"source": "one"}, {"source": "two"}],
    )

    mock_gotenberg_clients.async_factory.assert_called_once_with(
        host="http://localhost:3000", timeout=3, backend="httpx"
    )
    assert mock_gotenberg_clients.async_client.chromium.url_to_pdf.call_args_list == [call(), call()]
    first.url.assert_called_once_with(url="https://one.example")
    second.url.assert_called_once_with(url="http://two.example")
    first.run.assert_awaited_once_with()
    second.run.assert_awaited_once_with()
    assert [item.data for item in result["output"]] == [b"%PDF-one", b"%PDF-two"]
    assert [item.meta for item in result["output"]] == [{"source": "one"}, {"source": "two"}]


@pytest.mark.asyncio
async def test_run_async_routes_mixed_mime_and_local_sources(tmp_path: Path, mock_gotenberg_clients) -> None:
    office_path = tmp_path / "report.docx"
    office_path.write_bytes(data=b"docx")
    html_route = MagicMock()
    html_route.string_index.return_value = html_route
    html_route.run = AsyncMock(return_value=_response(content=b"%PDF-html"))
    office_route = MagicMock()
    office_route.convert.return_value = office_route
    office_route.run = AsyncMock(return_value=_response(content=b"%PDF-office"))
    mock_gotenberg_clients.async_client.chromium.html_to_pdf.return_value = html_route
    mock_gotenberg_clients.async_client.libre_office.to_pdf.return_value = office_route

    result = await GotenbergFileConverter().run_async(
        sources=[ByteStream(data=b"<p>Async</p>", mime_type="text/html"), office_path]
    )

    html_route.string_index.assert_called_once_with(index="<p>Async</p>")
    html_route.run.assert_awaited_once_with()
    office_route.convert.assert_called_once_with(input_file_path=office_path)
    office_route.run.assert_awaited_once_with()
    assert [item.data for item in result["output"]] == [b"%PDF-html", b"%PDF-office"]


def test_explicit_meta_overrides_bytestream_meta_and_preserves_other_keys(mock_gotenberg_clients) -> None:
    route = MagicMock()
    route.string_index.return_value = route
    route.run.return_value = _response()
    mock_gotenberg_clients.sync_client.chromium.html_to_pdf.return_value = route
    source = ByteStream(
        data=b"<p>Metadata</p>",
        meta={"file_path": "original.html", "priority": "source", "source_only": True},
        mime_type="text/html",
    )

    result = GotenbergFileConverter().run(
        sources=[source],
        meta={"priority": "explicit", "explicit_only": True},
    )

    _assert_pdf(
        result=result,
        meta={
            "file_path": "original.html",
            "priority": "explicit",
            "source_only": True,
            "explicit_only": True,
        },
    )


def test_metadata_list_length_must_match_sources() -> None:
    with pytest.raises(ValueError, match="length of the metadata list"):
        GotenbergFileConverter().run(
            sources=["https://one.example", "https://two.example"],
            meta=[{"source": "one"}],
        )


@pytest.mark.asyncio
async def test_run_async_metadata_list_length_must_match_sources() -> None:
    with pytest.raises(ValueError, match="length of the metadata list"):
        await GotenbergFileConverter().run_async(
            sources=["https://one.example", "https://two.example"],
            meta=[{"source": "one"}],
        )


def test_resource_validation(tmp_path: Path) -> None:
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "style.css"
    duplicate = second_dir / "style.css"
    first.write_text(data="one", encoding="utf-8")
    duplicate.write_text(data="two", encoding="utf-8")
    index = tmp_path / "index.html"
    index.write_text(data="reserved", encoding="utf-8")

    with pytest.raises(TypeError, match="expected Path"):
        GotenbergFileConverter().run(sources=["https://example.test"], resources=[str(first)])
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GotenbergFileConverter().run(sources=["https://example.test"], resources=[tmp_path / "missing.css"])
    with pytest.raises(ValueError, match="unique"):
        GotenbergFileConverter().run(sources=["https://example.test"], resources=[first, duplicate])
    with pytest.raises(ValueError, match="reserved"):
        GotenbergFileConverter().run(sources=["https://example.test"], resources=[index])


def test_markdown_rejects_resource_conflicting_with_staged_source(tmp_path: Path) -> None:
    resource = tmp_path / "notes.md"
    resource.write_text(data="resource", encoding="utf-8")
    source = ByteStream(
        data=b"# Source",
        meta={"file_path": "/uploads/notes.markdown"},
        mime_type="text/markdown",
    )

    with pytest.raises(ValueError, match="conflicts"):
        GotenbergFileConverter().run(sources=[source], resources=[resource])


@pytest.mark.parametrize(
    ("mime_type", "expected_kind"),
    [("text/html", "HTML"), ("text/markdown", "Markdown")],
)
def test_invalid_utf8_is_rejected_for_text_mime_routes(mime_type: str, expected_kind: str) -> None:
    source = ByteStream(data=b"\xff", mime_type=mime_type)

    with pytest.raises(ValueError, match=f"{expected_kind} sources must contain UTF-8 text"):
        GotenbergFileConverter().run(sources=[source])


def test_sdk_zip_response_is_rejected(mock_gotenberg_clients) -> None:
    route = MagicMock()
    route.url.return_value = route
    route.run.return_value = _response(content=b"PK archive", is_zip=True)
    mock_gotenberg_clients.sync_client.chromium.url_to_pdf.return_value = route

    with pytest.raises(RuntimeError, match="ZIP archive"):
        GotenbergFileConverter().run(sources=["https://example.test"])


def _assert_integration_pdf(result: dict) -> None:
    assert len(result["output"]) == 1
    output = result["output"][0]
    assert isinstance(output, ByteStream)
    assert output.mime_type == "application/pdf"
    assert output.data.startswith(b"%PDF-")


@pytest.mark.integration
def test_libreoffice_converts_local_file_routed_by_suffix() -> None:
    source = Path("tests/test_files/docx/sample_docx.docx")

    result = GotenbergFileConverter().run(sources=[source])

    _assert_integration_pdf(result=result)


@pytest.mark.integration
def test_html_converts_document() -> None:
    source = ByteStream(
        data=b"<!doctype html><html><body><h1>Haystack HTML integration test</h1></body></html>",
        mime_type="text/html",
    )

    result = GotenbergFileConverter().run(sources=[source])

    _assert_integration_pdf(result=result)


@pytest.mark.integration
def test_markdown_converts_document() -> None:
    source = ByteStream(
        data=b"# Haystack Markdown integration test\n\nConverted by Gotenberg.",
        mime_type="text/markdown",
    )

    result = GotenbergFileConverter().run(sources=[source])

    _assert_integration_pdf(result=result)


@pytest.mark.integration
def test_url_converts_page_without_external_network() -> None:
    # The Gotenberg container's own health endpoint makes this deterministic and external-network independent.
    result = GotenbergFileConverter().run(sources=["http://localhost:3000/health"])

    _assert_integration_pdf(result=result)
