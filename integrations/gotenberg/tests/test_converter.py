# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx2
import pytest
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.gotenberg import GotenbergFileConverter

CONVERTER_MODULE = "haystack_integrations.components.converters.gotenberg.converter"


def _response(
    content: bytes = b"%PDF-mocked", *, status_code: int = 200, content_type: str = "application/pdf"
) -> httpx2.Response:
    return httpx2.Response(
        status_code=status_code,
        content=content,
        headers={"Content-Type": content_type},
        request=httpx2.Request(method="POST", url="http://gotenberg.test/forms/convert"),
    )


def _assert_pdf(result: dict, content: bytes = b"%PDF-mocked", meta: dict | None = None) -> None:
    assert result == {"output": [ByteStream(data=content, meta=meta or {}, mime_type="application/pdf")]}


def _uploaded_files(post_call: call) -> list[tuple[str, tuple]]:
    return post_call.kwargs["files"]


@pytest.fixture(autouse=True)
def mock_httpx2_clients(request: pytest.FixtureRequest):
    """Patch HTTP clients for unit tests without affecting integration tests."""
    if request.node.get_closest_marker("integration"):
        yield None
        return

    with (
        patch(target=f"{CONVERTER_MODULE}.httpx2.Client", autospec=True) as sync_factory,
        patch(target=f"{CONVERTER_MODULE}.httpx2.AsyncClient", autospec=True) as async_factory,
    ):
        sync_client = sync_factory.return_value.__enter__.return_value
        sync_client.post.return_value = _response()
        async_client = MagicMock()
        async_client.post = AsyncMock(return_value=_response())
        async_factory.return_value.__aenter__.return_value = async_client
        yield SimpleNamespace(
            sync_factory=sync_factory,
            sync_client=sync_client,
            async_factory=async_factory,
            async_client=async_client,
        )


def test_init_and_serialization_round_trip() -> None:
    converter = GotenbergFileConverter(url="https://gotenberg.example/", timeout=12.5, concurrency_limit=3)

    assert converter.url == "https://gotenberg.example"
    assert converter.timeout == 12.5
    assert converter.concurrency_limit == 3
    data = converter.to_dict()
    assert data == {
        "type": "haystack_integrations.components.converters.gotenberg.converter.GotenbergFileConverter",
        "init_parameters": {"url": "https://gotenberg.example", "timeout": 12.5, "concurrency_limit": 3},
    }
    restored = GotenbergFileConverter.from_dict(data=data)
    assert restored.url == converter.url
    assert restored.timeout == converter.timeout
    assert restored.concurrency_limit == converter.concurrency_limit


@pytest.mark.parametrize(
    "url", ["", "localhost:3000", "ftp://example.com", "http:///missing-host", "http://example.test:bad"]
)
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


@pytest.mark.parametrize("url", ["http://example.test/page", "https://example.test/secure?q=1"])
def test_url_posts_multipart_scalar(mock_httpx2_clients, url: str) -> None:
    result = GotenbergFileConverter(url="http://gotenberg:3000", timeout=7).run(sources=[url])

    mock_httpx2_clients.sync_factory.assert_called_once_with(base_url="http://gotenberg:3000", timeout=7)
    mock_httpx2_clients.sync_client.post.assert_called_once_with(
        "/forms/chromium/convert/url", files=[("url", (None, url))]
    )
    _assert_pdf(result=result)


@pytest.mark.parametrize("url", ["http:///missing-host", "https://", "https://example.test:not-a-port/page"])
def test_malformed_http_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError, match=r"Malformed HTTP\(S\) URL"):
        GotenbergFileConverter().run(sources=[url])


@pytest.mark.parametrize("url", ["ftp://example.test/file", "file:///tmp/page.html"])
def test_unsupported_url_schemes_are_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        GotenbergFileConverter().run(sources=[url])


@pytest.mark.parametrize("source_type", ["str", "path"])
def test_libreoffice_local_upload_preserves_filename(tmp_path: Path, mock_httpx2_clients, source_type: str) -> None:
    path = tmp_path / "REPORT.DOCX"
    path.write_bytes(b"docx bytes")

    result = GotenbergFileConverter().run(sources=[str(path) if source_type == "str" else path])

    mock_httpx2_clients.sync_client.post.assert_called_once_with(
        "/forms/libreoffice/convert",
        files=[
            (
                "files",
                ("REPORT.DOCX", b"docx bytes", "application/octet-stream"),
            )
        ],
    )
    _assert_pdf(result=result)


@pytest.mark.parametrize("suffix", ".pages .xlsx .key .vsdx .png .pdf .uot".split())
def test_supported_libreoffice_suffixes_use_direct_endpoint(tmp_path: Path, mock_httpx2_clients, suffix: str) -> None:
    path = tmp_path / f"report{suffix}"
    path.write_bytes(b"content")

    GotenbergFileConverter().run(sources=[path])

    post_call = mock_httpx2_clients.sync_client.post.call_args
    assert post_call.args == ("/forms/libreoffice/convert",)
    assert _uploaded_files(post_call)[0][0] == "files"
    assert _uploaded_files(post_call)[0][1][:2] == (path.name, b"content")


@pytest.mark.parametrize("suffix", [".html", ".htm", ".xhtml"])
def test_local_html_uploads_index_and_resources(tmp_path: Path, mock_httpx2_clients, suffix: str) -> None:
    source = tmp_path / f"page{suffix}"
    source.write_text("<h1>Local HTML</h1>", encoding="utf-8")
    resource = tmp_path / "styles.css"
    resource.write_text("body {}", encoding="utf-8")

    result = GotenbergFileConverter().run(sources=[source], resources=[resource])

    post_call = mock_httpx2_clients.sync_client.post.call_args
    assert post_call.args == ("/forms/chromium/convert/html",)
    assert _uploaded_files(post_call) == [
        ("files", ("index.html", "<h1>Local HTML</h1>", "text/html")),
        ("files", ("styles.css", b"body {}", "text/css")),
    ]
    _assert_pdf(result=result)


@pytest.mark.parametrize("suffix", [".md", ".markdown"])
def test_local_markdown_uploads_matching_template_source_and_resources(
    tmp_path: Path, mock_httpx2_clients, suffix: str
) -> None:
    source = tmp_path / f"release notes{suffix}"
    source.write_text("# Local Markdown", encoding="utf-8")
    resource = tmp_path / "logo.png"
    resource.write_bytes(b"png")

    result = GotenbergFileConverter().run(sources=[source], resources=[resource])

    post_call = mock_httpx2_clients.sync_client.post.call_args
    assert post_call.args == ("/forms/chromium/convert/markdown",)
    parts = _uploaded_files(post_call)
    assert [part[0] for part in parts] == ["files", "files", "files"]
    assert parts[0][1][0] == "index.html"
    assert '{{ toHTML "release_notes.md" }}' in parts[0][1][1]
    assert parts[1] == ("files", ("release_notes.md", "# Local Markdown", "text/markdown"))
    assert parts[2] == ("files", ("logo.png", b"png", "image/png"))
    _assert_pdf(result=result)


@pytest.mark.parametrize("source_type", ["str", "path"])
def test_missing_local_file_is_rejected(tmp_path: Path, source_type: str) -> None:
    path = tmp_path / "missing.docx"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GotenbergFileConverter().run(sources=[str(path) if source_type == "str" else path])


def test_colon_in_local_string_path_is_not_treated_as_url(tmp_path: Path, mock_httpx2_clients) -> None:
    source = tmp_path / "quarterly:report.docx"
    source.write_bytes(b"docx")

    GotenbergFileConverter().run(sources=[str(source)])

    assert mock_httpx2_clients.sync_client.post.call_args.args == ("/forms/libreoffice/convert",)


@pytest.mark.parametrize(
    ("filename", "error"),
    [("document", "missing a file extension"), ("program.exe", "Unsupported local source suffix")],
)
def test_invalid_local_suffix_is_rejected(tmp_path: Path, filename: str, error: str) -> None:
    source = tmp_path / filename
    source.write_bytes(b"content")
    with pytest.raises(ValueError, match=error):
        GotenbergFileConverter().run(sources=[source])


def test_unsupported_source_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="expected str, Path, or ByteStream"):
        GotenbergFileConverter().run(sources=[object()])


@pytest.mark.parametrize("mime_type", [None, "", "   ", " ; charset=UTF-8"])
def test_bytestream_requires_non_empty_mime_type(mime_type: str | None) -> None:
    with pytest.raises(ValueError, match="require a non-empty mime_type"):
        GotenbergFileConverter().run(sources=[ByteStream(data=b"document", mime_type=mime_type)])


@pytest.mark.parametrize("mime_type", ["text/html", "application/xhtml+xml"])
def test_html_mime_aliases_upload_index_html(mock_httpx2_clients, mime_type: str) -> None:
    source = ByteStream(data=b"<p>Typed HTML</p>", mime_type=mime_type)

    result = GotenbergFileConverter().run(sources=[source])

    mock_httpx2_clients.sync_client.post.assert_called_once_with(
        "/forms/chromium/convert/html", files=[("files", ("index.html", "<p>Typed HTML</p>", "text/html"))]
    )
    _assert_pdf(result=result)


@pytest.mark.parametrize("mime_type", ["text/markdown", "text/x-markdown"])
def test_markdown_mime_aliases_upload_template_and_source(mock_httpx2_clients, mime_type: str) -> None:
    source = ByteStream(data=b"# Typed Markdown", mime_type=mime_type)

    result = GotenbergFileConverter().run(sources=[source])

    post_call = mock_httpx2_clients.sync_client.post.call_args
    assert post_call.args == ("/forms/chromium/convert/markdown",)
    index, markdown = _uploaded_files(post_call)
    assert index[1][0] == "index.html"
    assert '{{ toHTML "document.md" }}' in index[1][1]
    assert markdown == ("files", ("document.md", "# Typed Markdown", "text/markdown"))
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
        ("  TEXT/PLAIN ; charset=UTF-8 ", ".txt"),
    ],
)
def test_office_bytestream_derives_valid_upload_extension(
    mock_httpx2_clients, mime_type: str, expected_suffix: str
) -> None:
    source = ByteStream(data=b"office bytes", mime_type=mime_type)

    result = GotenbergFileConverter().run(sources=[source])

    post_call = mock_httpx2_clients.sync_client.post.call_args
    assert post_call.args == ("/forms/libreoffice/convert",)
    field, upload = _uploaded_files(post_call)[0]
    assert field == "files"
    assert upload[0] == f"0-document{expected_suffix}"
    assert upload[1] == b"office bytes"
    _assert_pdf(result=result)


def test_unsupported_bytestream_mime_type_is_rejected() -> None:
    source = ByteStream(data=b"unknown", mime_type="application/x-unrecognized-gotenberg")
    with pytest.raises(ValueError, match="Unsupported ByteStream MIME type"):
        GotenbergFileConverter().run(sources=[source])


def test_mime_extension_overrides_metadata_suffix_and_preserves_metadata(mock_httpx2_clients) -> None:
    source = ByteStream(
        data=b"actual docx",
        meta={"file_path": "/uploads/misleading report:final.markdown", "source": "upload"},
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    result = GotenbergFileConverter().run(sources=[source])

    upload = _uploaded_files(mock_httpx2_clients.sync_client.post.call_args)[0][1]
    assert upload[0] == "0-misleading_report_final.docx"
    assert upload[1] == b"actual docx"
    _assert_pdf(result=result, meta={"file_path": "/uploads/misleading report:final.markdown", "source": "upload"})


def test_mixed_batch_posts_in_input_order_and_keeps_pdf_order(tmp_path: Path, mock_httpx2_clients) -> None:
    office = tmp_path / "report.docx"
    office.write_bytes(b"docx")
    mock_httpx2_clients.sync_client.post.side_effect = [
        _response(b"%PDF-url"),
        _response(b"%PDF-html"),
        _response(b"%PDF-markdown"),
        _response(b"%PDF-office"),
    ]

    result = GotenbergFileConverter().run(
        sources=[
            "https://example.test",
            ByteStream(data=b"<h1>HTML</h1>", mime_type="text/html"),
            ByteStream(data=b"# Markdown", mime_type="text/markdown"),
            office,
        ]
    )

    assert [item.args[0] for item in mock_httpx2_clients.sync_client.post.call_args_list] == [
        "/forms/chromium/convert/url",
        "/forms/chromium/convert/html",
        "/forms/chromium/convert/markdown",
        "/forms/libreoffice/convert",
    ]
    assert [item.data for item in result["output"]] == [b"%PDF-url", b"%PDF-html", b"%PDF-markdown", b"%PDF-office"]
    assert all(item.mime_type == "application/pdf" for item in result["output"])


def test_resources_only_appear_in_html_and_markdown_requests(tmp_path: Path, mock_httpx2_clients) -> None:
    resource = tmp_path / "styles.css"
    resource.write_bytes(b"body {}")
    office = tmp_path / "report.docx"
    office.write_bytes(b"docx")

    GotenbergFileConverter().run(
        sources=[
            "https://example.test",
            office,
            ByteStream(data=b"<p>HTML</p>", mime_type="text/html"),
            ByteStream(data=b"# Markdown", mime_type="text/markdown"),
        ],
        resources=[resource],
    )

    calls = mock_httpx2_clients.sync_client.post.call_args_list
    assert len(_uploaded_files(calls[0])) == 1
    assert len(_uploaded_files(calls[1])) == 1
    assert _uploaded_files(calls[2])[-1] == ("files", ("styles.css", b"body {}", "text/css"))
    assert _uploaded_files(calls[3])[-1] == ("files", ("styles.css", b"body {}", "text/css"))


def test_shared_and_explicit_metadata_behavior(mock_httpx2_clients) -> None:
    mock_httpx2_clients.sync_client.post.side_effect = [_response(b"%PDF-one"), _response(b"%PDF-two")]
    first = ByteStream(
        data=b"<p>one</p>",
        meta={"file_path": "original.html", "priority": "source", "source_only": True},
        mime_type="text/html",
    )

    result = GotenbergFileConverter().run(
        sources=[first, "https://two.example"], meta={"priority": "explicit", "batch": "shared"}
    )

    assert [item.meta for item in result["output"]] == [
        {
            "file_path": "original.html",
            "priority": "explicit",
            "source_only": True,
            "batch": "shared",
        },
        {"priority": "explicit", "batch": "shared"},
    ]


@pytest.mark.asyncio
async def test_run_async_uses_async_client_and_preserves_order(mock_httpx2_clients) -> None:
    mock_httpx2_clients.async_client.post.side_effect = [_response(b"%PDF-one"), _response(b"%PDF-two")]

    result = await GotenbergFileConverter(timeout=3).run_async(
        sources=["https://one.example", "http://two.example"],
        meta=[{"source": "one"}, {"source": "two"}],
    )

    mock_httpx2_clients.async_factory.assert_called_once_with(base_url="http://localhost:3000", timeout=3)
    assert mock_httpx2_clients.async_client.post.call_args_list == [
        call("/forms/chromium/convert/url", files=[("url", (None, "https://one.example"))]),
        call("/forms/chromium/convert/url", files=[("url", (None, "http://two.example"))]),
    ]
    assert [item.data for item in result["output"]] == [b"%PDF-one", b"%PDF-two"]
    assert [item.meta for item in result["output"]] == [{"source": "one"}, {"source": "two"}]


@pytest.mark.asyncio
async def test_run_async_posts_requests_concurrently(mock_httpx2_clients) -> None:
    second_request_started = asyncio.Event()

    async def post(*args, **kwargs) -> httpx2.Response:
        del args
        if mock_httpx2_clients.async_client.post.await_count == 2:
            second_request_started.set()
        await asyncio.wait_for(second_request_started.wait(), timeout=1)
        url = kwargs["files"][0][1][1]
        return _response(b"%PDF-one" if url == "https://one.example" else b"%PDF-two")

    mock_httpx2_clients.async_client.post.side_effect = post

    result = await GotenbergFileConverter(concurrency_limit=2).run_async(
        sources=["https://one.example", "https://two.example"]
    )

    assert [item.data for item in result["output"]] == [b"%PDF-one", b"%PDF-two"]


@pytest.mark.asyncio
async def test_run_async_limits_in_flight_requests(mock_httpx2_clients) -> None:
    two_requests_started = asyncio.Event()
    active_requests = 0
    maximum_active_requests = 0

    async def post(*args, **kwargs) -> httpx2.Response:
        nonlocal active_requests, maximum_active_requests
        del args
        active_requests += 1
        maximum_active_requests = max(maximum_active_requests, active_requests)
        if active_requests == 2:
            two_requests_started.set()
        await asyncio.wait_for(two_requests_started.wait(), timeout=1)
        active_requests -= 1
        url = kwargs["files"][0][1][1]
        return _response(url.encode())

    mock_httpx2_clients.async_client.post.side_effect = post

    result = await GotenbergFileConverter(concurrency_limit=2).run_async(
        sources=["https://one.example", "https://two.example", "https://three.example"]
    )

    assert maximum_active_requests == 2
    assert [item.data for item in result["output"]] == [
        b"https://one.example",
        b"https://two.example",
        b"https://three.example",
    ]


@pytest.mark.asyncio
async def test_run_async_routes_mixed_sources(tmp_path: Path, mock_httpx2_clients) -> None:
    office = tmp_path / "report.docx"
    office.write_bytes(b"docx")
    mock_httpx2_clients.async_client.post.side_effect = [_response(b"%PDF-html"), _response(b"%PDF-office")]

    result = await GotenbergFileConverter().run_async(
        sources=[ByteStream(data=b"<p>Async</p>", mime_type="text/html"), office]
    )

    assert [item.args[0] for item in mock_httpx2_clients.async_client.post.call_args_list] == [
        "/forms/chromium/convert/html",
        "/forms/libreoffice/convert",
    ]
    assert [item.data for item in result["output"]] == [b"%PDF-html", b"%PDF-office"]


def test_metadata_list_length_must_match_sources() -> None:
    with pytest.raises(ValueError, match="length of the metadata list"):
        GotenbergFileConverter().run(sources=["https://one.example", "https://two.example"], meta=[{"source": "one"}])


@pytest.mark.asyncio
async def test_run_async_metadata_list_length_must_match_sources() -> None:
    with pytest.raises(ValueError, match="length of the metadata list"):
        await GotenbergFileConverter().run_async(
            sources=["https://one.example", "https://two.example"], meta=[{"source": "one"}]
        )


def test_resource_validation(tmp_path: Path) -> None:
    first_dir, second_dir = tmp_path / "one", tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    first, duplicate = first_dir / "style.css", second_dir / "style.css"
    first.write_text("one", encoding="utf-8")
    duplicate.write_text("two", encoding="utf-8")
    index = tmp_path / "index.html"
    index.write_text("reserved", encoding="utf-8")

    with pytest.raises(TypeError, match="expected Path"):
        GotenbergFileConverter().run(sources=["https://example.test"], resources=[str(first)])
    with pytest.raises(FileNotFoundError, match="does not exist"):
        GotenbergFileConverter().run(sources=["https://example.test"], resources=[tmp_path / "missing.css"])
    with pytest.raises(ValueError, match="unique"):
        GotenbergFileConverter().run(sources=["https://example.test"], resources=[first, duplicate])
    with pytest.raises(ValueError, match="reserved"):
        GotenbergFileConverter().run(sources=["https://example.test"], resources=[index])


def test_markdown_rejects_resource_conflicting_with_source_name(tmp_path: Path) -> None:
    resource = tmp_path / "notes.md"
    resource.write_text("resource", encoding="utf-8")
    source = ByteStream(data=b"# Source", meta={"file_path": "/uploads/notes.markdown"}, mime_type="text/markdown")

    with pytest.raises(ValueError, match="conflicts"):
        GotenbergFileConverter().run(sources=[source], resources=[resource])


@pytest.mark.parametrize(("mime_type", "kind"), [("text/html", "HTML"), ("text/markdown", "Markdown")])
def test_invalid_utf8_is_rejected_for_text_routes(mime_type: str, kind: str) -> None:
    with pytest.raises(ValueError, match=f"{kind} sources must contain UTF-8 text"):
        GotenbergFileConverter().run(sources=[ByteStream(data=b"\xff", mime_type=mime_type)])


def test_http_error_is_raised(mock_httpx2_clients) -> None:
    mock_httpx2_clients.sync_client.post.return_value = _response(
        b"failure", status_code=422, content_type="text/plain"
    )

    with pytest.raises(httpx2.HTTPStatusError):
        GotenbergFileConverter().run(sources=["https://example.test"])


@pytest.mark.parametrize(
    "content_type", ["application/zip", " Application/ZIP ; charset=binary ", "APPLICATION/X-ZIP-COMPRESSED"]
)
def test_zip_content_type_is_rejected(mock_httpx2_clients, content_type: str) -> None:
    mock_httpx2_clients.sync_client.post.return_value = _response(b"PK archive", content_type=content_type)

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
        data=b"# Haystack Markdown integration test\n\nConverted by Gotenberg.", mime_type="text/markdown"
    )
    result = GotenbergFileConverter().run(sources=[source])
    _assert_integration_pdf(result=result)


@pytest.mark.integration
def test_url_converts_page_without_external_network() -> None:
    # The Gotenberg container's own health endpoint makes this deterministic and external-network independent.
    result = GotenbergFileConverter().run(sources=["http://localhost:3000/health"])
    _assert_integration_pdf(result=result)
