# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.components.converters.docling_serve import ConversionMode, DoclingServeConverter, ExportType


def _mock_httpx_response(content: str, export_type: ExportType = ExportType.MARKDOWN) -> httpx.Response:
    content_key = {"markdown": "md_content", "text": "text_content", "json": "json_content"}[export_type.value]
    json_data: dict = {"document": {content_key: content}, "status": "success"}
    return httpx.Response(200, json=json_data, request=httpx.Request("POST", "http://test"))


def _mock_task_response(task_id: str = "task-1", status: str = "pending") -> httpx.Response:
    json_data = {"task_id": task_id, "task_type": "convert", "task_status": status}
    return httpx.Response(200, json=json_data, request=httpx.Request("POST", "http://test"))


def _mock_status_response(
    task_id: str = "task-1", status: str = "success", error_message: str | None = None
) -> httpx.Response:
    json_data = {
        "task_id": task_id,
        "task_type": "convert",
        "task_status": status,
        "error_message": error_message,
    }
    return httpx.Response(200, json=json_data, request=httpx.Request("GET", "http://test"))


def _mock_failed_result_response(error_message: str = "conversion failed", status: str = "failure") -> httpx.Response:
    json_data = {"status": status, "errors": [{"error_message": error_message}]}
    return httpx.Response(200, json=json_data, request=httpx.Request("GET", "http://test"))


class TestDoclingServeConverterInit:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("DOCLING_SERVE_API_KEY", raising=False)
        converter = DoclingServeConverter()
        assert converter.base_url == "http://localhost:5001"
        assert converter.export_type == ExportType.MARKDOWN
        assert converter.convert_options == {}
        assert converter.timeout == 120.0
        assert converter.mode == ConversionMode.SYNC
        assert converter.poll_interval == 2.0
        assert converter.job_timeout == 600.0
        # Default is Secret(DOCLING_SERVE_API_KEY, strict=False) — resolves to None when env var is unset
        assert converter.api_key is not None
        assert converter.api_key.resolve_value() is None

    def test_custom_params(self):
        converter = DoclingServeConverter(
            base_url="http://myserver:8080/",
            export_type=ExportType.TEXT,
            convert_options={"do_ocr": True},
            timeout=60.0,
            api_key=None,
            mode="async",
            poll_interval=5.0,
            job_timeout=300.0,
        )
        assert converter.base_url == "http://myserver:8080"  # trailing slash stripped
        assert converter.export_type == ExportType.TEXT
        assert converter.convert_options == {"do_ocr": True}
        assert converter.timeout == 60.0
        assert converter.mode == ConversionMode.ASYNC
        assert converter.poll_interval == 5.0
        assert converter.job_timeout == 300.0

    def test_trailing_slash_stripped(self):
        converter = DoclingServeConverter(base_url="http://localhost:5001/")
        assert converter.base_url == "http://localhost:5001"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("DOCLING_SERVE_API_KEY", "test-key")
        converter = DoclingServeConverter()
        assert converter.api_key is not None
        assert converter.api_key.resolve_value() == "test-key"

    def test_api_key_none_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DOCLING_SERVE_API_KEY", "test-key")
        converter = DoclingServeConverter(api_key=None)
        assert converter.api_key is None

    def test_invalid_async_job_settings(self):
        with pytest.raises(ValueError, match="poll_interval"):
            DoclingServeConverter(mode=ConversionMode.ASYNC, poll_interval=0)

        with pytest.raises(ValueError, match="job_timeout"):
            DoclingServeConverter(mode=ConversionMode.ASYNC, job_timeout=0)

    def test_sync_mode_allows_unused_async_job_settings(self):
        converter = DoclingServeConverter(poll_interval=0, job_timeout=0)
        assert converter.mode == ConversionMode.SYNC
        assert converter.poll_interval == 0
        assert converter.job_timeout == 0


class TestDoclingServeConverterSerialization:
    def test_to_dict(self):
        converter = DoclingServeConverter(
            base_url="http://localhost:5001",
            export_type=ExportType.TEXT,
            convert_options={"do_ocr": False},
            timeout=30.0,
            api_key=None,
            mode=ConversionMode.ASYNC,
            poll_interval=4.0,
            job_timeout=120.0,
        )
        data = converter.to_dict()
        assert data["type"].endswith("DoclingServeConverter")
        params = data["init_parameters"]
        assert params["base_url"] == "http://localhost:5001"
        assert params["export_type"] == "text"
        assert params["convert_options"] == {"do_ocr": False}
        assert params["timeout"] == 30.0
        assert params["api_key"] is None
        assert params["mode"] == "async"
        assert params["poll_interval"] == 4.0
        assert params["job_timeout"] == 120.0

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.converters.docling_serve.converter.DoclingServeConverter",
            "init_parameters": {
                "base_url": "http://myserver:9000",
                "export_type": "json",
                "convert_options": {},
                "timeout": 60.0,
                "api_key": None,
                "mode": "async",
                "poll_interval": 3.0,
                "job_timeout": 240.0,
            },
        }
        converter = DoclingServeConverter.from_dict(data)
        assert converter.base_url == "http://myserver:9000"
        assert converter.export_type == ExportType.JSON
        assert converter.timeout == 60.0
        assert converter.mode == ConversionMode.ASYNC
        assert converter.poll_interval == 3.0
        assert converter.job_timeout == 240.0

    def test_to_dict_with_api_key(self, monkeypatch):
        monkeypatch.setenv("DOCLING_API_KEY", "test-key")
        converter = DoclingServeConverter(api_key=Secret.from_env_var("DOCLING_API_KEY"))
        data = converter.to_dict()
        assert data["init_parameters"]["api_key"] is not None
        assert data["init_parameters"]["api_key"]["type"] == "env_var"

    def test_roundtrip(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "val")
        converter = DoclingServeConverter(
            base_url="http://remote:5001",
            export_type=ExportType.MARKDOWN,
            convert_options={"table_mode": "fast"},
            timeout=45.0,
            api_key=Secret.from_env_var("MY_KEY"),
        )
        data = converter.to_dict()
        restored = DoclingServeConverter.from_dict(data)
        assert restored.base_url == converter.base_url
        assert restored.export_type == converter.export_type
        assert restored.convert_options == converter.convert_options
        assert restored.timeout == converter.timeout
        assert restored.api_key is not None


class TestDoclingServeConverterHeaders:
    def test_no_api_key(self):
        converter = DoclingServeConverter(api_key=None)
        assert converter._headers() == {}

    def test_with_api_key(self):
        converter = DoclingServeConverter(api_key=Secret.from_token("my-secret-token"))
        headers = converter._headers()
        assert headers["X-Api-Key"] == "my-secret-token"


class TestDoclingServeConverterRun:
    def test_run_url_source(self):
        converter = DoclingServeConverter(api_key=None)
        mock_resp = _mock_httpx_response("# Hello World")

        with patch("httpx.Client.post", return_value=mock_resp):
            result = converter.run(sources=["https://example.com/doc.pdf"])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "# Hello World"

    def test_run_url_uses_source_endpoint(self):
        converter = DoclingServeConverter(api_key=None)
        mock_resp = _mock_httpx_response("content")

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            converter.run(sources=["https://example.com/doc.pdf"])

        url = mock_post.call_args[0][0]
        assert "/v1/convert/source" in url

    def test_run_file_uses_file_endpoint(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-test")
        converter = DoclingServeConverter(api_key=None)
        mock_resp = _mock_httpx_response("# PDF content")

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            result = converter.run(sources=[pdf])

        url = mock_post.call_args[0][0]
        assert "/v1/convert/file" in url
        assert result["documents"][0].content == "# PDF content"

    def test_run_bytestream_uses_file_endpoint(self):
        converter = DoclingServeConverter(api_key=None)
        mock_resp = _mock_httpx_response("content")
        bs = ByteStream(data=b"bytes", meta={"file_name": "report.pdf"})

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            converter.run(sources=[bs])

        url = mock_post.call_args[0][0]
        assert "/v1/convert/file" in url

    def test_run_multiple_sources(self):
        converter = DoclingServeConverter(api_key=None)
        mock_resp = _mock_httpx_response("content")

        with patch("httpx.Client.post", return_value=mock_resp):
            result = converter.run(sources=["https://a.com/1.pdf", "https://b.com/2.pdf"])

        assert len(result["documents"]) == 2

    def test_run_with_meta(self):
        converter = DoclingServeConverter(api_key=None)
        mock_resp = _mock_httpx_response("text")

        with patch("httpx.Client.post", return_value=mock_resp):
            result = converter.run(sources=["https://example.com/doc.pdf"], meta={"author": "Alice"})

        assert result["documents"][0].meta["author"] == "Alice"

    def test_run_bytestream_meta_merged(self):
        converter = DoclingServeConverter(api_key=None)
        mock_resp = _mock_httpx_response("text")
        bs = ByteStream(data=b"bytes", meta={"file_path": "doc.pdf"})

        with patch("httpx.Client.post", return_value=mock_resp):
            result = converter.run(sources=[bs], meta={"page": 1})

        doc = result["documents"][0]
        assert doc.meta["file_path"] == "doc.pdf"
        assert doc.meta["page"] == 1

    def test_run_skips_on_http_status_error(self, caplog):
        converter = DoclingServeConverter(api_key=None)
        error_response = httpx.Response(500, text="Server error", request=httpx.Request("POST", "http://test"))

        with patch(
            "httpx.Client.post",
            side_effect=httpx.HTTPStatusError("err", request=error_response.request, response=error_response),
        ):
            with caplog.at_level(logging.WARNING):
                result = converter.run(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "HTTP 500" in caplog.text

    def test_run_skips_on_connection_error(self, caplog):
        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.Client.post", side_effect=httpx.ConnectError("Connection refused")):
            with caplog.at_level(logging.WARNING):
                result = converter.run(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "Could not connect" in caplog.text

    def test_run_skips_when_no_content(self, caplog):
        converter = DoclingServeConverter(api_key=None)
        mock_resp = httpx.Response(
            200,
            json={"document": {"md_content": None}, "status": "success"},
            request=httpx.Request("POST", "http://test"),
        )

        with patch("httpx.Client.post", return_value=mock_resp):
            with caplog.at_level(logging.WARNING):
                result = converter.run(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "No content returned" in caplog.text

    def test_run_skips_failed_sync_url_response_details(self, caplog):
        converter = DoclingServeConverter(api_key=None)
        mock_resp = _mock_failed_result_response("url conversion failed")

        with patch("httpx.Client.post", return_value=mock_resp):
            with caplog.at_level(logging.WARNING):
                result = converter.run(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "url conversion failed" in caplog.text

    def test_run_skips_failed_sync_file_response_details(self, tmp_path, caplog):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-test")
        converter = DoclingServeConverter(api_key=None)
        mock_resp = _mock_failed_result_response("file conversion failed")

        with patch("httpx.Client.post", return_value=mock_resp):
            with caplog.at_level(logging.WARNING):
                result = converter.run(sources=[pdf])

        assert result["documents"] == []
        assert "file conversion failed" in caplog.text

    def test_run_skips_missing_local_file(self, tmp_path, caplog):
        missing = tmp_path / "missing.pdf"
        good = tmp_path / "good.pdf"
        good.write_bytes(b"%PDF-test")
        converter = DoclingServeConverter(api_key=None)
        mock_resp = _mock_httpx_response("# Good content")

        with patch("httpx.Client.post", return_value=mock_resp):
            with caplog.at_level(logging.WARNING):
                result = converter.run(sources=[missing, good])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "# Good content"
        assert "Could not read local source" in caplog.text

    def test_run_text_export(self):
        converter = DoclingServeConverter(export_type=ExportType.TEXT, api_key=None)
        mock_resp = _mock_httpx_response("plain text content", ExportType.TEXT)

        with patch("httpx.Client.post", return_value=mock_resp):
            result = converter.run(sources=["https://example.com/doc.pdf"])

        assert result["documents"][0].content == "plain text content"

    def test_run_json_export(self):
        converter = DoclingServeConverter(export_type=ExportType.JSON, api_key=None)
        json_doc = {"schema_name": "DoclingDocument", "pages": []}
        mock_resp = _mock_httpx_response(json_doc, ExportType.JSON)  # type: ignore[arg-type]

        with patch("httpx.Client.post", return_value=mock_resp):
            result = converter.run(sources=["https://example.com/doc.pdf"])

        content = json.loads(result["documents"][0].content)
        assert content["schema_name"] == "DoclingDocument"

    def test_run_sends_api_key_header(self):
        converter = DoclingServeConverter(api_key=Secret.from_token("my-secret-token"))
        mock_resp = _mock_httpx_response("content")

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            converter.run(sources=["https://example.com/doc.pdf"])

        headers = mock_post.call_args[1]["headers"]
        assert headers["X-Api-Key"] == "my-secret-token"

    def test_run_url_payload_includes_to_formats(self):
        converter = DoclingServeConverter(export_type=ExportType.MARKDOWN, api_key=None)
        mock_resp = _mock_httpx_response("content")

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            converter.run(sources=["https://example.com/doc.pdf"])

        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["to_formats"] == ["md"]

    def test_run_convert_options_merged_into_payload(self):
        converter = DoclingServeConverter(convert_options={"do_ocr": True, "table_mode": "accurate"}, api_key=None)
        mock_resp = _mock_httpx_response("content")

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            converter.run(sources=["https://example.com/doc.pdf"])

        payload = mock_post.call_args[1]["json"]
        assert payload["options"]["do_ocr"] is True
        assert payload["options"]["table_mode"] == "accurate"

    def test_run_async_mode_uses_job_endpoints_for_url_source(self):
        converter = DoclingServeConverter(api_key=None, mode=ConversionMode.ASYNC, poll_interval=3.0)

        with (
            patch("httpx.Client.post", return_value=_mock_task_response()) as mock_post,
            patch(
                "httpx.Client.get", side_effect=[_mock_status_response(), _mock_httpx_response("async content")]
            ) as mock_get,
        ):
            result = converter.run(sources=["https://example.com/doc.pdf"])

        assert result["documents"][0].content == "async content"

        post_url = mock_post.call_args[0][0]
        payload = mock_post.call_args[1]["json"]
        assert post_url.endswith("/v1/convert/source/async")
        assert payload["options"]["to_formats"] == ["md"]
        assert payload["target"] == {"kind": "inbody"}

        status_call, result_call = mock_get.call_args_list
        assert status_call.args[0].endswith("/v1/status/poll/task-1")
        assert status_call.kwargs["params"] == {"wait": 3.0}
        assert result_call.args[0].endswith("/v1/result/task-1")

    def test_run_async_mode_uses_job_endpoints_for_file_source(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-test")
        converter = DoclingServeConverter(api_key=None, mode="async")

        with (
            patch("httpx.Client.post", return_value=_mock_task_response()) as mock_post,
            patch("httpx.Client.get", side_effect=[_mock_status_response(), _mock_httpx_response("async content")]),
        ):
            result = converter.run(sources=[pdf])

        assert result["documents"][0].content == "async content"

        post_url = mock_post.call_args[0][0]
        data = mock_post.call_args[1]["data"]
        assert post_url.endswith("/v1/convert/file/async")
        assert data["to_formats"] == "md"
        assert data["target_type"] == "inbody"

    def test_run_async_mode_skips_failed_job(self, caplog):
        converter = DoclingServeConverter(api_key=None, mode=ConversionMode.ASYNC)

        with (
            patch("httpx.Client.post", return_value=_mock_task_response()),
            patch(
                "httpx.Client.get",
                return_value=_mock_status_response(status="failure", error_message="conversion failed"),
            ),
        ):
            with caplog.at_level(logging.WARNING):
                result = converter.run(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "conversion failed" in caplog.text

    def test_run_async_mode_skips_failed_result(self, caplog):
        converter = DoclingServeConverter(api_key=None, mode=ConversionMode.ASYNC)

        with (
            patch("httpx.Client.post", return_value=_mock_task_response()),
            patch("httpx.Client.get", side_effect=[_mock_status_response(), _mock_failed_result_response("bad file")]),
        ):
            with caplog.at_level(logging.WARNING):
                result = converter.run(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "bad file" in caplog.text

    def test_run_async_mode_skips_skipped_result(self, caplog):
        converter = DoclingServeConverter(api_key=None, mode=ConversionMode.ASYNC)

        with (
            patch("httpx.Client.post", return_value=_mock_task_response()),
            patch(
                "httpx.Client.get",
                side_effect=[_mock_status_response(), _mock_failed_result_response("skipped file", status="skipped")],
            ),
        ):
            with caplog.at_level(logging.WARNING):
                result = converter.run(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "skipped file" in caplog.text

    def test_run_async_mode_skips_timed_out_job(self, caplog):
        converter = DoclingServeConverter(
            api_key=None, mode=ConversionMode.ASYNC, poll_interval=0.001, job_timeout=0.001
        )

        with (
            patch("httpx.Client.post", return_value=_mock_task_response()),
            patch("httpx.Client.get", return_value=_mock_status_response(status="pending")),
        ):
            with caplog.at_level(logging.WARNING):
                result = converter.run(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "Timed out" in caplog.text


class TestDoclingServeConverterFilename:
    def test_bytestream_filename_from_file_name(self):
        converter = DoclingServeConverter(api_key=None)
        bs = ByteStream(data=b"bytes", meta={"file_name": "report.pdf"})
        mock_resp = _mock_httpx_response("content")

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            converter.run(sources=[bs])

        files = mock_post.call_args[1]["files"]
        assert files["files"][0] == "report.pdf"

    def test_bytestream_filename_from_file_path(self):
        converter = DoclingServeConverter(api_key=None)
        bs = ByteStream(data=b"bytes", meta={"file_path": "/data/reports/annual.pdf"})
        mock_resp = _mock_httpx_response("content")

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            converter.run(sources=[bs])

        files = mock_post.call_args[1]["files"]
        assert files["files"][0] == "annual.pdf"

    def test_bytestream_filename_from_name(self):
        converter = DoclingServeConverter(api_key=None)
        bs = ByteStream(data=b"bytes", meta={"name": "summary.docx"})
        mock_resp = _mock_httpx_response("content")

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            converter.run(sources=[bs])

        files = mock_post.call_args[1]["files"]
        assert files["files"][0] == "summary.docx"

    def test_bytestream_filename_fallback(self):
        converter = DoclingServeConverter(api_key=None)
        bs = ByteStream(data=b"bytes")
        mock_resp = _mock_httpx_response("content")

        with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
            converter.run(sources=[bs])

        files = mock_post.call_args[1]["files"]
        assert files["files"][0] == "document"


class TestDoclingServeConverterRunAsync:
    @pytest.mark.asyncio
    async def test_run_async_url_source(self):
        converter = DoclingServeConverter(api_key=None)
        mock_resp = httpx.Response(
            200,
            json={"document": {"md_content": "# Async content"}, "status": "success"},
            request=httpx.Request("POST", "http://test"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await converter.run_async(sources=["https://example.com/doc.pdf"])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "# Async content"

    @pytest.mark.asyncio
    async def test_run_async_file_source(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-test")
        converter = DoclingServeConverter(api_key=None)
        mock_resp = httpx.Response(
            200,
            json={"document": {"md_content": "# Async file content"}, "status": "success"},
            request=httpx.Request("POST", "http://test"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
            result = await converter.run_async(sources=[pdf])

        post_url = mock_post.call_args[0][0]
        assert post_url.endswith("/v1/convert/file")
        assert result["documents"][0].content == "# Async file content"

    @pytest.mark.asyncio
    async def test_run_async_skips_failed_sync_response_details(self, caplog):
        converter = DoclingServeConverter(api_key=None)

        with patch(
            "httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_mock_failed_result_response("async failure")
        ):
            with caplog.at_level(logging.WARNING):
                result = await converter.run_async(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "async failure" in caplog.text

    @pytest.mark.asyncio
    async def test_run_async_skips_on_status_error(self, caplog):
        converter = DoclingServeConverter(api_key=None)
        error_response = httpx.Response(500, text="error", request=httpx.Request("POST", "http://test"))

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError("err", request=error_response.request, response=error_response),
        ):
            with caplog.at_level(logging.WARNING):
                result = await converter.run_async(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "HTTP 500" in caplog.text

    @pytest.mark.asyncio
    async def test_run_async_skips_on_connection_error(self, caplog):
        converter = DoclingServeConverter(api_key=None)

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("timeout"),
        ):
            with caplog.at_level(logging.WARNING):
                result = await converter.run_async(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "Could not connect" in caplog.text

    @pytest.mark.asyncio
    async def test_run_async_skips_missing_local_file(self, tmp_path, caplog):
        missing = tmp_path / "missing.pdf"
        good = tmp_path / "good.pdf"
        good.write_bytes(b"%PDF-test")
        converter = DoclingServeConverter(api_key=None)
        mock_resp = httpx.Response(
            200,
            json={"document": {"md_content": "# Good content"}, "status": "success"},
            request=httpx.Request("POST", "http://test"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with caplog.at_level(logging.WARNING):
                result = await converter.run_async(sources=[missing, good])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "# Good content"
        assert "Could not read local source" in caplog.text

    @pytest.mark.asyncio
    async def test_run_async_multiple_sources(self):
        converter = DoclingServeConverter(api_key=None)
        mock_resp = httpx.Response(
            200,
            json={"document": {"md_content": "content"}, "status": "success"},
            request=httpx.Request("POST", "http://test"),
        )

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await converter.run_async(sources=["https://a.com/1.pdf", "https://b.com/2.pdf"])

        assert len(result["documents"]) == 2

    @pytest.mark.asyncio
    async def test_run_async_async_mode_uses_job_endpoints(self):
        converter = DoclingServeConverter(api_key=None, mode=ConversionMode.ASYNC, poll_interval=1.5)

        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_mock_task_response()) as mock_post,
            patch(
                "httpx.AsyncClient.get",
                new_callable=AsyncMock,
                side_effect=[_mock_status_response(), _mock_httpx_response("async content")],
            ) as mock_get,
        ):
            result = await converter.run_async(sources=["https://example.com/doc.pdf"])

        assert result["documents"][0].content == "async content"

        post_url = mock_post.call_args[0][0]
        payload = mock_post.call_args[1]["json"]
        assert post_url.endswith("/v1/convert/source/async")
        assert payload["target"] == {"kind": "inbody"}

        status_call, result_call = mock_get.call_args_list
        assert status_call.args[0].endswith("/v1/status/poll/task-1")
        assert status_call.kwargs["params"] == {"wait": 1.5}
        assert result_call.args[0].endswith("/v1/result/task-1")

    @pytest.mark.asyncio
    async def test_run_async_async_mode_uses_job_endpoints_for_file_source(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-test")
        converter = DoclingServeConverter(api_key=None, mode=ConversionMode.ASYNC)

        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_mock_task_response()) as mock_post,
            patch(
                "httpx.AsyncClient.get",
                new_callable=AsyncMock,
                side_effect=[_mock_status_response(), _mock_httpx_response("async file content")],
            ),
        ):
            result = await converter.run_async(sources=[pdf])

        post_url = mock_post.call_args[0][0]
        data = mock_post.call_args[1]["data"]
        assert post_url.endswith("/v1/convert/file/async")
        assert data["target_type"] == "inbody"
        assert result["documents"][0].content == "async file content"

    @pytest.mark.asyncio
    async def test_run_async_async_mode_skips_failed_result(self, caplog):
        converter = DoclingServeConverter(api_key=None, mode=ConversionMode.ASYNC)

        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_mock_task_response()),
            patch(
                "httpx.AsyncClient.get",
                new_callable=AsyncMock,
                side_effect=[_mock_status_response(), _mock_failed_result_response("bad async result")],
            ),
        ):
            with caplog.at_level(logging.WARNING):
                result = await converter.run_async(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "bad async result" in caplog.text

    @pytest.mark.asyncio
    async def test_run_async_async_mode_skips_timed_out_job(self, caplog):
        converter = DoclingServeConverter(
            api_key=None, mode=ConversionMode.ASYNC, poll_interval=0.001, job_timeout=0.001
        )

        with (
            patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_mock_task_response()),
            patch(
                "httpx.AsyncClient.get",
                new_callable=AsyncMock,
                return_value=_mock_status_response(status="pending"),
            ),
        ):
            with caplog.at_level(logging.WARNING):
                result = await converter.run_async(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "Timed out" in caplog.text


class TestDoclingServeConverterIntegration:
    @pytest.mark.integration
    def test_run_integration(self):
        """Requires a running DoclingServe instance at http://localhost:5001."""
        converter = DoclingServeConverter(api_key=None)
        result = converter.run(sources=["https://arxiv.org/pdf/2206.01062"])
        assert len(result["documents"]) > 0
        assert result["documents"][0].content

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_integration(self):
        """Requires a running DoclingServe instance at http://localhost:5001."""
        converter = DoclingServeConverter(api_key=None)
        result = await converter.run_async(sources=["https://arxiv.org/pdf/2206.01062"])
        assert len(result["documents"]) > 0
        assert result["documents"][0].content
