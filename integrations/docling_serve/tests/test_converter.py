# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.components.converters.docling_serve import DoclingServeConverter, ExportType


def _mock_response(content: str, export_type: ExportType = ExportType.MARKDOWN) -> MagicMock:
    content_key = {"markdown": "md_content", "text": "text_content", "json": "json_content"}[export_type.value]
    mock = MagicMock()
    mock.json.return_value = {"document": {content_key: content}, "status": "success"}
    mock.raise_for_status = MagicMock()
    return mock


class TestDoclingServeConverterInit:
    def test_defaults(self):
        converter = DoclingServeConverter()
        assert converter.base_url == "http://localhost:5001"
        assert converter.export_type == ExportType.MARKDOWN
        assert converter.convert_options == {}
        assert converter.timeout == 120.0
        assert converter.api_key is None

    def test_custom_params(self):
        converter = DoclingServeConverter(
            base_url="http://myserver:8080/",
            export_type=ExportType.TEXT,
            convert_options={"do_ocr": True},
            timeout=60.0,
        )
        assert converter.base_url == "http://myserver:8080"  # trailing slash stripped
        assert converter.export_type == ExportType.TEXT
        assert converter.convert_options == {"do_ocr": True}
        assert converter.timeout == 60.0

    def test_trailing_slash_stripped(self):
        converter = DoclingServeConverter(base_url="http://localhost:5001/")
        assert converter.base_url == "http://localhost:5001"


class TestDoclingServeConverterSerialization:
    def test_to_dict(self):
        converter = DoclingServeConverter(
            base_url="http://localhost:5001",
            export_type=ExportType.TEXT,
            convert_options={"do_ocr": False},
            timeout=30.0,
        )
        data = converter.to_dict()
        assert data["type"].endswith("DoclingServeConverter")
        params = data["init_parameters"]
        assert params["base_url"] == "http://localhost:5001"
        assert params["export_type"] == "text"
        assert params["convert_options"] == {"do_ocr": False}
        assert params["timeout"] == 30.0
        assert params["api_key"] is None

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.converters.docling_serve.converter.DoclingServeConverter",
            "init_parameters": {
                "base_url": "http://myserver:9000",
                "export_type": "json",
                "convert_options": {},
                "timeout": 60.0,
                "api_key": None,
            },
        }
        converter = DoclingServeConverter.from_dict(data)
        assert converter.base_url == "http://myserver:9000"
        assert converter.export_type == ExportType.JSON
        assert converter.timeout == 60.0

    def test_to_dict_with_api_key(self, monkeypatch):
        monkeypatch.setenv("DOCLING_API_KEY", "test-key")
        converter = DoclingServeConverter(api_key=Secret.from_env_var("DOCLING_API_KEY"))
        data = converter.to_dict()
        assert data["init_parameters"]["api_key"] is not None
        assert data["init_parameters"]["api_key"]["type"] == "env_var"

    def test_roundtrip(self):
        converter = DoclingServeConverter(
            base_url="http://remote:5001",
            export_type=ExportType.MARKDOWN,
            convert_options={"table_mode": "fast"},
            timeout=45.0,
        )
        data = converter.to_dict()
        restored = DoclingServeConverter.from_dict(data)
        assert restored.base_url == converter.base_url
        assert restored.export_type == converter.export_type
        assert restored.convert_options == converter.convert_options
        assert restored.timeout == converter.timeout


class TestDoclingServeConverterPayload:
    def test_payload_markdown(self):
        converter = DoclingServeConverter(export_type=ExportType.MARKDOWN)
        entry = {"kind": "http", "url": "https://example.com/doc.pdf"}
        payload = converter._build_payload(entry)
        assert payload["options"]["to_formats"] == ["md"]
        assert payload["sources"] == [entry]

    def test_payload_text(self):
        converter = DoclingServeConverter(export_type=ExportType.TEXT)
        entry = {"kind": "http", "url": "https://example.com/doc.pdf"}
        payload = converter._build_payload(entry)
        assert payload["options"]["to_formats"] == ["text"]

    def test_payload_merges_convert_options(self):
        converter = DoclingServeConverter(convert_options={"do_ocr": True, "table_mode": "accurate"})
        entry = {"kind": "http", "url": "https://example.com/doc.pdf"}
        payload = converter._build_payload(entry)
        assert payload["options"]["do_ocr"] is True
        assert payload["options"]["table_mode"] == "accurate"
        assert payload["options"]["to_formats"] == ["md"]

    def test_source_entry_http_url(self):
        converter = DoclingServeConverter()
        entry, extra = converter._source_entry("https://example.com/file.pdf")
        assert entry["kind"] == "http"
        assert entry["url"] == "https://example.com/file.pdf"
        assert extra == {}

    def test_source_entry_file_path(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-test")
        converter = DoclingServeConverter()
        entry, extra = converter._source_entry(pdf)
        assert entry["kind"] == "file"
        assert entry["filename"] == "test.pdf"
        assert "base64_string" in entry
        assert extra == {}

    def test_source_entry_bytestream(self):
        converter = DoclingServeConverter()
        bs = ByteStream(data=b"some bytes", meta={"file_name": "report.pdf", "source": "s3"})
        entry, extra = converter._source_entry(bs)
        assert entry["kind"] == "file"
        assert entry["filename"] == "report.pdf"
        assert "base64_string" in entry
        assert extra == {"file_name": "report.pdf", "source": "s3"}

    def test_source_entry_bytestream_no_meta(self):
        converter = DoclingServeConverter()
        bs = ByteStream(data=b"bytes")
        entry, _ = converter._source_entry(bs)
        assert entry["filename"] == "document"


class TestDoclingServeConverterRun:
    def test_run_url_source(self):
        converter = DoclingServeConverter()
        mock_resp = _mock_response("# Hello World")

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = converter.run(sources=["https://example.com/doc.pdf"])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "# Hello World"

    def test_run_multiple_sources(self):
        converter = DoclingServeConverter()
        mock_resp = _mock_response("content")

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = converter.run(sources=["https://a.com/1.pdf", "https://b.com/2.pdf"])

        assert len(result["documents"]) == 2

    def test_run_with_meta(self):
        converter = DoclingServeConverter()
        mock_resp = _mock_response("text")

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = converter.run(sources=["https://example.com/doc.pdf"], meta={"author": "Alice"})

        assert result["documents"][0].meta["author"] == "Alice"

    def test_run_bytestream_meta_merged(self):
        converter = DoclingServeConverter()
        mock_resp = _mock_response("text")
        bs = ByteStream(data=b"bytes", meta={"file_path": "doc.pdf"})

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = converter.run(sources=[bs], meta={"page": 1})

        doc = result["documents"][0]
        assert doc.meta["file_path"] == "doc.pdf"
        assert doc.meta["page"] == 1

    def test_run_skips_on_http_error(self, caplog):
        converter = DoclingServeConverter()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.side_effect = Exception("Connection refused")
            mock_client_cls.return_value = mock_client

            with caplog.at_level(logging.WARNING):
                result = converter.run(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "Could not convert source" in caplog.text

    def test_run_skips_when_no_content(self, caplog):
        converter = DoclingServeConverter()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"document": {"md_content": None}, "status": "success"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            with caplog.at_level(logging.WARNING):
                result = converter.run(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "No content returned" in caplog.text

    def test_run_text_export(self):
        converter = DoclingServeConverter(export_type=ExportType.TEXT)
        mock_resp = _mock_response("plain text content", ExportType.TEXT)

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = converter.run(sources=["https://example.com/doc.pdf"])

        assert result["documents"][0].content == "plain text content"

    def test_run_json_export(self):
        converter = DoclingServeConverter(export_type=ExportType.JSON)
        json_doc = {"schema_name": "DoclingDocument", "pages": []}
        mock_resp = _mock_response(json_doc, ExportType.JSON)  # type: ignore[arg-type]

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = converter.run(sources=["https://example.com/doc.pdf"])

        content = json.loads(result["documents"][0].content)
        assert content["schema_name"] == "DoclingDocument"

    def test_run_sends_api_key_header(self):
        converter = DoclingServeConverter(api_key=Secret.from_token("my-secret-token"))
        mock_resp = _mock_response("content")

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            converter.run(sources=["https://example.com/doc.pdf"])

        _, kwargs = mock_client.post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer my-secret-token"

    def test_run_file_path(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-test")
        converter = DoclingServeConverter()
        mock_resp = _mock_response("# PDF content")

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.post.return_value = mock_resp
            mock_client_cls.return_value = mock_client

            result = converter.run(sources=[pdf])

        payload = mock_client.post.call_args[1]["json"]
        assert payload["sources"][0]["kind"] == "file"
        assert payload["sources"][0]["filename"] == "test.pdf"
        assert result["documents"][0].content == "# PDF content"


class TestDoclingServeConverterArun:
    @pytest.mark.asyncio
    async def test_arun_url_source(self):
        converter = DoclingServeConverter()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"document": {"md_content": "# Async content"}, "status": "success"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await converter.arun(sources=["https://example.com/doc.pdf"])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "# Async content"

    @pytest.mark.asyncio
    async def test_arun_skips_on_error(self, caplog):
        converter = DoclingServeConverter()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=Exception("timeout"))
            mock_client_cls.return_value = mock_client

            with caplog.at_level(logging.WARNING):
                result = await converter.arun(sources=["https://example.com/doc.pdf"])

        assert result["documents"] == []
        assert "Could not convert source" in caplog.text

    @pytest.mark.asyncio
    async def test_arun_multiple_sources(self):
        converter = DoclingServeConverter()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"document": {"md_content": "content"}, "status": "success"}
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await converter.arun(sources=["https://a.com/1.pdf", "https://b.com/2.pdf"])

        assert len(result["documents"]) == 2


class TestDoclingServeConverterIntegration:
    @pytest.mark.integration
    def test_run_integration(self):
        """Requires a running DoclingServe instance at http://localhost:5001."""
        converter = DoclingServeConverter()
        result = converter.run(sources=["https://arxiv.org/pdf/2206.01062"])
        assert len(result["documents"]) > 0
        assert result["documents"][0].content

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_arun_integration(self):
        """Requires a running DoclingServe instance at http://localhost:5001."""
        converter = DoclingServeConverter()
        result = await converter.arun(sources=["https://arxiv.org/pdf/2206.01062"])
        assert len(result["documents"]) > 0
        assert result["documents"][0].content
