# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import httpx
import pytest
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.components.converters.docling_serve import DoclingServeConverter

SAMPLE_RESPONSE = {
    "document": {
        "md_content": "# Sample Document\n\nThis is the content.",
    },
    "status": "success",
    "processing_time": 1.23,
    "errors": [],
    "timings": {},
}


def _mock_response(json_data=None, status_code=200):
    """Create a mock httpx.Response."""
    if json_data is None:
        json_data = SAMPLE_RESPONSE
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", "http://test"),
    )


class TestInit:
    def test_defaults(self):
        converter = DoclingServeConverter()
        assert converter.base_url == "http://localhost:5001"
        assert converter.timeout == 300
        assert converter.convert_options == {}
        assert converter.api_key is not None

    def test_custom_params(self):
        converter = DoclingServeConverter(
            base_url="http://myserver:8080",
            api_key=Secret.from_token("test-key"),
            timeout=60,
            convert_options={"from_formats": ["pdf"], "do_ocr": True},
        )
        assert converter.base_url == "http://myserver:8080"
        assert converter.api_key.resolve_value() == "test-key"
        assert converter.timeout == 60
        assert converter.convert_options == {"from_formats": ["pdf"], "do_ocr": True}

    def test_api_key_none(self):
        converter = DoclingServeConverter(api_key=None)
        assert converter.api_key is None


class TestSerialization:
    def test_to_dict(self):
        converter = DoclingServeConverter(
            base_url="http://myserver:8080",
            api_key=Secret.from_env_var("MY_KEY"),
            timeout=60,
            convert_options={"from_formats": ["pdf"]},
        )
        result = converter.to_dict()
        assert result["type"] == (
            "haystack_integrations.components.converters.docling_serve.converter.DoclingServeConverter"
        )
        assert result["init_parameters"]["base_url"] == "http://myserver:8080"
        assert result["init_parameters"]["timeout"] == 60
        assert result["init_parameters"]["convert_options"] == {"from_formats": ["pdf"]}
        assert result["init_parameters"]["api_key"] == {"type": "env_var", "env_vars": ["MY_KEY"], "strict": True}

    def test_to_dict_no_api_key(self):
        converter = DoclingServeConverter(api_key=None)
        result = converter.to_dict()
        assert result["init_parameters"]["api_key"] is None

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.converters.docling_serve.converter.DoclingServeConverter",
            "init_parameters": {
                "base_url": "http://myserver:8080",
                "api_key": {"type": "env_var", "env_vars": ["MY_KEY"], "strict": True},
                "timeout": 60,
                "convert_options": {"do_ocr": False},
            },
        }
        converter = DoclingServeConverter.from_dict(data)
        assert converter.base_url == "http://myserver:8080"
        assert converter.timeout == 60
        assert converter.convert_options == {"do_ocr": False}
        assert isinstance(converter.api_key, Secret)

    def test_from_dict_no_api_key(self):
        data = {
            "type": "haystack_integrations.components.converters.docling_serve.converter.DoclingServeConverter",
            "init_parameters": {
                "base_url": "http://localhost:5001",
                "api_key": None,
                "timeout": 300,
                "convert_options": {},
            },
        }
        converter = DoclingServeConverter.from_dict(data)
        assert converter.api_key is None

    def test_round_trip(self):
        converter = DoclingServeConverter(
            base_url="http://myserver:8080",
            api_key=Secret.from_env_var("MY_KEY"),
            timeout=120,
            convert_options={"from_formats": ["pdf", "docx"], "to_formats": ["md"]},
        )
        data = converter.to_dict()
        restored = DoclingServeConverter.from_dict(data)
        assert restored.base_url == converter.base_url
        assert restored.timeout == converter.timeout
        assert restored.convert_options == converter.convert_options


class TestRunWithFilePath:
    def test_converts_file(self, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 fake content")

        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.Client.post", return_value=_mock_response()) as mock_post:
            result = converter.run(sources=[str(test_file)])

        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert doc.content == "# Sample Document\n\nThis is the content."
        assert doc.meta["source_file"] == str(test_file)
        assert doc.meta["conversion_status"] == "success"
        assert doc.meta["processing_time"] == 1.23

        # Verify the request was made to the file endpoint
        call_kwargs = mock_post.call_args
        assert "/v1/convert/file" in call_kwargs.args[0]

    def test_converts_path_object(self, tmp_path):
        test_file = tmp_path / "doc.docx"
        test_file.write_bytes(b"fake docx")

        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.Client.post", return_value=_mock_response()):
            result = converter.run(sources=[test_file])

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["source_file"] == str(test_file)


class TestRunWithByteStream:
    def test_converts_bytestream(self):
        bs = ByteStream(data=b"fake pdf content", meta={"file_name": "report.pdf"}, mime_type="application/pdf")
        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.Client.post", return_value=_mock_response()) as mock_post:
            result = converter.run(sources=[bs])

        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert doc.content == "# Sample Document\n\nThis is the content."
        assert doc.meta["source_file"] == "report.pdf"
        assert doc.meta["file_name"] == "report.pdf"

        call_kwargs = mock_post.call_args
        assert "/v1/convert/file" in call_kwargs.args[0]

    def test_bytestream_meta_merged(self):
        bs = ByteStream(data=b"content", meta={"file_name": "doc.pdf", "custom_key": "custom_value"})
        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.Client.post", return_value=_mock_response()):
            result = converter.run(sources=[bs], meta={"user_key": "user_value"})

        doc = result["documents"][0]
        assert doc.meta["custom_key"] == "custom_value"
        assert doc.meta["user_key"] == "user_value"


class TestRunWithURL:
    def test_converts_url(self):
        converter = DoclingServeConverter(api_key=None)
        url = "https://example.com/document.pdf"

        with patch("httpx.Client.post", return_value=_mock_response()) as mock_post:
            result = converter.run(sources=[url])

        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert doc.meta["source_file"] == url

        call_kwargs = mock_post.call_args
        assert "/v1/convert/source" in call_kwargs.args[0]
        body = call_kwargs.kwargs["json"]
        assert body["http_sources"] == [{"url": url}]

    def test_http_url_detected(self):
        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.Client.post", return_value=_mock_response()) as mock_post:
            converter.run(sources=["http://example.com/doc.pdf"])

        call_kwargs = mock_post.call_args
        assert "/v1/convert/source" in call_kwargs.args[0]


class TestRunWithMeta:
    def test_single_dict_meta(self, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"content")

        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.Client.post", return_value=_mock_response()):
            result = converter.run(sources=[str(test_file)], meta={"category": "report"})

        assert result["documents"][0].meta["category"] == "report"

    def test_list_of_dicts_meta(self, tmp_path):
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"a")
        f2.write_bytes(b"b")

        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.Client.post", return_value=_mock_response()):
            result = converter.run(
                sources=[str(f1), str(f2)],
                meta=[{"category": "report"}, {"category": "invoice"}],
            )

        assert result["documents"][0].meta["category"] == "report"
        assert result["documents"][1].meta["category"] == "invoice"

    def test_meta_list_length_mismatch_raises(self, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"content")

        converter = DoclingServeConverter(api_key=None)

        with pytest.raises(ValueError, match="metadata"):
            converter.run(sources=[str(test_file)], meta=[{"a": 1}, {"b": 2}])


class TestRunWithConvertOptions:
    def test_options_passed_to_file_endpoint(self, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"content")

        converter = DoclingServeConverter(
            api_key=None,
            convert_options={"from_formats": ["pdf"], "do_ocr": True},
        )

        with patch("httpx.Client.post", return_value=_mock_response()) as mock_post:
            converter.run(sources=[str(test_file)])

        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["data"] == {"from_formats": ["pdf"], "do_ocr": True}

    def test_options_passed_to_source_endpoint(self):
        converter = DoclingServeConverter(
            api_key=None,
            convert_options={"to_formats": ["json"], "do_ocr": False},
        )

        with patch("httpx.Client.post", return_value=_mock_response()) as mock_post:
            converter.run(sources=["https://example.com/doc.pdf"])

        call_kwargs = mock_post.call_args
        body = call_kwargs.kwargs["json"]
        assert body["options"] == {"to_formats": ["json"], "do_ocr": False}


class TestRunWithAuth:
    def test_api_key_sent_in_headers(self, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"content")

        converter = DoclingServeConverter(api_key=Secret.from_token("my-secret-key"))

        with patch("httpx.Client.post", return_value=_mock_response()) as mock_post:
            converter.run(sources=[str(test_file)])

        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs["headers"]
        assert headers["X-Api-Key"] == "my-secret-key"

    def test_no_api_key_header_when_none(self, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"content")

        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.Client.post", return_value=_mock_response()) as mock_post:
            converter.run(sources=[str(test_file)])

        call_kwargs = mock_post.call_args
        headers = call_kwargs.kwargs["headers"]
        assert "X-Api-Key" not in headers


class TestRunErrorHandling:
    def test_http_error_logged_not_raised(self, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"content")

        converter = DoclingServeConverter(api_key=None)

        error_response = httpx.Response(
            status_code=500,
            text="Internal Server Error",
            request=httpx.Request("POST", "http://test"),
        )

        with patch("httpx.Client.post", return_value=error_response):
            result = converter.run(sources=[str(test_file)])

        # Should return empty list, not raise
        assert result["documents"] == []

    def test_connection_error_logged_not_raised(self, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"content")

        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.Client.post", side_effect=httpx.ConnectError("Connection refused")):
            result = converter.run(sources=[str(test_file)])

        assert result["documents"] == []

    def test_partial_failure(self, tmp_path):
        """When one source fails, others should still be converted."""
        good_file = tmp_path / "good.pdf"
        bad_file = tmp_path / "bad.pdf"
        good_file.write_bytes(b"good")
        bad_file.write_bytes(b"bad")

        converter = DoclingServeConverter(api_key=None)

        call_count = 0

        def mock_post(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    status_code=500,
                    text="Error",
                    request=httpx.Request("POST", "http://test"),
                )
            return _mock_response()

        with patch("httpx.Client.post", side_effect=mock_post):
            result = converter.run(sources=[str(bad_file), str(good_file)])

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["source_file"] == str(good_file)


class TestRunWithTextContentFallback:
    def test_falls_back_to_text_content(self, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"content")

        response_data = {
            "document": {
                "md_content": None,
                "text_content": "Plain text fallback.",
            },
            "status": "success",
            "processing_time": 0.5,
        }

        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.Client.post", return_value=_mock_response(response_data)):
            result = converter.run(sources=[str(test_file)])

        assert result["documents"][0].content == "Plain text fallback."

    def test_empty_string_when_no_content(self, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"content")

        response_data = {
            "document": {},
            "status": "success",
            "processing_time": 0.1,
        }

        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.Client.post", return_value=_mock_response(response_data)):
            result = converter.run(sources=[str(test_file)])

        assert result["documents"][0].content == ""


class TestRunAsync:
    @pytest.mark.asyncio
    async def test_converts_file(self, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"content")

        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.AsyncClient.post", return_value=_mock_response()) as mock_post:
            result = await converter.run_async(sources=[str(test_file)])

        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert doc.content == "# Sample Document\n\nThis is the content."
        assert doc.meta["conversion_status"] == "success"

        call_kwargs = mock_post.call_args
        assert "/v1/convert/file" in call_kwargs.args[0]

    @pytest.mark.asyncio
    async def test_converts_url(self):
        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.AsyncClient.post", return_value=_mock_response()) as mock_post:
            result = await converter.run_async(sources=["https://example.com/doc.pdf"])

        assert len(result["documents"]) == 1

        call_kwargs = mock_post.call_args
        assert "/v1/convert/source" in call_kwargs.args[0]

    @pytest.mark.asyncio
    async def test_error_handling(self, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"content")

        converter = DoclingServeConverter(api_key=None)

        with patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")):
            result = await converter.run_async(sources=[str(test_file)])

        assert result["documents"] == []


class TestMixedSources:
    def test_file_and_url_together(self, tmp_path):
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"content")

        converter = DoclingServeConverter(api_key=None)

        call_count = 0

        def mock_post(_url, **_kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_response()

        with patch("httpx.Client.post", side_effect=mock_post):
            result = converter.run(
                sources=[str(test_file), "https://example.com/doc.pdf"],
                meta=[{"type": "local"}, {"type": "remote"}],
            )

        assert len(result["documents"]) == 2
        assert result["documents"][0].meta["type"] == "local"
        assert result["documents"][1].meta["type"] == "remote"


@pytest.mark.integration
class TestIntegration:
    @pytest.mark.skipif(
        not os.environ.get("DOCLING_SERVE_URL"),
        reason="Set DOCLING_SERVE_URL to run integration tests (e.g. http://localhost:5001)",
    )
    def test_convert_file(self, tmp_path):
        """Convert a simple text file against a running docling-serve instance."""
        test_file = tmp_path / "hello.md"
        test_file.write_text("# Hello\n\nThis is a test document.")

        url = os.environ["DOCLING_SERVE_URL"]
        converter = DoclingServeConverter(base_url=url, api_key=None)
        result = converter.run(sources=[str(test_file)])

        assert len(result["documents"]) == 1
        doc = result["documents"][0]
        assert doc.content
        assert doc.meta["conversion_status"] == "success"
        assert doc.meta["processing_time"] > 0
