# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from haystack import Document
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.components.audio.gladia import GladiaTranscriber


class TestGladiaTranscriber:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GLADIA_API_KEY", "test-key")
        transcriber = GladiaTranscriber()
        assert transcriber.api_base_url == "https://api.gladia.io/v2"
        assert transcriber.polling_interval == 1.0
        assert transcriber.timeout == 300.0
        assert transcriber.gladia_params == {}

    def test_init_custom(self):
        transcriber = GladiaTranscriber(
            api_key=Secret.from_token("custom-key"),
            api_base_url="https://custom.gladia.io/v2/",
            polling_interval=0.5,
            timeout=60.0,
            gladia_params={"language": "english"},
        )
        assert transcriber.api_key.resolve_value() == "custom-key"
        assert transcriber.api_base_url == "https://custom.gladia.io/v2"
        assert transcriber.polling_interval == 0.5
        assert transcriber.timeout == 60.0
        assert transcriber.gladia_params == {"language": "english"}

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("GLADIA_API_KEY", "test-key")
        transcriber = GladiaTranscriber()
        dict_output = transcriber.to_dict()
        assert dict_output == {
            "type": "haystack_integrations.components.audio.gladia.transcriber.GladiaTranscriber",
            "init_parameters": {
                "api_key": {"env_vars": ["GLADIA_API_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": "https://api.gladia.io/v2",
                "polling_interval": 1.0,
                "timeout": 300.0,
                "gladia_params": {},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("GLADIA_API_KEY", "test-key")
        data = {
            "type": "haystack_integrations.components.audio.gladia.transcriber.GladiaTranscriber",
            "init_parameters": {
                "api_key": {"env_vars": ["GLADIA_API_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": "https://api.gladia.io/v2",
                "polling_interval": 0.5,
            },
        }
        transcriber = GladiaTranscriber.from_dict(data)
        assert transcriber.polling_interval == 0.5

    def test_run_empty(self):
        transcriber = GladiaTranscriber(api_key=Secret.from_token("key"))
        res = transcriber.run(sources=[])
        assert res == {"documents": []}

    @pytest.mark.asyncio
    async def test_run_async_empty(self):
        transcriber = GladiaTranscriber(api_key=Secret.from_token("key"))
        res = await transcriber.run_async(sources=[])
        assert res == {"documents": []}

    def test_run_url_success(self):
        transcriber = GladiaTranscriber(api_key=Secret.from_token("key"), polling_interval=0.01)

        mock_post_resp = Mock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "id": "job-123",
            "result_url": "https://api.gladia.io/v2/transcription/job-123",
        }

        mock_poll_resp = Mock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {
            "status": "done",
            "result": {"transcription": {"full_transcript": "Hello world transcription"}},
        }

        with patch("httpx.Client.post", return_value=mock_post_resp) as mock_post, patch(
            "httpx.Client.get", return_value=mock_poll_resp
        ) as mock_get:
            output = transcriber.run(sources=["https://example.com/audio.mp3"])

            mock_post.assert_called_once()
            mock_get.assert_called_once()
            docs = output["documents"]
            assert len(docs) == 1
            assert docs[0].content == "Hello world transcription"

    def test_run_file_success(self, tmp_path):
        transcriber = GladiaTranscriber(api_key=Secret.from_token("key"), polling_interval=0.01)
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"dummy audio content")

        mock_upload_resp = Mock()
        mock_upload_resp.status_code = 200
        mock_upload_resp.json.return_value = {"audio_url": "https://gladia-storage.com/uploaded.mp3"}

        mock_post_resp = Mock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "id": "job-456",
            "result_url": "https://api.gladia.io/v2/transcription/job-456",
        }

        mock_poll_resp = Mock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {
            "status": "done",
            "result": {"transcription": {"full_transcript": "Transcript from file"}},
        }

        with patch("httpx.Client.post", side_effect=[mock_upload_resp, mock_post_resp]), patch(
            "httpx.Client.get", return_value=mock_poll_resp
        ):
            output = transcriber.run(sources=[audio_file])
            docs = output["documents"]
            assert len(docs) == 1
            assert docs[0].content == "Transcript from file"

    def test_run_bytestream_success(self):
        transcriber = GladiaTranscriber(api_key=Secret.from_token("key"), polling_interval=0.01)
        bytestream = ByteStream(data=b"audio data", meta={"filename": "test.wav"})

        mock_upload_resp = Mock()
        mock_upload_resp.status_code = 200
        mock_upload_resp.json.return_value = {"audio_url": "https://gladia-storage.com/stream.wav"}

        mock_post_resp = Mock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "id": "job-789",
            "result_url": "https://api.gladia.io/v2/transcription/job-789",
        }

        mock_poll_resp = Mock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {
            "status": "done",
            "result": {"transcription": {"full_transcript": "Transcript from stream"}},
        }

        with patch("httpx.Client.post", side_effect=[mock_upload_resp, mock_post_resp]), patch(
            "httpx.Client.get", return_value=mock_poll_resp
        ):
            output = transcriber.run(sources=[bytestream])
            docs = output["documents"]
            assert len(docs) == 1
            assert docs[0].content == "Transcript from stream"

    @pytest.mark.asyncio
    async def test_run_async_url_success(self):
        transcriber = GladiaTranscriber(api_key=Secret.from_token("key"), polling_interval=0.01)

        mock_post_resp = Mock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "id": "job-async",
            "result_url": "https://api.gladia.io/v2/transcription/job-async",
        }

        mock_poll_resp = Mock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {
            "status": "done",
            "result": {"transcription": {"full_transcript": "Async transcript"}},
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_post_resp), patch(
            "httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_poll_resp
        ):
            output = await transcriber.run_async(sources=["https://example.com/audio.mp3"])
            docs = output["documents"]
            assert len(docs) == 1
            assert docs[0].content == "Async transcript"

    def test_run_error_status(self):
        transcriber = GladiaTranscriber(api_key=Secret.from_token("key"), polling_interval=0.01)

        mock_post_resp = Mock()
        mock_post_resp.status_code = 200
        mock_post_resp.json.return_value = {
            "id": "job-err",
            "result_url": "https://api.gladia.io/v2/transcription/job-err",
        }

        mock_poll_resp = Mock()
        mock_poll_resp.status_code = 200
        mock_poll_resp.json.return_value = {"status": "error", "error": "Corrupt audio format"}

        with (
            patch("httpx.Client.post", return_value=mock_post_resp),
            patch("httpx.Client.get", return_value=mock_poll_resp),
        ):
            with pytest.raises(RuntimeError, match="Gladia transcription failed: Corrupt audio format"):
                transcriber.run(sources=["https://example.com/audio.mp3"])

    @pytest.mark.skipif(
        not os.environ.get("GLADIA_API_KEY"),
        reason="Export GLADIA_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    def test_run_integration(self, test_files_path):
        transcriber = GladiaTranscriber(api_key=Secret.from_env_var("GLADIA_API_KEY"))
        audio_file = test_files_path / "audio" / "answer.wav"
        result = transcriber.run(sources=[audio_file])
        docs = result["documents"]
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].content

    @pytest.mark.skipif(
        not os.environ.get("GLADIA_API_KEY"),
        reason="Export GLADIA_API_KEY to run integration tests.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_integration(self, test_files_path):
        transcriber = GladiaTranscriber(api_key=Secret.from_env_var("GLADIA_API_KEY"))
        audio_file = test_files_path / "audio" / "answer.wav"
        result = await transcriber.run_async(sources=[audio_file])
        docs = result["documents"]
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].content
