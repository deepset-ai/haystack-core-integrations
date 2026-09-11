# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.components.audio.whisper import RemoteWhisperTranscriber


class TestInitialization:
    def test_init_default(self):
        transcriber = RemoteWhisperTranscriber(api_key=Secret.from_token("test_api_key"))
        assert transcriber.client is None
        assert transcriber.async_client is None
        assert transcriber.model == "whisper-1"
        assert transcriber.organization is None
        assert transcriber.whisper_params == {"response_format": "json"}

    def test_init_custom_parameters(self):
        transcriber = RemoteWhisperTranscriber(
            api_key=Secret.from_token("test_api_key"),
            model="whisper-1",
            organization="test-org",
            api_base_url="test_api_url",
            language="en",
            prompt="test-prompt",
            response_format="json",
            temperature="0.5",
        )

        assert transcriber.model == "whisper-1"
        assert transcriber.api_key == Secret.from_token("test_api_key")
        assert transcriber.organization == "test-org"
        assert transcriber.api_base_url == "test_api_url"
        assert transcriber.whisper_params == {
            "language": "en",
            "prompt": "test-prompt",
            "response_format": "json",
            "temperature": "0.5",
        }

    def test_init_overwrites_non_json_response_format(self):
        # Only `response_format="json"` is supported; any other value is overwritten.
        transcriber = RemoteWhisperTranscriber(response_format="text")
        assert transcriber.whisper_params["response_format"] == "json"


class TestSerialization:
    def test_to_dict_default_parameters(self):
        transcriber = RemoteWhisperTranscriber()
        data = transcriber.to_dict()
        assert data == {
            "type": "haystack_integrations.components.audio.whisper.whisper_remote.RemoteWhisperTranscriber",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "whisper-1",
                "api_base_url": None,
                "organization": None,
                "http_client_kwargs": None,
                "response_format": "json",
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        transcriber = RemoteWhisperTranscriber(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="whisper-1",
            organization="test-org",
            api_base_url="test_api_url",
            http_client_kwargs={"proxy": "http://localhost:8080"},
            language="en",
            prompt="test-prompt",
            response_format="json",
            temperature="0.5",
        )
        data = transcriber.to_dict()
        assert data == {
            "type": "haystack_integrations.components.audio.whisper.whisper_remote.RemoteWhisperTranscriber",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "whisper-1",
                "organization": "test-org",
                "api_base_url": "test_api_url",
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
                "language": "en",
                "prompt": "test-prompt",
                "response_format": "json",
                "temperature": "0.5",
            },
        }

    def test_from_dict_with_default_parameters(self):
        data = {
            "type": "haystack_integrations.components.audio.whisper.whisper_remote.RemoteWhisperTranscriber",
            "init_parameters": {
                "model": "whisper-1",
                "api_base_url": "https://api.openai.com/v1",
                "organization": None,
                "http_client_kwargs": None,
                "response_format": "json",
            },
        }

        transcriber = RemoteWhisperTranscriber.from_dict(data)

        assert transcriber.model == "whisper-1"
        assert transcriber.organization is None
        assert transcriber.api_base_url == "https://api.openai.com/v1"
        assert transcriber.whisper_params == {"response_format": "json"}
        assert transcriber.http_client_kwargs is None

    def test_from_dict_with_custom_init_parameters(self):
        data = {
            "type": "haystack_integrations.components.audio.whisper.whisper_remote.RemoteWhisperTranscriber",
            "init_parameters": {
                "model": "whisper-1",
                "organization": "test-org",
                "api_base_url": "test_api_url",
                "language": "en",
                "prompt": "test-prompt",
                "response_format": "json",
                "temperature": "0.5",
            },
        }
        transcriber = RemoteWhisperTranscriber.from_dict(data)

        assert transcriber.model == "whisper-1"
        assert transcriber.organization == "test-org"
        assert transcriber.api_base_url == "test_api_url"
        assert transcriber.whisper_params == {
            "language": "en",
            "prompt": "test-prompt",
            "response_format": "json",
            "temperature": "0.5",
        }

    def test_from_dict_with_default_parameters_no_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        data = {
            "type": "haystack_integrations.components.audio.whisper.whisper_remote.RemoteWhisperTranscriber",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "whisper-1",
                "api_base_url": "https://api.openai.com/v1",
                "organization": None,
                "response_format": "json",
            },
        }

        transcriber = RemoteWhisperTranscriber.from_dict(data)
        assert transcriber.client is None
        assert transcriber.async_client is None


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        transcriber = RemoteWhisperTranscriber()

        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            transcriber.warm_up()

    def test_sync_lifecycle(self):
        with (
            patch("haystack_integrations.components.audio.whisper.whisper_remote.OpenAI") as mock_client_cls,
            patch("haystack_integrations.components.audio.whisper.whisper_remote.init_http_client"),
        ):
            transcriber = RemoteWhisperTranscriber(api_key=Secret.from_token("test_api_key"))
            client = mock_client_cls.return_value

            transcriber.warm_up()
            assert transcriber.client is client
            assert transcriber.async_client is None

            transcriber.close()
            client.close.assert_called_once_with()
            assert transcriber.client is None

            transcriber.warm_up()
            assert mock_client_cls.call_count == 2

    @pytest.mark.asyncio
    async def test_async_lifecycle(self):
        with (
            patch("haystack_integrations.components.audio.whisper.whisper_remote.AsyncOpenAI") as mock_client_cls,
            patch("haystack_integrations.components.audio.whisper.whisper_remote.init_http_client"),
        ):
            transcriber = RemoteWhisperTranscriber(api_key=Secret.from_token("test_api_key"))
            client = mock_client_cls.return_value
            client.close = AsyncMock()

            await transcriber.warm_up_async()
            assert transcriber.client is None
            assert transcriber.async_client is client

            await transcriber.close_async()
            client.close.assert_awaited_once_with()
            assert transcriber.async_client is None

            await transcriber.warm_up_async()
            assert mock_client_cls.call_count == 2

    def test_warm_up_is_idempotent(self):
        with (
            patch("haystack_integrations.components.audio.whisper.whisper_remote.OpenAI") as mock_openai,
            patch("haystack_integrations.components.audio.whisper.whisper_remote.init_http_client"),
        ):
            transcriber = RemoteWhisperTranscriber(api_key=Secret.from_token("test_api_key"))
            transcriber.warm_up()
            transcriber.warm_up()

        mock_openai.assert_called_once()

    @pytest.mark.asyncio
    async def test_warm_up_async_is_idempotent(self):
        with (
            patch("haystack_integrations.components.audio.whisper.whisper_remote.AsyncOpenAI") as mock_async_openai,
            patch("haystack_integrations.components.audio.whisper.whisper_remote.init_http_client"),
        ):
            transcriber = RemoteWhisperTranscriber(api_key=Secret.from_token("test_api_key"))
            await transcriber.warm_up_async()
            await transcriber.warm_up_async()

        mock_async_openai.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_is_safe_without_warm_up(self):
        transcriber = RemoteWhisperTranscriber(api_key=Secret.from_token("test_api_key"))

        transcriber.close()
        await transcriber.close_async()

        assert transcriber.client is None
        assert transcriber.async_client is None

    @pytest.mark.asyncio
    async def test_close_and_close_async_are_independent(self):
        sync_client = Mock()
        async_client = Mock()
        async_client.close = AsyncMock()
        transcriber = RemoteWhisperTranscriber(api_key=Secret.from_token("test_api_key"))
        transcriber.client = sync_client
        transcriber.async_client = async_client

        transcriber.close()

        sync_client.close.assert_called_once_with()
        async_client.close.assert_not_awaited()
        assert transcriber.client is None
        assert transcriber.async_client is async_client

        await transcriber.close_async()

        async_client.close.assert_awaited_once_with()
        assert transcriber.async_client is None


class TestRun:
    def test_run_with_path(self, test_files_path):
        transcriber = RemoteWhisperTranscriber()

        mock_client = Mock()
        mock_client.audio.transcriptions.create = Mock(return_value=Mock(text="this is the content of the document."))
        transcriber.client = mock_client

        audio_path = str(test_files_path / "audio" / "this is the content of the document.wav")
        output = transcriber.run(sources=[audio_path])

        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "this is the content of the document."
        assert docs[0].meta["file_path"] == audio_path
        mock_client.audio.transcriptions.create.assert_called_once()

    def test_run_with_bytestream(self):
        transcriber = RemoteWhisperTranscriber()

        mock_client = Mock()
        mock_client.audio.transcriptions.create = Mock(return_value=Mock(text="answer."))
        transcriber.client = mock_client

        source = ByteStream(data=b"fake audio bytes")
        source.meta["file_path"] = "answer.wav"
        output = transcriber.run(sources=[source])

        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "answer."
        assert docs[0].meta["file_path"] == "answer.wav"
        mock_client.audio.transcriptions.create.assert_called_once()


class TestRunAsync:
    @pytest.mark.asyncio
    async def test_run_async_with_path(self, test_files_path):
        transcriber = RemoteWhisperTranscriber()

        mock_client = Mock()
        mock_client.audio.transcriptions.create = AsyncMock(
            return_value=Mock(text="this is the content of the document.")
        )
        transcriber.async_client = mock_client

        audio_path = str(test_files_path / "audio" / "this is the content of the document.wav")
        output = await transcriber.run_async(sources=[audio_path])

        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "this is the content of the document."
        assert docs[0].meta["file_path"] == audio_path
        mock_client.audio.transcriptions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_async_with_bytestream(self):
        transcriber = RemoteWhisperTranscriber()

        mock_client = Mock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=Mock(text="answer."))
        transcriber.async_client = mock_client

        source = ByteStream(data=b"fake audio bytes")
        source.meta["file_path"] = "answer.wav"
        output = await transcriber.run_async(sources=[source])

        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "answer."
        assert docs[0].meta["file_path"] == "answer.wav"
        mock_client.audio.transcriptions.create.assert_awaited_once()


@pytest.mark.integration
class TestIntegration:
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_whisper_remote_transcriber(self, test_files_path):
        transcriber = RemoteWhisperTranscriber()

        paths = [
            test_files_path / "audio" / "this is the content of the document.wav",
            str(test_files_path / "audio" / "the context for this answer is here.wav"),
            ByteStream.from_file_path(test_files_path / "audio" / "answer.wav"),
        ]

        output = transcriber.run(sources=paths)

        docs = output["documents"]
        assert len(docs) == 3
        assert docs[0].content.strip().lower() == "this is the content of the document."
        assert test_files_path / "audio" / "this is the content of the document.wav" == docs[0].meta["file_path"]

        assert docs[1].content.strip().lower() == "the context for this answer is here."
        assert str(test_files_path / "audio" / "the context for this answer is here.wav") == docs[1].meta["file_path"]

        assert docs[2].content.strip().lower() == "answer."


@pytest.mark.integration
class TestAsyncIntegration:
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    async def test_whisper_remote_transcriber_async(self, test_files_path):
        transcriber = RemoteWhisperTranscriber()

        paths = [
            test_files_path / "audio" / "this is the content of the document.wav",
            str(test_files_path / "audio" / "the context for this answer is here.wav"),
            ByteStream.from_file_path(test_files_path / "audio" / "answer.wav"),
        ]

        output = await transcriber.run_async(sources=paths)

        docs = output["documents"]
        assert len(docs) == 3
        assert docs[0].content.strip().lower() == "this is the content of the document."
        assert test_files_path / "audio" / "this is the content of the document.wav" == docs[0].meta["file_path"]

        assert docs[1].content.strip().lower() == "the context for this answer is here."
        assert str(test_files_path / "audio" / "the context for this answer is here.wav") == docs[1].meta["file_path"]

        assert docs[2].content.strip().lower() == "answer."
