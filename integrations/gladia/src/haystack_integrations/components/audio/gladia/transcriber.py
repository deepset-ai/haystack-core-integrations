# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import time
from pathlib import Path
from typing import Any, Sequence, Union

import httpx
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


@component
class GladiaTranscriber:
    """
    Transcribes audio files or audio URLs using Gladia's Batch Audio Transcription API v2.

    Usage example:
    ```python
    from haystack_integrations.components.audio.gladia import GladiaTranscriber

    transcriber = GladiaTranscriber()
    results = transcriber.run(sources=["sample.mp3"])
    print(results["documents"][0].content)
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("GLADIA_API_KEY"),
        api_base_url: str = "https://api.gladia.io/v2",
        polling_interval: float = 1.0,
        timeout: float = 300.0,
        gladia_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of GladiaTranscriber.

        :param api_key: Gladia API key.
        :param api_base_url: Gladia API base URL. Defaults to 'https://api.gladia.io/v2'.
        :param polling_interval: Interval in seconds between status checks during polling.
        :param timeout: Maximum total time in seconds to wait for a transcription job to complete.
        :param gladia_params: Additional parameters sent to the Gladia transcription endpoint
            (e.g., language, diarization settings).
        """
        self.api_key = api_key
        self.api_base_url = api_base_url.rstrip("/")
        self.polling_interval = polling_interval
        self.timeout = timeout
        self.gladia_params = gladia_params or {}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict() if self.api_key else None,
            api_base_url=self.api_base_url,
            polling_interval=self.polling_interval,
            timeout=self.timeout,
            gladia_params=self.gladia_params,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GladiaTranscriber":
        """
        Deserializes the component from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _get_headers(self) -> dict[str, str]:
        key = self.api_key.resolve_value() if self.api_key else None
        headers = {}
        if key:
            headers["X-Gladia-Key"] = key
        return headers

    def _upload_audio(self, client: httpx.Client, source: Union[str, Path, ByteStream]) -> str:
        headers = self._get_headers()
        url = f"{self.api_base_url}/upload"

        if isinstance(source, (str, Path)):
            path_str = str(source)
            if path_str.startswith(("http://", "https://")):
                return path_str
            file_path = Path(path_str)
            with open(file_path, "rb") as f:
                files = {"audio": (file_path.name, f.read())}
        elif isinstance(source, ByteStream):
            filename = source.meta.get("filename", "audio.wav")
            files = {"audio": (filename, source.data)}
        else:
            msg = f"Unsupported source type: {type(source)}"
            raise ValueError(msg)

        response = client.post(url, files=files, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["audio_url"]

    async def _upload_audio_async(self, client: httpx.AsyncClient, source: Union[str, Path, ByteStream]) -> str:
        headers = self._get_headers()
        url = f"{self.api_base_url}/upload"

        if isinstance(source, (str, Path)):
            path_str = str(source)
            if path_str.startswith(("http://", "https://")):
                return path_str
            file_path = Path(path_str)
            content = file_path.read_bytes()
            files = {"audio": (file_path.name, content)}
        elif isinstance(source, ByteStream):
            filename = source.meta.get("filename", "audio.wav")
            files = {"audio": (filename, source.data)}
        else:
            msg = f"Unsupported source type: {type(source)}"
            raise ValueError(msg)

        response = await client.post(url, files=files, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["audio_url"]

    def _transcribe_url(self, client: httpx.Client, audio_url: str) -> Document:
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        create_url = f"{self.api_base_url}/transcription"
        payload = {"audio_url": audio_url, **self.gladia_params}

        response = client.post(create_url, json=payload, headers=headers)
        response.raise_for_status()
        res_data = response.json()

        result_url = res_data.get("result_url") or f"{self.api_base_url}/transcription/{res_data['id']}"

        start_time = time.time()
        poll_headers = self._get_headers()

        while True:
            if time.time() - start_time > self.timeout:
                msg = f"Gladia transcription job timed out after {self.timeout} seconds."
                raise RuntimeError(msg)

            poll_resp = client.get(result_url, headers=poll_headers)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()

            status = poll_data.get("status")
            if status == "done":
                transcription_res = poll_data.get("result", {}).get("transcription", {})
                full_transcript = transcription_res.get("full_transcript", "")
                return Document(content=full_transcript, meta={"gladia_result": poll_data})
            elif status == "error":
                msg = f"Gladia transcription failed: {poll_data.get('error')}"
                raise RuntimeError(msg)

            time.sleep(self.polling_interval)

    async def _transcribe_url_async(self, client: httpx.AsyncClient, audio_url: str) -> Document:
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        create_url = f"{self.api_base_url}/transcription"
        payload = {"audio_url": audio_url, **self.gladia_params}

        response = await client.post(create_url, json=payload, headers=headers)
        response.raise_for_status()
        res_data = response.json()

        result_url = res_data.get("result_url") or f"{self.api_base_url}/transcription/{res_data['id']}"

        start_time = time.time()
        poll_headers = self._get_headers()

        while True:
            if time.time() - start_time > self.timeout:
                msg = f"Gladia transcription job timed out after {self.timeout} seconds."
                raise RuntimeError(msg)

            poll_resp = await client.get(result_url, headers=poll_headers)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()

            status = poll_data.get("status")
            if status == "done":
                transcription_res = poll_data.get("result", {}).get("transcription", {})
                full_transcript = transcription_res.get("full_transcript", "")
                return Document(content=full_transcript, meta={"gladia_result": poll_data})
            elif status == "error":
                msg = f"Gladia transcription failed: {poll_data.get('error')}"
                raise RuntimeError(msg)

            await asyncio.sleep(self.polling_interval)

    @component.output_types(documents=list[Document])
    def run(self, sources: Sequence[Union[str, Path, ByteStream]]) -> dict[str, list[Document]]:
        """
        Transcribes the given audio sources.

        :param sources: A list of file paths, audio URLs, or ByteStream objects to transcribe.
        :returns: A dictionary with key 'documents' containing transcribed Document objects.
        """
        if not sources:
            return {"documents": []}

        documents = []
        with httpx.Client(timeout=self.timeout) as client:
            for source in sources:
                audio_url = self._upload_audio(client, source)
                doc = self._transcribe_url(client, audio_url)
                documents.append(doc)

        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(self, sources: Sequence[Union[str, Path, ByteStream]]) -> dict[str, list[Document]]:
        """
        Asynchronously transcribes the given audio sources.

        :param sources: A list of file paths, audio URLs, or ByteStream objects to transcribe.
        :returns: A dictionary with key 'documents' containing transcribed Document objects.
        """
        if not sources:
            return {"documents": []}

        documents = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for source in sources:
                audio_url = await self._upload_audio_async(client, source)
                doc = await self._transcribe_url_async(client, audio_url)
                documents.append(doc)

        return {"documents": documents}
