# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import mimetypes
import time
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)

_FILE_CONVERT_PATH = "/v1/convert/file"
_SOURCE_CONVERT_PATH = "/v1/convert/source"
_STATUS_POLL_PATH = "/v1/status/poll"
_RESULT_PATH = "/v1/result"
_TERMINAL_TASK_STATUSES = {"success", "failure"}
_FAILED_CONVERSION_STATUSES = {"failure", "skipped"}


class ExportType(str, Enum):
    """
    Enumeration of export formats supported by DoclingServe.

    - `MARKDOWN`: Converts documents to Markdown format.
    - `TEXT`: Extracts plain text.
    - `JSON`: Returns the full Docling document as a JSON string.
    """

    MARKDOWN = "markdown"
    TEXT = "text"
    JSON = "json"


class ConversionMode(str, Enum):
    """
    Execution mode for DoclingServe conversions.

    - `SYNC`: Uses DoclingServe's synchronous conversion endpoints.
    - `ASYNC`: Uses DoclingServe's async job endpoints and polls for completion.
    """

    SYNC = "sync"
    ASYNC = "async"


class DoclingServeConversionError(Exception):
    """Raised when DoclingServe reports an async task or conversion failure."""


class DoclingServeTimeoutError(DoclingServeConversionError):
    """Raised when a DoclingServe async task exceeds job_timeout."""


def _is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https")


def _resolve_filename(source: str | Path | ByteStream) -> str:
    """Extract a filename for a source, used as a hint to docling-serve for format detection."""
    if isinstance(source, ByteStream):
        meta = source.meta or {}
        raw = meta.get("file_path") or meta.get("file_name") or meta.get("name")
        return Path(raw).name if raw else "document"
    return Path(source).name


def _guess_mime_type(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


@component
class DoclingServeConverter:
    """
    Converts documents to Haystack Documents using a DoclingServe server.

    See [DoclingServe](https://github.com/docling-project/docling-serve).

    DoclingServe hosts Docling in a scalable HTTP server, supporting PDFs, Office documents, HTML, and many other
    formats. Unlike the local `DoclingConverter`, this component has no heavy ML dependencies — all processing
    happens on the remote server.

    Local files and ByteStreams are uploaded via the ``/v1/convert/file`` endpoint. URL strings are sent to
    ``/v1/convert/source``.

    Supports both synchronous (`run`) and asynchronous (`run_async`) execution.

    ### Usage example

    ```python
    from haystack_integrations.components.converters.docling_serve import DoclingServeConverter

    converter = DoclingServeConverter(base_url="http://localhost:5001")
    result = converter.run(sources=["https://arxiv.org/pdf/2206.01062"])
    print(result["documents"][0].content[:200])
    ```
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:5001",
        export_type: ExportType = ExportType.MARKDOWN,
        convert_options: dict[str, Any] | None = None,
        timeout: float = 120.0,
        api_key: Secret | None = Secret.from_env_var("DOCLING_SERVE_API_KEY", strict=False),
        mode: ConversionMode | str = ConversionMode.SYNC,
        poll_interval: float = 2.0,
        job_timeout: float = 600.0,
    ) -> None:
        """
        Initializes the DoclingServeConverter.

        :param base_url:
            Base URL of the DoclingServe instance. Defaults to `"http://localhost:5001"`.
        :param export_type:
            The output format for converted documents. One of `ExportType.MARKDOWN` (default),
            `ExportType.TEXT`, or `ExportType.JSON`.
        :param convert_options:
            Optional dictionary of conversion options passed directly to the DoclingServe API
            (e.g. `{"do_ocr": True, "ocr_engine": "tesseract"}`).
            See [DoclingServe options](https://github.com/docling-project/docling-serve/blob/main/docs/usage.md).
            Note: `to_formats` is set automatically based on `export_type` and should not be included here.
        :param timeout:
            HTTP request timeout in seconds. Defaults to `120.0`.
        :param api_key:
            API key for authenticating with a secured DoclingServe instance. Reads from the
            `DOCLING_SERVE_API_KEY` environment variable by default. Set to `None` to disable
            authentication.
        :param mode:
            Conversion mode. `sync` uses DoclingServe's synchronous endpoints. `async` submits
            conversion jobs to DoclingServe's async endpoints and polls until completion.
        :param poll_interval:
            Controls both the server-side long-poll wait (?wait= parameter) and the maximum local sleep between polls.
            A higher value reduces round-trips; a lower value increases polling frequency.
        :param job_timeout:
            Maximum time in seconds to wait for each async conversion job.
        """
        self.mode = ConversionMode(mode)
        if self.mode == ConversionMode.ASYNC:
            if poll_interval <= 0:
                msg = "poll_interval must be greater than 0."
                raise ValueError(msg)
            if job_timeout <= 0:
                msg = "job_timeout must be greater than 0."
                raise ValueError(msg)

        self.base_url = base_url.rstrip("/")
        self.export_type = ExportType(export_type)
        self.convert_options = dict(convert_options) if convert_options else {}
        self.timeout = timeout
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.job_timeout = job_timeout

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            A dictionary representation of the component.
        """
        return default_to_dict(
            self,
            base_url=self.base_url,
            export_type=self.export_type.value,
            convert_options=self.convert_options,
            timeout=self.timeout,
            api_key=self.api_key.to_dict() if self.api_key else None,
            mode=self.mode.value,
            poll_interval=self.poll_interval,
            job_timeout=self.job_timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DoclingServeConverter":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary representation of the component.
        :returns:
            A new `DoclingServeConverter` instance.
        """
        deserialize_secrets_inplace(data.get("init_parameters", {}), keys=["api_key"])
        return default_from_dict(cls, data)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            resolved = self.api_key.resolve_value()
            if resolved:
                headers["X-Api-Key"] = resolved
        return headers

    def _to_format(self) -> str:
        return {"markdown": "md", "text": "text", "json": "json"}[self.export_type.value]

    def _extract_content(self, data: dict[str, Any]) -> str | None:
        doc = data.get("document", {})
        if self.export_type == ExportType.MARKDOWN:
            return doc.get("md_content")
        if self.export_type == ExportType.TEXT:
            return doc.get("text_content")
        if self.export_type == ExportType.JSON:
            content = doc.get("json_content")
            return json.dumps(content) if content is not None else None
        return None

    def _raise_for_failed_conversion(self, data: dict[str, Any]) -> None:
        status = data.get("status")
        if status not in _FAILED_CONVERSION_STATUSES:
            return

        errors = data.get("errors") or []
        details = "; ".join(
            str(error.get("error_message", error)) if isinstance(error, dict) else str(error) for error in errors
        )
        msg = f"DoclingServe conversion finished with status '{status}'"
        if details:
            msg = f"{msg}: {details}"
        raise DoclingServeConversionError(msg)

    def _extract_task_id(self, data: dict[str, Any]) -> str:
        task_id = data.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            msg = "DoclingServe async task response did not include a task_id."
            raise DoclingServeConversionError(msg)
        return task_id

    def _raise_for_failed_task(self, data: dict[str, Any]) -> None:
        if data.get("task_status") != "failure":
            return

        error_message = data.get("error_message") or "DoclingServe async task failed."
        raise DoclingServeConversionError(str(error_message))

    def _async_source_payload(self, url: str) -> dict[str, Any]:
        return {
            "options": {**self.convert_options, "to_formats": [self._to_format()]},
            "sources": [{"kind": "http", "url": url}],
            "target": {"kind": "inbody"},
        }

    def _post_file(self, client: httpx.Client, source: str | Path | ByteStream) -> dict[str, Any]:
        filename = _resolve_filename(source)
        file_bytes = source.data if isinstance(source, ByteStream) else Path(source).read_bytes()
        mime_type = (
            (source.mime_type or _guess_mime_type(filename))
            if isinstance(source, ByteStream)
            else _guess_mime_type(filename)
        )
        options = {**self.convert_options, "to_formats": self._to_format()}
        response = client.post(
            f"{self.base_url}{_FILE_CONVERT_PATH}",
            files={"files": (filename, file_bytes, mime_type)},
            data=options,
            headers=self._headers(),
        )
        response.raise_for_status()
        data = response.json()
        self._raise_for_failed_conversion(data)
        return data

    def _submit_file_job(self, client: httpx.Client, source: str | Path | ByteStream) -> str:
        filename = _resolve_filename(source)
        file_bytes = source.data if isinstance(source, ByteStream) else Path(source).read_bytes()
        mime_type = (
            (source.mime_type or _guess_mime_type(filename))
            if isinstance(source, ByteStream)
            else _guess_mime_type(filename)
        )
        options = {**self.convert_options, "to_formats": self._to_format(), "target_type": "inbody"}
        response = client.post(
            f"{self.base_url}{_FILE_CONVERT_PATH}/async",
            files={"files": (filename, file_bytes, mime_type)},
            data=options,
            headers=self._headers(),
        )
        response.raise_for_status()
        return self._extract_task_id(response.json())

    def _post_url(self, client: httpx.Client, url: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "options": {**self.convert_options, "to_formats": [self._to_format()]},
            "sources": [{"kind": "http", "url": url}],
        }
        response = client.post(
            f"{self.base_url}{_SOURCE_CONVERT_PATH}",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        data = response.json()
        self._raise_for_failed_conversion(data)
        return data

    def _submit_url_job(self, client: httpx.Client, url: str) -> str:
        response = client.post(
            f"{self.base_url}{_SOURCE_CONVERT_PATH}/async",
            json=self._async_source_payload(url),
            headers=self._headers(),
        )
        response.raise_for_status()
        return self._extract_task_id(response.json())

    def _poll_job_status(self, client: httpx.Client, task_id: str, wait: float) -> dict[str, Any]:
        response = client.get(
            f"{self.base_url}{_STATUS_POLL_PATH}/{task_id}",
            params={"wait": wait},
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    def _wait_for_job(self, client: httpx.Client, task_id: str) -> None:
        deadline = time.monotonic() + self.job_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                msg = f"Timed out waiting for DoclingServe task {task_id} after {self.job_timeout:.2f}s."
                raise DoclingServeTimeoutError(msg)

            wait = min(self.poll_interval, remaining)
            poll_started = time.monotonic()
            status = self._poll_job_status(client, task_id, wait)
            task_status = status.get("task_status")
            if task_status in _TERMINAL_TASK_STATUSES:
                self._raise_for_failed_task(status)
                return

            sleep_for = min(self.poll_interval, remaining) - (time.monotonic() - poll_started)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _fetch_job_result(self, client: httpx.Client, task_id: str) -> dict[str, Any]:
        response = client.get(
            f"{self.base_url}{_RESULT_PATH}/{task_id}",
            headers=self._headers(),
        )
        response.raise_for_status()
        data = response.json()
        self._raise_for_failed_conversion(data)
        return data

    def _post_file_job(self, client: httpx.Client, source: str | Path | ByteStream) -> dict[str, Any]:
        task_id = self._submit_file_job(client, source)
        self._wait_for_job(client, task_id)
        return self._fetch_job_result(client, task_id)

    def _post_url_job(self, client: httpx.Client, url: str) -> dict[str, Any]:
        task_id = self._submit_url_job(client, url)
        self._wait_for_job(client, task_id)
        return self._fetch_job_result(client, task_id)

    async def _post_file_async(self, client: httpx.AsyncClient, source: str | Path | ByteStream) -> dict[str, Any]:
        filename = _resolve_filename(source)
        file_bytes = source.data if isinstance(source, ByteStream) else Path(source).read_bytes()
        mime_type = (
            (source.mime_type or _guess_mime_type(filename))
            if isinstance(source, ByteStream)
            else _guess_mime_type(filename)
        )
        options = {**self.convert_options, "to_formats": self._to_format()}
        response = await client.post(
            f"{self.base_url}{_FILE_CONVERT_PATH}",
            files={"files": (filename, file_bytes, mime_type)},
            data=options,
            headers=self._headers(),
        )
        response.raise_for_status()
        data = response.json()
        self._raise_for_failed_conversion(data)
        return data

    async def _submit_file_job_async(self, client: httpx.AsyncClient, source: str | Path | ByteStream) -> str:
        filename = _resolve_filename(source)
        file_bytes = source.data if isinstance(source, ByteStream) else Path(source).read_bytes()
        mime_type = (
            (source.mime_type or _guess_mime_type(filename))
            if isinstance(source, ByteStream)
            else _guess_mime_type(filename)
        )
        options = {**self.convert_options, "to_formats": self._to_format(), "target_type": "inbody"}
        response = await client.post(
            f"{self.base_url}{_FILE_CONVERT_PATH}/async",
            files={"files": (filename, file_bytes, mime_type)},
            data=options,
            headers=self._headers(),
        )
        response.raise_for_status()
        return self._extract_task_id(response.json())

    async def _post_url_async(self, client: httpx.AsyncClient, url: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "options": {**self.convert_options, "to_formats": [self._to_format()]},
            "sources": [{"kind": "http", "url": url}],
        }
        response = await client.post(
            f"{self.base_url}{_SOURCE_CONVERT_PATH}",
            json=payload,
            headers=self._headers(),
        )
        response.raise_for_status()
        data = response.json()
        self._raise_for_failed_conversion(data)
        return data

    async def _submit_url_job_async(self, client: httpx.AsyncClient, url: str) -> str:
        response = await client.post(
            f"{self.base_url}{_SOURCE_CONVERT_PATH}/async",
            json=self._async_source_payload(url),
            headers=self._headers(),
        )
        response.raise_for_status()
        return self._extract_task_id(response.json())

    async def _poll_job_status_async(self, client: httpx.AsyncClient, task_id: str, wait: float) -> dict[str, Any]:
        response = await client.get(
            f"{self.base_url}{_STATUS_POLL_PATH}/{task_id}",
            params={"wait": wait},
            headers=self._headers(),
        )
        response.raise_for_status()
        return response.json()

    async def _wait_for_job_async(self, client: httpx.AsyncClient, task_id: str) -> None:
        deadline = time.monotonic() + self.job_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                msg = f"Timed out waiting for DoclingServe task {task_id} after {self.job_timeout:.2f}s."
                raise DoclingServeTimeoutError(msg)

            wait = min(self.poll_interval, remaining)
            poll_started = time.monotonic()
            status = await self._poll_job_status_async(client, task_id, wait)
            task_status = status.get("task_status")
            if task_status in _TERMINAL_TASK_STATUSES:
                self._raise_for_failed_task(status)
                return

            sleep_for = min(self.poll_interval, remaining) - (time.monotonic() - poll_started)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    async def _fetch_job_result_async(self, client: httpx.AsyncClient, task_id: str) -> dict[str, Any]:
        response = await client.get(
            f"{self.base_url}{_RESULT_PATH}/{task_id}",
            headers=self._headers(),
        )
        response.raise_for_status()
        data = response.json()
        self._raise_for_failed_conversion(data)
        return data

    async def _post_file_job_async(self, client: httpx.AsyncClient, source: str | Path | ByteStream) -> dict[str, Any]:
        task_id = await self._submit_file_job_async(client, source)
        await self._wait_for_job_async(client, task_id)
        return await self._fetch_job_result_async(client, task_id)

    async def _post_url_job_async(self, client: httpx.AsyncClient, url: str) -> dict[str, Any]:
        task_id = await self._submit_url_job_async(client, url)
        await self._wait_for_job_async(client, task_id)
        return await self._fetch_job_result_async(client, task_id)

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Converts documents by sending them to DoclingServe and returns Haystack Documents.

        :param sources:
            List of sources to convert. Each item can be a URL string, a local file path, or a
            `ByteStream`. URL strings are sent to `/v1/convert/source`; all other sources are
            uploaded to `/v1/convert/file`.
        :param meta:
            Optional metadata to attach to the output Documents. Can be a single dict applied to
            all documents, or a list of dicts with one entry per source.
        :returns:
            A dictionary with key `"documents"` containing the converted Haystack Documents.
        """
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))
        documents: list[Document] = []

        with httpx.Client(timeout=self.timeout) as client:
            for source, source_meta in zip(sources, meta_list, strict=True):
                bytestream_meta = source.meta or {} if isinstance(source, ByteStream) else {}
                merged_meta = {**bytestream_meta, **source_meta}
                try:
                    if isinstance(source, str) and _is_url(source):
                        result = (
                            self._post_url_job(client, source)
                            if self.mode == ConversionMode.ASYNC
                            else self._post_url(client, source)
                        )
                    else:
                        result = (
                            self._post_file_job(client, source)
                            if self.mode == ConversionMode.ASYNC
                            else self._post_file(client, source)
                        )
                    content = self._extract_content(result)
                    if content is not None:
                        documents.append(Document(content=content, meta=merged_meta))
                    else:
                        logger.warning("No content returned for source {source}.", source=source)
                except httpx.HTTPStatusError as e:
                    logger.warning(
                        "DoclingServe returned HTTP {status} for {source}: {body}",
                        status=e.response.status_code,
                        source=source,
                        body=e.response.text,
                    )
                except httpx.HTTPError as e:
                    logger.warning(
                        "Could not connect to DoclingServe for {source}. Skipping it. Error: {error}",
                        source=source,
                        error=e,
                    )
                except DoclingServeTimeoutError as e:
                    logger.warning(
                        "Timed out waiting for DoclingServe conversion for {source}. Skipping it. Error: {error}",
                        source=source,
                        error=e,
                    )
                except DoclingServeConversionError as e:
                    logger.warning(
                        "DoclingServe conversion failed for {source}. Skipping it. Error: {error}",
                        source=source,
                        error=e,
                    )

        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously converts documents by sending them to DoclingServe.

        This is the async equivalent of `run()`, useful when DoclingServe requests should not
        block the event loop.

        :param sources:
            List of sources to convert. Each item can be a URL string, a local file path, or a
            `ByteStream`. URL strings are sent to `/v1/convert/source`; all other sources are
            uploaded to `/v1/convert/file`.
        :param meta:
            Optional metadata to attach to the output Documents.
        :returns:
            A dictionary with key `"documents"` containing the converted Haystack Documents.
        """
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))
        documents: list[Document] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for source, source_meta in zip(sources, meta_list, strict=True):
                bytestream_meta = source.meta or {} if isinstance(source, ByteStream) else {}
                merged_meta = {**bytestream_meta, **source_meta}
                try:
                    if isinstance(source, str) and _is_url(source):
                        result = (
                            await self._post_url_job_async(client, source)
                            if self.mode == ConversionMode.ASYNC
                            else await self._post_url_async(client, source)
                        )
                    else:
                        result = (
                            await self._post_file_job_async(client, source)
                            if self.mode == ConversionMode.ASYNC
                            else await self._post_file_async(client, source)
                        )
                    content = self._extract_content(result)
                    if content is not None:
                        documents.append(Document(content=content, meta=merged_meta))
                    else:
                        logger.warning("No content returned for source {source}.", source=source)
                except httpx.HTTPStatusError as e:
                    logger.warning(
                        "DoclingServe returned HTTP {status} for {source}: {body}",
                        status=e.response.status_code,
                        source=source,
                        body=e.response.text,
                    )
                except httpx.HTTPError as e:
                    logger.warning(
                        "Could not connect to DoclingServe for {source}. Skipping it. Error: {error}",
                        source=source,
                        error=e,
                    )
                except DoclingServeTimeoutError as e:
                    logger.warning(
                        "Timed out waiting for DoclingServe conversion for {source}. Skipping it. Error: {error}",
                        source=source,
                        error=e,
                    )
                except DoclingServeConversionError as e:
                    logger.warning(
                        "DoclingServe conversion failed for {source}. Skipping it. Error: {error}",
                        source=source,
                        error=e,
                    )

        return {"documents": documents}
