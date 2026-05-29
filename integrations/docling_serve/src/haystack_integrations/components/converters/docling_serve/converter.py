# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import mimetypes
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
        """
        self.base_url = base_url.rstrip("/")
        self.export_type = ExportType(export_type)
        self.convert_options = dict(convert_options) if convert_options else {}
        self.timeout = timeout
        self.api_key = api_key

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
        return response.json()

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
        return response.json()

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
        return response.json()

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
        return response.json()

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
                        result = self._post_url(client, source)
                    else:
                        result = self._post_file(client, source)
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
                        result = await self._post_url_async(client, source)
                    else:
                        result = await self._post_file_async(client, source)
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

        return {"documents": documents}
