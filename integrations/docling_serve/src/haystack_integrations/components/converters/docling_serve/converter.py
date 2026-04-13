# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Haystack converter component for docling-serve."""

import mimetypes
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


def _is_url(source: str) -> bool:
    """Check if a string looks like a URL."""
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https")


def _resolve_source_name(source: str | Path | ByteStream) -> str:
    """Extract a human-readable name for a source."""
    if isinstance(source, ByteStream):
        meta = source.meta or {}
        return meta.get("file_path") or meta.get("file_name") or meta.get("name") or "document"
    return str(source)


def _guess_mime_type(filename: str) -> str:
    """Guess the MIME type of a file based on its name."""
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def _build_file_upload(
    source: str | Path | ByteStream,
) -> tuple[str, bytes, str]:
    """
    Prepare file upload data from a source.

    :returns: Tuple of (filename, file_bytes, mime_type).
    """
    if isinstance(source, ByteStream):
        meta = source.meta or {}
        filename = meta.get("file_name") or meta.get("file_path") or meta.get("name") or "document"
        filename = Path(filename).name
        mime_type = source.mime_type or _guess_mime_type(filename)
        return filename, source.data, mime_type

    file_path = Path(source)
    return file_path.name, file_path.read_bytes(), _guess_mime_type(file_path.name)


def _extract_document(response_json: dict[str, Any], source_name: str, extra_meta: dict[str, Any]) -> Document:
    """
    Extract a Haystack Document from a docling-serve response.

    :param response_json: The parsed JSON response from docling-serve.
    :param source_name: Human-readable name of the source.
    :param extra_meta: Additional metadata to merge in.
    :returns: A Haystack Document.
    """
    document_data = response_json.get("document", {})
    content = document_data.get("md_content") or document_data.get("text_content") or ""

    meta = {
        "source_file": source_name,
        "conversion_status": response_json.get("status", ""),
        "processing_time": response_json.get("processing_time", 0.0),
        **extra_meta,
    }

    return Document(content=content, meta=meta)


@component
class DoclingServeConverter:
    """
    Convert documents using a running docling-serve instance.

    Sends files or URLs to a docling-serve API endpoint and converts the responses
    into Haystack Document objects. Local files and ByteStreams are uploaded via the
    ``/v1/convert/file`` endpoint, while URL strings are sent to ``/v1/convert/source``.

    ### Usage example

    ```python
    from haystack_integrations.components.converters.docling_serve import DoclingServeConverter

    converter = DoclingServeConverter(base_url="http://localhost:5001")
    result = converter.run(sources=["path/to/document.pdf"])
    documents = result["documents"]
    ```
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:5001",
        api_key: Secret | None = Secret.from_env_var("DOCLING_SERVE_API_KEY", strict=False),
        timeout: int = 300,
        convert_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the DoclingServeConverter.

        :param base_url: Root URL of the docling-serve instance (e.g. ``http://localhost:5001``).
        :param api_key: API key for authentication. Reads from the ``DOCLING_SERVE_API_KEY``
            environment variable by default. Set to ``None`` to disable authentication.
        :param timeout: Request timeout in seconds. Document conversion can be slow,
            so the default is 300 seconds.
        :param convert_options: Dictionary of conversion parameters passed to docling-serve.
            Supports all parameters from docling-serve's ``ConvertDocumentsRequestOptions``,
            such as ``from_formats``, ``to_formats``, ``do_ocr``, ``ocr_engine``,
            ``table_mode``, etc. See the
            `docling-serve documentation <https://github.com/docling-project/docling-serve/blob/main/docs/usage.md>`_
            for the full list.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.convert_options = convert_options or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            base_url=self.base_url,
            api_key=self.api_key.to_dict() if self.api_key else None,
            timeout=self.timeout,
            convert_options=self.convert_options,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DoclingServeConverter":
        """Deserialize the component from a dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers including authentication if configured."""
        headers: dict[str, str] = {"accept": "application/json"}
        if self.api_key:
            resolved = self.api_key.resolve_value()
            if resolved:
                headers["X-Api-Key"] = resolved
        return headers

    def _convert_file_sync(
        self,
        source: str | Path | ByteStream,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Convert a local file or ByteStream via the /v1/convert/file endpoint (sync)."""
        filename, file_bytes, mime_type = _build_file_upload(source)
        url = f"{self.base_url}{_FILE_CONVERT_PATH}"

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                url,
                files={"files": (filename, file_bytes, mime_type)},
                data=self.convert_options,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()

    def _convert_url_sync(
        self,
        source_url: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Convert a URL source via the /v1/convert/source endpoint (sync)."""
        url = f"{self.base_url}{_SOURCE_CONVERT_PATH}"
        payload: dict[str, Any] = {
            "options": self.convert_options,
            "http_sources": [{"url": source_url}],
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    async def _convert_file_async(
        self,
        source: str | Path | ByteStream,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Convert a local file or ByteStream via the /v1/convert/file endpoint (async)."""
        filename, file_bytes, mime_type = _build_file_upload(source)
        url = f"{self.base_url}{_FILE_CONVERT_PATH}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                files={"files": (filename, file_bytes, mime_type)},
                data=self.convert_options,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()

    async def _convert_url_async(
        self,
        source_url: str,
        headers: dict[str, str],
    ) -> dict[str, Any]:
        """Convert a URL source via the /v1/convert/source endpoint (async)."""
        url = f"{self.base_url}{_SOURCE_CONVERT_PATH}"
        payload: dict[str, Any] = {
            "options": self.convert_options,
            "http_sources": [{"url": source_url}],
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Convert sources to Documents using docling-serve.

        :param sources: List of file paths, URLs, or ByteStream objects to convert.
            Strings starting with ``http://`` or ``https://`` are treated as URLs and sent
            to the ``/v1/convert/source`` endpoint. All other sources are uploaded to
            ``/v1/convert/file``.
        :param meta: Optional metadata to attach to the Documents.
            Can be a single dictionary (applied to all Documents) or a list of dictionaries
            (one per source). If a source is a ByteStream, its metadata is also merged.
        :returns:
            A dictionary with key ``"documents"`` containing the output Documents.
        """
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))
        headers = self._build_headers()
        documents: list[Document] = []

        for source, source_meta in zip(sources, meta_list, strict=True):
            source_name = _resolve_source_name(source)
            merged_meta = {**(source.meta if isinstance(source, ByteStream) else {}), **source_meta}

            try:
                if isinstance(source, str) and _is_url(source):
                    result = self._convert_url_sync(source, headers)
                else:
                    result = self._convert_file_sync(source, headers)

                documents.append(_extract_document(result, source_name, merged_meta))

            except httpx.HTTPStatusError as e:
                body = e.response.text
                logger.warning(
                    "docling-serve returned HTTP {status} for {source}: {body}",
                    status=e.response.status_code,
                    source=source_name,
                    body=body,
                )
            except httpx.HTTPError as e:
                logger.warning(
                    "Failed to call docling-serve for {source}: {error}",
                    source=source_name,
                    error=str(e),
                )

        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously convert sources to Documents using docling-serve.

        :param sources: List of file paths, URLs, or ByteStream objects to convert.
            Strings starting with ``http://`` or ``https://`` are treated as URLs and sent
            to the ``/v1/convert/source`` endpoint. All other sources are uploaded to
            ``/v1/convert/file``.
        :param meta: Optional metadata to attach to the Documents.
            Can be a single dictionary (applied to all Documents) or a list of dictionaries
            (one per source). If a source is a ByteStream, its metadata is also merged.
        :returns:
            A dictionary with key ``"documents"`` containing the output Documents.
        """
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))
        headers = self._build_headers()
        documents: list[Document] = []

        for source, source_meta in zip(sources, meta_list, strict=True):
            source_name = _resolve_source_name(source)
            merged_meta = {**(source.meta if isinstance(source, ByteStream) else {}), **source_meta}

            try:
                if isinstance(source, str) and _is_url(source):
                    result = await self._convert_url_async(source, headers)
                else:
                    result = await self._convert_file_async(source, headers)

                documents.append(_extract_document(result, source_name, merged_meta))

            except httpx.HTTPStatusError as e:
                body = e.response.text
                logger.warning(
                    "docling-serve returned HTTP {status} for {source}: {body}",
                    status=e.response.status_code,
                    source=source_name,
                    body=body,
                )
            except httpx.HTTPError as e:
                logger.warning(
                    "Failed to call docling-serve for {source}: {error}",
                    source=source_name,
                    error=str(e),
                )

        return {"documents": documents}
