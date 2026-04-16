# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

logger = logging.getLogger(__name__)


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


@component
class DoclingServeConverter:
    """
    Converts documents to Haystack Documents using a DoclingServe server.

    See [DoclingServe](https://github.com/docling-project/docling-serve).

    DoclingServe hosts Docling in a scalable HTTP server, supporting PDFs, Office documents, HTML, and many other
    formats. Unlike the local `DoclingConverter`, this component has no heavy ML dependencies — all processing
    happens on the remote server.

    Supports both synchronous (`run`) and asynchronous (`arun`) execution.

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
        api_key: Secret | None = None,
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
        :param timeout:
            HTTP request timeout in seconds. Defaults to `120.0`.
        :param api_key:
            Optional API key for authenticating with a secured DoclingServe instance.
        """
        self.base_url = base_url.rstrip("/")
        self.export_type = ExportType(export_type)
        self.convert_options = convert_options or {}
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
        if api_key := data.get("init_parameters", {}).get("api_key"):
            data["init_parameters"]["api_key"] = Secret.from_dict(api_key)
        return default_from_dict(cls, data)

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key.resolve_value()}"
        return headers

    def _to_format(self) -> str:
        return {"markdown": "md", "text": "text", "json": "json"}[self.export_type.value]

    def _build_payload(self, source_entry: dict[str, Any]) -> dict[str, Any]:
        options = {**self.convert_options, "to_formats": [self._to_format()]}
        return {"options": options, "sources": [source_entry]}

    def _source_entry(self, source: str | Path | ByteStream) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Convert a source to a DoclingServe API source entry.

        :returns:
            A tuple of (source_entry dict, extra_meta dict from ByteStream if applicable).
        """
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            return {"kind": "http", "url": source}, {}
        if isinstance(source, ByteStream):
            b64 = base64.b64encode(source.data).decode()
            filename = (source.meta or {}).get("file_name", "document")
            return {"kind": "file", "base64_string": b64, "filename": filename}, source.meta or {}
        path = Path(source)
        b64 = base64.b64encode(path.read_bytes()).decode()
        return {"kind": "file", "base64_string": b64, "filename": path.name}, {}

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
            `ByteStream`.
        :param meta:
            Optional metadata to attach to the output Documents. Can be a single dict applied to
            all documents, or a list of dicts with one entry per source.
        :returns:
            A dictionary with key `"documents"` containing the converted Haystack Documents.
        """
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))
        documents: list[Document] = []
        headers = self._headers()

        with httpx.Client(timeout=self.timeout) as client:
            for source, source_meta in zip(sources, meta_list, strict=True):
                try:
                    source_entry, bytestream_meta = self._source_entry(source)
                    payload = self._build_payload(source_entry)
                    response = client.post(
                        f"{self.base_url}/v1/convert/source",
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    content = self._extract_content(response.json())
                    if content is not None:
                        documents.append(Document(content=content, meta={**bytestream_meta, **source_meta}))
                    else:
                        logger.warning("No content returned for source {source}.", source=source)
                except Exception as e:
                    logger.warning(
                        "Could not convert source {source}. Skipping it. Error: {error}",
                        source=source,
                        error=e,
                    )

        return {"documents": documents}

    async def arun(
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
            `ByteStream`.
        :param meta:
            Optional metadata to attach to the output Documents.
        :returns:
            A dictionary with key `"documents"` containing the converted Haystack Documents.
        """
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))
        documents: list[Document] = []
        headers = self._headers()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for source, source_meta in zip(sources, meta_list, strict=True):
                try:
                    source_entry, bytestream_meta = self._source_entry(source)
                    payload = self._build_payload(source_entry)
                    response = await client.post(
                        f"{self.base_url}/v1/convert/source",
                        json=payload,
                        headers=headers,
                    )
                    response.raise_for_status()
                    content = self._extract_content(response.json())
                    if content is not None:
                        documents.append(Document(content=content, meta={**bytestream_meta, **source_meta}))
                    else:
                        logger.warning("No content returned for source {source}.", source=source)
                except Exception as e:
                    logger.warning(
                        "Could not convert source {source}. Skipping it. Error: {error}",
                        source=source,
                        error=e,
                    )

        return {"documents": documents}
