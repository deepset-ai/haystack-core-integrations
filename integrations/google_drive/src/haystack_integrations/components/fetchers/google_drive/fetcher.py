# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any

import httpx
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.common.google_drive.errors import GoogleDriveConfigError, GoogleDriveRequestError
from haystack_integrations.common.google_drive.utils import (
    DEFAULT_API_BASE_URL,
    build_async_retrying,
    build_retrying,
    resolve_access_token,
)

logger = logging.getLogger(__name__)

_FOLDER_MIME = "application/vnd.google-apps.folder"
# Native Google Workspace types share this prefix; they cannot be downloaded with `alt=media` and must be exported.
_GOOGLE_APPS_PREFIX = "application/vnd.google-apps."

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

# Default export targets for native Google Docs/Sheets/Slides
_DEFAULT_EXPORT_MIME_TYPES = {
    "application/vnd.google-apps.document": _DOCX_MIME,
    "application/vnd.google-apps.spreadsheet": _XLSX_MIME,
    "application/vnd.google-apps.presentation": _PPTX_MIME,
}

_METADATA_FIELDS = "id,name,mimeType,webViewLink"
# A Drive sharing/edit URL embeds the file id as `/d/<id>` or `?id=<id>`.
_FILE_ID_PATTERNS = (re.compile(r"/d/([a-zA-Z0-9_-]+)"), re.compile(r"[?&]id=([a-zA-Z0-9_-]+)"))


def _resolve_file_id(value: str) -> str:
    """Extract the Drive file id from a sharing/edit URL, or return the value unchanged if it is a bare id."""
    for pattern in _FILE_ID_PATTERNS:
        match = pattern.search(value)
        if match:
            return match.group(1)
    return value


@component
class GoogleDriveFetcher:
    """
    Fetches the full content of Google Drive files via the Drive API v3.

    The fetcher complements `GoogleDriveRetriever`, which returns only metadata (and optionally exported text).
    Wire the retriever's `documents` (or a list of file ids / Drive URLs) into this fetcher to download the full
    content. It dispatches on each file's mime type and always returns `ByteStream`s, ready for a downstream
    converter (for example a `FileTypeRouter` in front of `PyPDFToDocument`, `DOCXToDocument`, `XLSXToDocument`,
    or `PPTXToDocument`):

    - **Binary files** (PDF, DOCX, images, ...) are downloaded as-is via `files.get?alt=media`.
    - **Native Google Docs/Sheets/Slides** are exported with `files.export`, by default to the Office formats
      (DOCX/XLSX/PPTX), configurable via `export_mime_types`.
    - **Folders** and other non-downloadable Google types (Forms, Sites, ...) are skipped.

    Each `ByteStream`'s `meta` carries `file_id`, `web_url`, `file_name`, and `content_type`.

    The fetcher takes a per-user `access_token` as a run input, typically wired from an upstream `OAuthTokenResolver`.
    The token must carry a delegated Google OAuth scope that allows reading file content (for example
    `https://www.googleapis.com/auth/drive.readonly`).

    ### Usage example
    ```python
    from haystack_integrations.components.fetchers.google_drive import GoogleDriveFetcher

    fetcher = GoogleDriveFetcher()

    # `access_token` is a per-user delegated Google OAuth bearer token.
    result = fetcher.run(
        access_token="my-delegated-google-token",
        targets=["https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view"],
    )
    streams = result["streams"]
    ```

    In a pipeline, connect `GoogleDriveRetriever.documents` to the fetcher's `targets` input and an upstream
    component that emits a per-user `access_token` to the fetcher's `access_token` input.
    """

    def __init__(
        self,
        *,
        api_base_url: str = DEFAULT_API_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        raise_on_failure: bool = True,
        export_mime_types: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the fetcher.

        :param api_base_url: The Drive API base URL. Defaults to `https://www.googleapis.com/drive/v3`.
        :param timeout: The HTTP timeout in seconds for each request to the Drive API.
        :param max_retries: The maximum number of retries for throttled (HTTP 429) or transient server errors.
        :param raise_on_failure: If `True`, a fetch failure raises an exception. If `False`, the failure is
            logged and the file is skipped, so the other files are still returned.
        :param export_mime_types: Optional mapping of native Google mime type (for example
            `application/vnd.google-apps.document`) to the mime type to export it as. Replaces the default mapping
            (Docs/Sheets/Slides to DOCX/XLSX/PPTX). Drive caps a single export at 10 MB.
        :raises GoogleDriveConfigError: If `max_retries` is negative.
        """
        if max_retries < 0:
            msg = "max_retries must be zero or a positive integer."
            raise GoogleDriveConfigError(msg)

        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.raise_on_failure = raise_on_failure
        self.export_mime_types = export_mime_types
        self._export_map = export_mime_types if export_mime_types is not None else _DEFAULT_EXPORT_MIME_TYPES

    @component.output_types(streams=list[ByteStream])
    def run(
        self,
        access_token: str | Secret,
        targets: list[Document] | list[str],
    ) -> dict[str, list[ByteStream]]:
        """
        Fetch the content of Google Drive files and return them as `ByteStream`s.

        :param access_token: A delegated Google OAuth bearer token for the user whose files are fetched, typically
            wired from an upstream `OAuthTokenResolver` (which emits a plain `str`). A `Secret` is also accepted and
            resolved internally.
        :param targets: The files to fetch, as either `Document`s emitted by `GoogleDriveRetriever` or raw Google
            Drive file ids / URLs (the two may also be mixed in one list). For a `Document`, the `file_id` in its
            meta is fetched and `mime_type`, `file_name`, and `web_url` are reused when present. For a raw string,
            the file id is parsed from a Drive URL (or used as-is) and the file's mime type is looked up. Folders
            and non-downloadable Google types are skipped.
        :returns: A dictionary with a `streams` key holding the fetched content as `ByteStream` objects. Each
            stream's `meta` carries `file_id`, `web_url`, `file_name`, and `content_type`.
        :raises GoogleDriveConfigError: If an item is neither a `Document` nor a `str`, or if `access_token` is a
            `Secret` that does not resolve to a string.
        :raises GoogleDriveRequestError: If a fetch fails and `raise_on_failure` is `True`.
        """
        token = resolve_access_token(access_token)
        resolved = self._collect_targets(targets)
        streams: list[ByteStream] = []
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            for file_id, hints in resolved:
                stream = self._process(client, token, file_id, hints)
                if stream is not None:
                    streams.append(stream)

        return {"streams": streams}

    @component.output_types(streams=list[ByteStream])
    async def run_async(
        self,
        access_token: str | Secret,
        targets: list[Document] | list[str],
    ) -> dict[str, list[ByteStream]]:
        """
        Asynchronously fetch the content of Google Drive files and return them as `ByteStream`s.

        :param access_token: A delegated Google OAuth bearer token for the user whose files are fetched, typically
            wired from an upstream `OAuthTokenResolver` (which emits a plain `str`). A `Secret` is also accepted and
            resolved internally.
        :param targets: The files to fetch, as either `Document`s emitted by `GoogleDriveRetriever` or raw Google
            Drive file ids / URLs (the two may also be mixed in one list). For a `Document`, the `file_id` in its
            meta is fetched and `mime_type`, `file_name`, and `web_url` are reused when present. For a raw string,
            the file id is parsed from a Drive URL (or used as-is) and the file's mime type is looked up. Folders
            and non-downloadable Google types are skipped.
        :returns: A dictionary with a `streams` key holding the fetched content as `ByteStream` objects. Each
            stream's `meta` carries `file_id`, `web_url`, `file_name`, and `content_type`.
        :raises GoogleDriveConfigError: If an item is neither a `Document` nor a `str`, or if `access_token` is a
            `Secret` that does not resolve to a string.
        :raises GoogleDriveRequestError: If a fetch fails and `raise_on_failure` is `True`.
        """
        token = resolve_access_token(access_token)
        resolved = self._collect_targets(targets)
        streams: list[ByteStream] = []
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            for file_id, hints in resolved:
                stream = await self._process_async(client, token, file_id, hints)
                if stream is not None:
                    streams.append(stream)

        return {"streams": streams}

    def _collect_targets(self, targets: list[Document] | list[str]) -> list[tuple[str, dict[str, Any]]]:
        """Resolve the input targets into `(file_id, hints)` pairs to fetch, dispatching each item by its type."""
        resolved: list[tuple[str, dict[str, Any]]] = []
        for item in targets:
            if isinstance(item, Document):
                target = self._resolve_document(item)
                if target is not None:
                    resolved.append(target)
            elif isinstance(item, str):
                resolved.append((_resolve_file_id(item), {}))
            else:
                msg = f"targets items must be a Document or a str (file id or URL), got {type(item).__name__}."
                raise GoogleDriveConfigError(msg)

        return resolved

    @staticmethod
    def _resolve_document(document: Document) -> tuple[str, dict[str, Any]] | None:
        """Turn a retriever `Document` into a `(file_id, hints)` pair, or `None` if it has no file id."""
        meta = document.meta or {}
        file_id = meta.get("file_id")
        if not file_id:
            logger.warning("Skipping document {id}: no `file_id` in its meta, nothing to fetch.", id=document.id)
            return None
        hints = {
            "mime_type": meta.get("mime_type"),
            "file_name": meta.get("file_name"),
            "web_url": meta.get("web_url"),
        }
        return file_id, hints

    # --- synchronous fetch path ---------------------------------------------------------------------------------

    def _process(self, client: httpx.Client, token: str, file_id: str, hints: dict[str, Any]) -> ByteStream | None:
        """Fetch a single file, honoring `raise_on_failure` for any error."""
        try:
            return self._dispatch(client, token, file_id, hints)
        except (GoogleDriveRequestError, httpx.HTTPError) as error:
            if self.raise_on_failure:
                raise
            logger.warning("Failed to fetch Google Drive file {file_id}: {error}", file_id=file_id, error=str(error))
            return None

    def _dispatch(self, client: httpx.Client, token: str, file_id: str, hints: dict[str, Any]) -> ByteStream | None:
        """Route a file to a binary download or an export based on its mime type."""
        mime_type = hints.get("mime_type")
        file_name = hints.get("file_name")
        web_url = hints.get("web_url")
        if mime_type is None:
            metadata = self._request_json(client, token, self._files_url(file_id), _metadata_params(), file_id)
            mime_type = metadata.get("mimeType")
            file_name = file_name or metadata.get("name")
            web_url = web_url or metadata.get("webViewLink")

        export_mime = self._export_map.get(mime_type) if mime_type else None
        if export_mime is not None:
            response = self._get(client, token, self._export_url(file_id), _export_params(export_mime))
            if response.is_error:
                raise self._build_error(response, file_id)
            return self._byte_stream(response.content, export_mime, file_id, web_url, file_name)

        if self._is_skippable(mime_type, file_id):
            return None

        response = self._get(client, token, self._files_url(file_id), _media_params())
        if response.is_error:
            raise self._build_error(response, file_id)
        return self._byte_stream(
            response.content, self._content_type(response) or mime_type, file_id, web_url, file_name
        )

    def _get(self, client: httpx.Client, token: str, url: str, params: dict[str, Any]) -> httpx.Response:
        """Issue a GET to the Drive API, retrying on throttling/transient errors."""
        headers = {"Authorization": f"Bearer {token}"}
        retrying = build_retrying(self.max_retries)
        return retrying(client.get, url, headers=headers, params=params)

    def _request_json(
        self, client: httpx.Client, token: str, url: str, params: dict[str, Any], file_id: str
    ) -> dict[str, Any]:
        response = self._get(client, token, url, params)
        if response.is_error:
            raise self._build_error(response, file_id)
        return response.json()

    # --- asynchronous fetch path --------------------------------------------------------------------------------

    async def _process_async(
        self, client: httpx.AsyncClient, token: str, file_id: str, hints: dict[str, Any]
    ) -> ByteStream | None:
        try:
            return await self._dispatch_async(client, token, file_id, hints)
        except (GoogleDriveRequestError, httpx.HTTPError) as error:
            if self.raise_on_failure:
                raise
            logger.warning("Failed to fetch Google Drive file {file_id}: {error}", file_id=file_id, error=str(error))
            return None

    async def _dispatch_async(
        self, client: httpx.AsyncClient, token: str, file_id: str, hints: dict[str, Any]
    ) -> ByteStream | None:
        mime_type = hints.get("mime_type")
        file_name = hints.get("file_name")
        web_url = hints.get("web_url")
        if mime_type is None:
            metadata = await self._request_json_async(
                client, token, self._files_url(file_id), _metadata_params(), file_id
            )
            mime_type = metadata.get("mimeType")
            file_name = file_name or metadata.get("name")
            web_url = web_url or metadata.get("webViewLink")

        export_mime = self._export_map.get(mime_type) if mime_type else None
        if export_mime is not None:
            response = await self._get_async(client, token, self._export_url(file_id), _export_params(export_mime))
            if response.is_error:
                raise self._build_error(response, file_id)
            return self._byte_stream(response.content, export_mime, file_id, web_url, file_name)

        if self._is_skippable(mime_type, file_id):
            return None

        response = await self._get_async(client, token, self._files_url(file_id), _media_params())
        if response.is_error:
            raise self._build_error(response, file_id)
        return self._byte_stream(
            response.content, self._content_type(response) or mime_type, file_id, web_url, file_name
        )

    async def _get_async(
        self, client: httpx.AsyncClient, token: str, url: str, params: dict[str, Any]
    ) -> httpx.Response:
        headers = {"Authorization": f"Bearer {token}"}
        retrying = build_async_retrying(self.max_retries)
        return await retrying(client.get, url, headers=headers, params=params)

    async def _request_json_async(
        self, client: httpx.AsyncClient, token: str, url: str, params: dict[str, Any], file_id: str
    ) -> dict[str, Any]:
        response = await self._get_async(client, token, url, params)
        if response.is_error:
            raise self._build_error(response, file_id)
        return response.json()

    # --- helpers (shared between sync and async) ----------------------------------------------------------------

    def _files_url(self, file_id: str) -> str:
        return f"{self.api_base_url}/files/{file_id}"

    def _export_url(self, file_id: str) -> str:
        return f"{self.api_base_url}/files/{file_id}/export"

    @staticmethod
    def _is_skippable(mime_type: str | None, file_id: str) -> bool:
        """Return whether a file has no downloadable content (a folder or a non-exportable Google type)."""
        if mime_type == _FOLDER_MIME:
            logger.warning("Skipping folder {file_id}: it has no downloadable content.", file_id=file_id)
            return True
        if mime_type and mime_type.startswith(_GOOGLE_APPS_PREFIX):
            logger.warning(
                "Skipping {file_id}: Google type {mime_type} cannot be downloaded or exported.",
                file_id=file_id,
                mime_type=mime_type,
            )
            return True
        return False

    @staticmethod
    def _byte_stream(
        data: bytes, content_type: str | None, file_id: str, web_url: str | None, file_name: str | None
    ) -> ByteStream:
        meta = {"file_id": file_id, "web_url": web_url, "file_name": file_name, "content_type": content_type}
        return ByteStream(
            data=data, mime_type=content_type, meta={key: value for key, value in meta.items() if value is not None}
        )

    @staticmethod
    def _build_error(response: httpx.Response, file_id: str) -> GoogleDriveRequestError:
        """Build a descriptive `GoogleDriveRequestError` for an error response."""
        status = response.status_code
        if status == httpx.codes.UNAUTHORIZED:
            msg = (
                "Google Drive rejected the access token (401 Unauthorized). The token may be expired, invalid, or "
                "missing a required scope (for example https://www.googleapis.com/auth/drive.readonly)."
            )
        elif status == httpx.codes.FORBIDDEN:
            msg = f"Google Drive denied access to file {file_id} (403 Forbidden): {response.text}"
        elif status == httpx.codes.NOT_FOUND:
            msg = f"Google Drive could not find file {file_id} (404 Not Found): {response.text}"
        else:
            msg = f"Google Drive fetch of file {file_id} failed with status {status}: {response.text}"

        return GoogleDriveRequestError(msg, status_code=status)

    @staticmethod
    def _content_type(response: httpx.Response) -> str | None:
        """Extract the bare content type (without parameters) from a response."""
        content_type = response.headers.get("Content-Type", "")
        return content_type.split(";")[0].strip() or None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns: The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            api_base_url=self.api_base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            raise_on_failure=self.raise_on_failure,
            export_mime_types=self.export_mime_types,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoogleDriveFetcher":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns: The deserialized component instance.
        """
        return default_from_dict(cls, data)


def _metadata_params() -> dict[str, Any]:
    return {"fields": _METADATA_FIELDS, "supportsAllDrives": True}


def _media_params() -> dict[str, Any]:
    return {"alt": "media", "supportsAllDrives": True}


def _export_params(export_mime: str) -> dict[str, Any]:
    return {"mimeType": export_mime, "supportsAllDrives": True}
