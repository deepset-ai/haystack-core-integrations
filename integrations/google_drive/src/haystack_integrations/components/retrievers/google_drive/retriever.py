# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

import httpx
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret

from haystack_integrations.common.google_drive.errors import GoogleDriveConfigError, GoogleDriveRequestError
from haystack_integrations.common.google_drive.utils import (
    DEFAULT_API_BASE_URL,
    build_async_retrying,
    build_retrying,
    resolve_access_token,
)

logger = logging.getLogger(__name__)

# The Drive `files.list` endpoint caps a single page at 1000 results; larger `top_k` values are paginated.
_MAX_PAGE_SIZE = 1000

# The file properties requested via the Drive `fields` selection.
_DEFAULT_FILE_FIELDS = (
    "id",
    "name",
    "mimeType",
    "webViewLink",
    "description",
    "fileExtension",
    "createdTime",
    "modifiedTime",
    "owners(displayName)",
    "lastModifyingUser(displayName)",
)

# The mime types that support content exporting: application/vnd.google-apps.*
_EXPORT_MIME_TYPES = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}


@component
class GoogleDriveRetriever:
    """
    Retrieves files from Google Drive via the Drive API v3 search (`files.list`) endpoint.

    Given a query, the retriever runs a full-text search over the user's Drive (and optionally shared
    drives) and maps each matching file to a Haystack `Document`. By default, each `Document` carries
    resource metadata (`file_name`, `file_id`, `web_url`, `mime_type`, `file_extension`, author, and
    timestamps) and uses the file `description` or `name` as `content`, because the Drive search API does
    not return a text snippet. Set `include_content=True` to additionally export native Google
    Docs/Sheets/Slides to text and use that as the `Document` content. Binary files (PDF, DOCX, ...) are
    never downloaded. Compose a downstream fetcher/converter on the returned `web_url`/`file_id` when full
    file content is needed.

    The retriever takes a per-user `access_token` as a run input, typically wired from an upstream
    `OAuthTokenResolver`. The token must carry a delegated Google OAuth scope that allows search
    (for example `https://www.googleapis.com/auth/drive.readonly`). The metadata-only
    `drive.metadata.readonly` scope cannot search file content or export documents.

    ### Usage example
    ```python
    from haystack import Pipeline
    from haystack.utils import Secret
    from haystack_integrations.components.connectors.oauth import OAuthTokenResolver
    from haystack_integrations.utils.oauth import OAuthRefreshTokenSource
    from haystack_integrations.components.retrievers.google_drive import GoogleDriveRetriever

    pipeline = Pipeline()
    pipeline.add_component(
        "resolver",
        OAuthTokenResolver(
            token_source=OAuthRefreshTokenSource(
                token_url="https://oauth2.googleapis.com/token",
                client_id="aaa-bbb-ccc",
                refresh_token=Secret.from_env_var("GOOGLE_REFRESH_TOKEN"),
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            ),
        ),
    )
    pipeline.add_component("retriever", GoogleDriveRetriever(top_k=5))
    pipeline.connect("resolver.access_token", "retriever.access_token")

    result = pipeline.run({"retriever": {"query": "quarterly roadmap"}})
    documents = result["retriever"]["documents"]
    ```
    """

    def __init__(
        self,
        *,
        include_content: bool = False,
        top_k: int = 10,
        query_filter: str | None = None,
        include_shared_drives: bool = False,
        order_by: str | None = None,
        fields: list[str] | None = None,
        api_base_url: str = DEFAULT_API_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the retriever.

        :param include_content: When `True`, native Google Docs/Sheets/Slides are exported to text and the
            result becomes the `Document` content. Binary files are never downloaded. When `False` (the
            default), `content` is the file `description` or `name` and no export request is made.
        :param top_k: The maximum number of documents to return. Maps to the Drive `pageSize` and is
            paginated when it exceeds a single page.
        :param query_filter: Optional Drive query clause AND-ed with the full-text search term, for example
            `"mimeType != 'application/vnd.google-apps.folder'"` or `"'<folderId>' in parents"`.
        :param include_shared_drives: When `True`, the search spans shared drives as well as the user's My
            Drive (sets `includeItemsFromAllDrives`, `supportsAllDrives`, and `corpora=allDrives`).
        :param order_by: Optional Drive `orderBy` expression, for example `"modifiedTime desc"`.
        :param fields: Optional list of file properties to request via the Drive `fields` selection.
            Defaults to a standard set covering the returned metadata.
        :param api_base_url: The Drive API base URL. Defaults to `https://www.googleapis.com/drive/v3`.
        :param timeout: The HTTP timeout in seconds for each request to the Drive API.
        :param max_retries: The maximum number of retries on HTTP 429 (rate limit), 500, 502, 503,
            or 504 responses. Set to 0 to disable retries.
        :raises GoogleDriveConfigError: If `top_k` is not positive or `max_retries` is negative.
        """
        if top_k <= 0:
            msg = "top_k must be a positive integer."
            raise GoogleDriveConfigError(msg)
        if max_retries < 0:
            msg = "max_retries must be zero or a positive integer."
            raise GoogleDriveConfigError(msg)

        self.include_content = include_content
        self.top_k = top_k
        self.query_filter = query_filter
        self.include_shared_drives = include_shared_drives
        self.order_by = order_by
        self.fields = fields
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    @component.output_types(documents=list[Document])
    def run(self, query: str, access_token: str | Secret, top_k: int | None = None) -> dict[str, list[Document]]:
        """
        Search Google Drive and return the matching documents.

        :param query: The search query string, matched against the full text of files via
            `fullText contains`.
        :param access_token: A delegated Google OAuth bearer token for the user whose Drive is searched,
            typically wired from an upstream `OAuthTokenResolver` (which emits a plain `str`). A `Secret` is also
            accepted and resolved internally.
        :param top_k: Overrides the `top_k` configured at initialization for this run.
        :returns: A dictionary with a `documents` key holding the list of retrieved `Document` objects.
        :raises GoogleDriveConfigError: If `access_token` is a `Secret` that does not resolve to a string.
        :raises GoogleDriveRequestError: If the Drive API returns an error response.
        :raises httpx.HTTPError: If a network-level error occurs (for example a timeout or connection failure).
        """
        token = resolve_access_token(access_token)
        limit = top_k or self.top_k
        documents: list[Document] = []
        page_token: str | None = None
        with httpx.Client(timeout=self.timeout) as client:
            while len(documents) < limit:
                size = min(_MAX_PAGE_SIZE, limit - len(documents))
                params = self._build_params(query, page_token, size)
                payload = self._get(client, token, params)
                files = payload.get("files") or []
                for file in files:
                    content = self._content_for(client, token, file)
                    documents.append(self._file_to_document(file, content))
                page_token = payload.get("nextPageToken")
                if not files or not page_token:
                    break

        return {"documents": documents[:limit]}

    @component.output_types(documents=list[Document])
    async def run_async(
        self, query: str, access_token: str | Secret, top_k: int | None = None
    ) -> dict[str, list[Document]]:
        """
        Asynchronously search Google Drive and return the matching documents.

        :param query: The search query string, matched against the full text of files via
            `fullText contains`.
        :param access_token: A delegated Google OAuth bearer token for the user whose Drive is searched,
            typically wired from an upstream `OAuthTokenResolver` (which emits a plain `str`). A `Secret` is also
            accepted and resolved internally.
        :param top_k: Overrides the `top_k` configured at initialization for this run.
        :returns: A dictionary with a `documents` key holding the list of retrieved `Document` objects.
        :raises GoogleDriveConfigError: If `access_token` is a `Secret` that does not resolve to a string.
        :raises GoogleDriveRequestError: If the Drive API returns an error response.
        :raises httpx.HTTPError: If a network-level error occurs (for example a timeout or connection failure).
        """
        token = resolve_access_token(access_token)
        limit = top_k or self.top_k
        documents: list[Document] = []
        page_token: str | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while len(documents) < limit:
                size = min(_MAX_PAGE_SIZE, limit - len(documents))
                params = self._build_params(query, page_token, size)
                payload = await self._get_async(client, token, params)
                files = payload.get("files") or []
                for file in files:
                    content = await self._content_for_async(client, token, file)
                    documents.append(self._file_to_document(file, content))
                page_token = payload.get("nextPageToken")
                if not files or not page_token:
                    break

        return {"documents": documents[:limit]}

    def _build_params(self, query: str, page_token: str | None, size: int) -> dict[str, Any]:
        """Build the Drive `files.list` query parameters for one page."""
        q = f"fullText contains '{self._escape_query(query)}'"
        if self.query_filter:
            q = f"{q} and ({self.query_filter})"
        fields = tuple(self.fields) if self.fields else _DEFAULT_FILE_FIELDS
        params: dict[str, Any] = {
            "q": q,
            "pageSize": size,
            "fields": f"nextPageToken, files({', '.join(fields)})",
        }
        if page_token:
            params["pageToken"] = page_token
        if self.order_by:
            params["orderBy"] = self.order_by
        if self.include_shared_drives:
            params["includeItemsFromAllDrives"] = True
            params["supportsAllDrives"] = True
            params["corpora"] = "allDrives"

        return params

    @staticmethod
    def _escape_query(query: str) -> str:
        """Escape backslashes and single quotes so the query is safe inside a Drive `q` string literal."""
        return query.replace("\\", "\\\\").replace("'", "\\'")

    def _content_for(self, client: httpx.Client, access_token: str, file: dict[str, Any]) -> str | None:
        """Return the exported text for a native Google file when `include_content` is set, else `None`."""
        if not self.include_content:
            return None
        mime_type = file.get("mimeType")
        file_id = file.get("id")
        if not mime_type or not file_id:
            return None
        export_mime = _EXPORT_MIME_TYPES.get(mime_type)
        if not export_mime:
            return None
        return self._export_content(client, access_token, file_id, export_mime)

    async def _content_for_async(
        self, client: httpx.AsyncClient, access_token: str, file: dict[str, Any]
    ) -> str | None:
        """Async variant of `_content_for`."""
        if not self.include_content:
            return None
        mime_type = file.get("mimeType")
        file_id = file.get("id")
        if not mime_type or not file_id:
            return None
        export_mime = _EXPORT_MIME_TYPES.get(mime_type)
        if not export_mime:
            return None
        return await self._export_content_async(client, access_token, file_id, export_mime)

    def _export_content(self, client: httpx.Client, access_token: str, file_id: str, export_mime: str) -> str | None:
        """Export a native Google file to text. On any failure, log a warning and return `None`."""
        url = f"{self.api_base_url}/files/{file_id}/export"
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            response = client.get(url, headers=headers, params={"mimeType": export_mime})
        except httpx.HTTPError as error:
            logger.warning(
                "Could not export Google Drive file {file_id} as {export_mime}: {error}. "
                "Falling back to metadata content.",
                file_id=file_id,
                export_mime=export_mime,
                error=str(error),
            )
            return None
        if response.is_error:
            logger.warning(
                "Could not export Google Drive file {file_id} as {export_mime} (status {status}). "
                "Falling back to metadata content.",
                file_id=file_id,
                export_mime=export_mime,
                status=response.status_code,
            )
            return None

        return response.text

    async def _export_content_async(
        self, client: httpx.AsyncClient, access_token: str, file_id: str, export_mime: str
    ) -> str | None:
        """Async variant of `_export_content`."""
        url = f"{self.api_base_url}/files/{file_id}/export"
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            response = await client.get(url, headers=headers, params={"mimeType": export_mime})
        except httpx.HTTPError as error:
            logger.warning(
                "Could not export Google Drive file {file_id} as {export_mime}: {error}. "
                "Falling back to metadata content.",
                file_id=file_id,
                export_mime=export_mime,
                error=str(error),
            )
            return None
        if response.is_error:
            logger.warning(
                "Could not export Google Drive file {file_id} as {export_mime} (status {status}). "
                "Falling back to metadata content.",
                file_id=file_id,
                export_mime=export_mime,
                status=response.status_code,
            )
            return None

        return response.text

    def _get(self, client: httpx.Client, access_token: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a `files.list` request, retrying on throttling/transient errors, and return the parsed JSON."""
        url = f"{self.api_base_url}/files"
        headers = {"Authorization": f"Bearer {access_token}"}
        retrying = build_retrying(self.max_retries)
        response: httpx.Response = retrying(client.get, url, headers=headers, params=params)

        return self._parse_response(response)

    async def _get_async(self, client: httpx.AsyncClient, access_token: str, params: dict[str, Any]) -> dict[str, Any]:
        """Async variant of `_get`."""
        url = f"{self.api_base_url}/files"
        headers = {"Authorization": f"Bearer {access_token}"}
        retrying = build_async_retrying(self.max_retries)
        response: httpx.Response = await retrying(client.get, url, headers=headers, params=params)

        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: httpx.Response) -> dict[str, Any]:
        """Return the parsed JSON for a successful response, or raise a descriptive error."""
        if not response.is_error:
            return response.json()

        status = response.status_code
        if status == httpx.codes.UNAUTHORIZED:
            msg = (
                "Google Drive rejected the access token (401 Unauthorized). The token may be expired, invalid, "
                "or missing a required scope (for example https://www.googleapis.com/auth/drive.readonly). The "
                "metadata-only drive.metadata.readonly scope cannot search file content or export documents."
            )
        elif status == httpx.codes.FORBIDDEN:
            msg = f"Google Drive denied the request (403 Forbidden): {response.text}"
        else:
            msg = f"Google Drive API request failed with status {status}: {response.text}"

        raise GoogleDriveRequestError(msg, status_code=status)

    @staticmethod
    def _display_name(identity: dict[str, Any] | None) -> str | None:
        """Extract the display name from a Drive user object (`owners[i]` / `lastModifyingUser`), if present."""
        if not identity:
            return None
        return identity.get("displayName")

    @staticmethod
    def _file_to_document(file: dict[str, Any], content: str | None = None) -> Document:
        """Map a single Drive file resource to a Haystack Document (content + resource metadata)."""
        name = file.get("name")
        text = content or file.get("description") or name or ""

        # Drive usually returns `fileExtension`; derive it from the name when absent (e.g. native Google docs).
        extension = file.get("fileExtension")
        if not extension and name:
            extension = os.path.splitext(name)[1].lstrip(".").lower() or None

        owners = file.get("owners") or []
        created_by = GoogleDriveRetriever._display_name(owners[0]) if owners else None

        meta = {
            "file_name": name,
            "file_id": file.get("id"),
            "web_url": file.get("webViewLink"),
            "mime_type": file.get("mimeType"),
            "file_extension": extension,
            "created_date_time": file.get("createdTime"),
            "last_modified_date_time": file.get("modifiedTime"),
            "created_by": created_by,
            "last_modified_by": GoogleDriveRetriever._display_name(file.get("lastModifyingUser")),
        }
        return Document(content=text, meta={key: value for key, value in meta.items() if value is not None})

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns: The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            include_content=self.include_content,
            top_k=self.top_k,
            query_filter=self.query_filter,
            include_shared_drives=self.include_shared_drives,
            order_by=self.order_by,
            fields=self.fields,
            api_base_url=self.api_base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GoogleDriveRetriever":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns: The deserialized component instance.
        """
        return default_from_dict(cls, data)
