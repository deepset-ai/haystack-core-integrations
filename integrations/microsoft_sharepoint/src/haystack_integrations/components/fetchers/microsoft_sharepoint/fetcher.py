# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import html
import json
import re
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.common.microsoft_sharepoint.errors import SharePointConfigError, SharePointRequestError
from haystack_integrations.common.microsoft_sharepoint.utils import (
    DEFAULT_GRAPH_URL,
    build_async_retrying,
    build_retrying,
    resolve_access_token,
)

logger = logging.getLogger(__name__)

# Microsoft Graph Search entity types. Files are driveItems; SharePoint pages and list rows are listItems.
# `site`/`list`/`drive` are containers with no single extractable content.
_FILE_MARKER = "driveItem"
_LIST_ITEM_MARKER = "listItem"

_JSON_MIME = "application/json"
_HTML_MIME = "text/html"

# Properties needed to render a list item and, if it is a page, to locate it in the Pages API.
_LIST_ITEM_QUERY = "$expand=fields&$select=id,webUrl,contentType,parentReference,sharepointIds"
_PAGE_QUERY = "$expand=canvasLayout"

_CONTENT_DISPOSITION_FILENAME = re.compile(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", re.IGNORECASE)


def _encode_share_url(url: str) -> str:
    """
    Encode a sharing URL into a Microsoft Graph share token.

    See https://learn.microsoft.com/en-us/graph/api/shares-get#encoding-sharing-urls
    """
    encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("utf-8").rstrip("=")
    return f"u!{encoded}"


@component
class MSSharePointFetcher:
    """
    Fetches the full content of Microsoft SharePoint and OneDrive items via the Microsoft Graph API.

    The fetcher complements `MSSharePointRetriever`, which only returns Search snippets and metadata. Wire the
    retriever's `documents` (or a list of `web_url`s) into this fetcher to download the full content. It
    dispatches on the entity type of each hit and always returns `ByteStream`s, ready for a downstream converter
    (for example a `FileTypeRouter` in front of `PyPDFToDocument`, `DOCXToDocument`, `HTMLToDocument`, or a JSON
    converter):

    - **Files** (`driveItem`) are downloaded as their raw bytes (PDF, DOCX, ...).
    - **List items** (`listItem`) are returned as a JSON `ByteStream` of the item's column values (`fields`).
    - **SharePoint pages** (`sitePage`) are returned as an HTML `ByteStream` built from the page's web parts.

    Each `ByteStream`'s `meta` carries `url`, `file_name`, `content_type`, and a normalized `entity_type`
    (`driveItem`, `listItem`, or `sitePage`).

    Everything is resolved through the Microsoft Graph `shares` endpoint (plus the Pages API for pages), so only
    the `web_url` already exposed by the retriever is needed. The fetcher takes a per-user `access_token` as a run
    input, typically wired from an upstream `OAuthResolver`. The token must carry delegated Microsoft Graph
    permissions (for example `Files.Read.All` for files and `Sites.Read.All` for list items and pages).

    ### Usage example
    ```python
    from haystack_integrations.components.fetchers.microsoft_sharepoint import MSSharePointFetcher

    fetcher = MSSharePointFetcher()

    # `access_token` is a per-user delegated Microsoft Graph bearer token.
    result = fetcher.run(
        access_token="my-delegated-graph-token",
        targets=["https://contoso.sharepoint.com/sites/contoso-team/contoso-designs.docx"],
    )
    streams = result["streams"]
    ```

    In a pipeline, connect `MSSharePointRetriever.documents` to the fetcher's `targets` input and an upstream
    component that emits a per-user `access_token` to the fetcher's `access_token` input.
    """

    def __init__(
        self,
        *,
        graph_url: str = DEFAULT_GRAPH_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        raise_on_failure: bool = True,
    ) -> None:
        """
        Initialize the fetcher.

        :param graph_url: The Microsoft Graph base URL. Defaults to `https://graph.microsoft.com/v1.0`.
            Override for sovereign clouds.
        :param timeout: The HTTP timeout in seconds for each request to Microsoft Graph.
        :param max_retries: The maximum number of retries for throttled (HTTP 429) or transient server errors.
        :param raise_on_failure: If `True`, a fetch failure raises an exception. If `False`, the failure is
            logged and the item is skipped, so the other items are still returned.
        :raises SharePointConfigError: If `max_retries` is negative.
        """
        if max_retries < 0:
            msg = "max_retries must be zero or a positive integer."
            raise SharePointConfigError(msg)

        self.graph_url = graph_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.raise_on_failure = raise_on_failure

    @component.output_types(streams=list[ByteStream])
    def run(
        self,
        access_token: str | Secret,
        targets: list[Document] | list[str],
    ) -> dict[str, list[ByteStream]]:
        """
        Fetch the content of SharePoint and OneDrive items and return them as `ByteStream`s.

        :param access_token: A delegated Microsoft Graph bearer token for the user whose content is fetched,
            typically wired from an upstream `OAuthResolver` (which emits a plain `str`). A `Secret` is also
            accepted and resolved internally.
        :param targets: The items to fetch, as either `Document`s emitted by `MSSharePointRetriever` or raw
            SharePoint/OneDrive `web_url` strings (the two may also be mixed in one list). For a `Document`, the
            `web_url` in its meta is fetched and `file_name`, `mime_type`, `entity_type`, and the SharePoint IDs
            are reused when present; container hits with no extractable content (for example `site` or `list`) are
            skipped. For a raw URL, the item is probed as a file and falls back to a list item.
        :returns: A dictionary with a `streams` key holding the fetched content as `ByteStream` objects. Each
            stream's `meta` carries `url`, `file_name`, `content_type`, and `entity_type`.
        :raises SharePointConfigError: If an item is neither a `Document` nor a `str`, or if `access_token` is a
            `Secret` that does not resolve to a string.
        :raises SharePointRequestError: If a fetch fails and `raise_on_failure` is `True`.
        """
        token = resolve_access_token(access_token)
        resolved = self._resolve_targets(targets)
        streams: list[ByteStream] = []
        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            for url, hints in resolved:
                stream = self._process(client, token, url, hints)
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
        Asynchronously fetch the content of SharePoint and OneDrive items and return them as `ByteStream`s.

        :param access_token: A delegated Microsoft Graph bearer token for the user whose content is fetched,
            typically wired from an upstream `OAuthResolver` (which emits a plain `str`). A `Secret` is also
            accepted and resolved internally.
        :param targets: The items to fetch, as either `Document`s emitted by `MSSharePointRetriever` or raw
            SharePoint/OneDrive `web_url` strings (the two may also be mixed in one list). For a `Document`, the
            `web_url` in its meta is fetched and `file_name`, `mime_type`, `entity_type`, and the SharePoint IDs
            are reused when present; container hits with no extractable content (for example `site` or `list`) are
            skipped. For a raw URL, the item is probed as a file and falls back to a list item.
        :returns: A dictionary with a `streams` key holding the fetched content as `ByteStream` objects. Each
            stream's `meta` carries `url`, `file_name`, `content_type`, and `entity_type`.
        :raises SharePointConfigError: If an item is neither a `Document` nor a `str`, or if `access_token` is a
            `Secret` that does not resolve to a string.
        :raises SharePointRequestError: If a fetch fails and `raise_on_failure` is `True`.
        """
        token = resolve_access_token(access_token)
        resolved = self._resolve_targets(targets)
        streams: list[ByteStream] = []
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            for url, hints in resolved:
                stream = await self._process_async(client, token, url, hints)
                if stream is not None:
                    streams.append(stream)

        return {"streams": streams}

    def _resolve_targets(self, targets: list[Document] | list[str]) -> list[tuple[str, dict[str, Any]]]:
        """Resolve the input targets into `(url, hints)` pairs to fetch, dispatching each item by its type."""
        resolved: list[tuple[str, dict[str, Any]]] = []
        for item in targets:
            if isinstance(item, Document):
                target = self._resolve_document(item)
                if target is not None:
                    resolved.append(target)
            elif isinstance(item, str):
                resolved.append((item, {}))
            else:
                msg = f"targets items must be a Document or a str (web_url), got {type(item).__name__}."
                raise SharePointConfigError(msg)

        return resolved

    @staticmethod
    def _resolve_document(document: Document) -> tuple[str, dict[str, Any]] | None:
        """Turn a retriever `Document` into a `(url, hints)` pair, or `None` if it has nothing to fetch."""
        meta = document.meta or {}
        url = meta.get("web_url")
        if not url:
            logger.warning("Skipping document {id}: no `web_url` in its meta, nothing to fetch.", id=document.id)
            return None
        entity_type = meta.get("entity_type")
        if entity_type and _FILE_MARKER not in entity_type and _LIST_ITEM_MARKER not in entity_type:
            logger.warning(
                "Skipping {url}: entity_type {entity_type} has no extractable content.",
                url=url,
                entity_type=entity_type,
            )
            return None
        hints = {
            "entity_type": entity_type,
            "file_name": meta.get("file_name"),
            "mime_type": meta.get("mime_type"),
            "site_id": meta.get("site_id"),
            "list_id": meta.get("list_id"),
            "list_item_id": meta.get("list_item_id"),
            "list_item_unique_id": meta.get("list_item_unique_id"),
        }
        return url, hints

    # --- synchronous fetch path ---------------------------------------------------------------------------------

    def _process(self, client: httpx.Client, token: str, url: str, hints: dict[str, Any]) -> ByteStream | None:
        """Fetch a single item, honoring `raise_on_failure` for any error."""
        try:
            return self._dispatch(client, token, url, hints)
        except (SharePointRequestError, httpx.HTTPError) as error:
            if self.raise_on_failure:
                raise
            logger.warning("Failed to fetch {url} from SharePoint: {error}", url=url, error=str(error))
            return None

    def _dispatch(self, client: httpx.Client, token: str, url: str, hints: dict[str, Any]) -> ByteStream | None:
        """Route a target to the right handler based on its entity type."""
        entity_type = hints.get("entity_type")
        if entity_type and _FILE_MARKER in entity_type:
            return self._fetch_file(client, token, url, hints)
        if entity_type and _LIST_ITEM_MARKER in entity_type:
            return self._fetch_list_item(client, token, url, hints)
        # No entity-type hint (raw URL input): probe as a file, fall back to a list item.
        return self._fetch_unknown(client, token, url, hints)

    def _fetch_file(self, client: httpx.Client, token: str, url: str, hints: dict[str, Any]) -> ByteStream:
        response = self._get(client, token, self._content_url(url))
        if response.is_error:
            raise self._build_error(response, url)
        return self._file_stream(response, url, hints)

    def _fetch_unknown(self, client: httpx.Client, token: str, url: str, hints: dict[str, Any]) -> ByteStream | None:
        response = self._get(client, token, self._content_url(url))
        if response.status_code == httpx.codes.NOT_FOUND:
            return self._fetch_list_item(client, token, url, hints)
        if response.is_error:
            raise self._build_error(response, url)
        return self._file_stream(response, url, hints)

    def _fetch_list_item(self, client: httpx.Client, token: str, url: str, hints: dict[str, Any]) -> ByteStream:
        list_item = self._get_json(client, token, self._list_item_url(url, hints), url)
        if self._is_page(list_item):
            site_id, page_id = self._page_ids(list_item, hints)
            if site_id and page_id:
                page = self._get_json(client, token, self._page_url(site_id, page_id), url)
                return self._page_stream(page, url, hints)
        return self._list_item_stream(list_item, url, hints)

    def _get(self, client: httpx.Client, token: str, url: str) -> httpx.Response:
        """Issue a GET to Microsoft Graph, retrying on throttling/transient errors."""
        headers = {"Authorization": f"Bearer {token}"}
        retrying = build_retrying(self.max_retries)
        return retrying(client.get, url, headers=headers)

    def _get_json(self, client: httpx.Client, token: str, url: str, source_url: str) -> dict[str, Any]:
        response = self._get(client, token, url)
        if response.is_error:
            raise self._build_error(response, source_url)
        return response.json()

    # --- asynchronous fetch path --------------------------------------------------------------------------------

    async def _process_async(
        self, client: httpx.AsyncClient, token: str, url: str, hints: dict[str, Any]
    ) -> ByteStream | None:
        try:
            return await self._dispatch_async(client, token, url, hints)
        except (SharePointRequestError, httpx.HTTPError) as error:
            if self.raise_on_failure:
                raise
            logger.warning("Failed to fetch {url} from SharePoint: {error}", url=url, error=str(error))
            return None

    async def _dispatch_async(
        self, client: httpx.AsyncClient, token: str, url: str, hints: dict[str, Any]
    ) -> ByteStream | None:
        entity_type = hints.get("entity_type")
        if entity_type and _FILE_MARKER in entity_type:
            return await self._fetch_file_async(client, token, url, hints)
        if entity_type and _LIST_ITEM_MARKER in entity_type:
            return await self._fetch_list_item_async(client, token, url, hints)
        return await self._fetch_unknown_async(client, token, url, hints)

    async def _fetch_file_async(
        self, client: httpx.AsyncClient, token: str, url: str, hints: dict[str, Any]
    ) -> ByteStream:
        response = await self._get_async(client, token, self._content_url(url))
        if response.is_error:
            raise self._build_error(response, url)
        return self._file_stream(response, url, hints)

    async def _fetch_unknown_async(
        self, client: httpx.AsyncClient, token: str, url: str, hints: dict[str, Any]
    ) -> ByteStream | None:
        response = await self._get_async(client, token, self._content_url(url))
        if response.status_code == httpx.codes.NOT_FOUND:
            return await self._fetch_list_item_async(client, token, url, hints)
        if response.is_error:
            raise self._build_error(response, url)
        return self._file_stream(response, url, hints)

    async def _fetch_list_item_async(
        self, client: httpx.AsyncClient, token: str, url: str, hints: dict[str, Any]
    ) -> ByteStream:
        list_item = await self._get_json_async(client, token, self._list_item_url(url, hints), url)
        if self._is_page(list_item):
            site_id, page_id = self._page_ids(list_item, hints)
            if site_id and page_id:
                page = await self._get_json_async(client, token, self._page_url(site_id, page_id), url)
                return self._page_stream(page, url, hints)
        return self._list_item_stream(list_item, url, hints)

    async def _get_async(self, client: httpx.AsyncClient, token: str, url: str) -> httpx.Response:
        headers = {"Authorization": f"Bearer {token}"}
        retrying = build_async_retrying(self.max_retries)
        return await retrying(client.get, url, headers=headers)

    async def _get_json_async(self, client: httpx.AsyncClient, token: str, url: str, source_url: str) -> dict[str, Any]:
        response = await self._get_async(client, token, url)
        if response.is_error:
            raise self._build_error(response, source_url)
        return response.json()

    # --- URL builders -------------------------------------------------------------------------------------------

    def _content_url(self, url: str) -> str:
        return f"{self.graph_url}/shares/{_encode_share_url(url)}/driveItem/content"

    def _list_item_url(self, url: str, hints: dict[str, Any]) -> str:
        site_id, list_id, item_id = hints.get("site_id"), hints.get("list_id"), hints.get("list_item_id")
        if site_id and list_id and item_id:
            return f"{self.graph_url}/sites/{site_id}/lists/{list_id}/items/{item_id}?{_LIST_ITEM_QUERY}"
        return f"{self.graph_url}/shares/{_encode_share_url(url)}/listItem?{_LIST_ITEM_QUERY}"

    def _page_url(self, site_id: str, page_id: str) -> str:
        return f"{self.graph_url}/sites/{site_id}/pages/{page_id}/microsoft.graph.sitePage?{_PAGE_QUERY}"

    # --- response handling (shared between sync and async) ------------------------------------------------------

    @staticmethod
    def _is_page(list_item: dict[str, Any]) -> bool:
        """Detect whether a list item is a SharePoint page (and therefore has web-part content)."""
        content_type_name = ((list_item.get("contentType") or {}).get("name") or "").lower()
        if "page" in content_type_name:
            return True
        if (list_item.get("webUrl") or "").lower().endswith(".aspx"):
            return True
        return "CanvasContent1" in (list_item.get("fields") or {})

    @staticmethod
    def _page_ids(list_item: dict[str, Any], hints: dict[str, Any]) -> tuple[str | None, str | None]:
        """Extract the `(site_id, page_id)` needed to read a page through the Pages API."""
        site_id = hints.get("site_id") or (list_item.get("parentReference") or {}).get("siteId")
        page_id = hints.get("list_item_unique_id") or (list_item.get("sharepointIds") or {}).get("listItemUniqueId")
        return site_id, page_id

    def _file_stream(self, response: httpx.Response, url: str, hints: dict[str, Any]) -> ByteStream:
        content_type = hints.get("mime_type") or self._content_type(response)
        file_name = hints.get("file_name") or self._file_name(response, url)
        return self._byte_stream(response.content, content_type, url, file_name, _FILE_MARKER)

    def _list_item_stream(self, list_item: dict[str, Any], url: str, hints: dict[str, Any]) -> ByteStream:
        fields = list_item.get("fields") or {}
        data = json.dumps(fields, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        file_name = fields.get("FileLeafRef") or fields.get("Title") or hints.get("file_name") or self._url_name(url)
        return self._byte_stream(data, _JSON_MIME, url, file_name, _LIST_ITEM_MARKER)

    def _page_stream(self, page: dict[str, Any], url: str, hints: dict[str, Any]) -> ByteStream:
        data = self._render_page_html(page)
        file_name = page.get("name") or hints.get("file_name") or self._url_name(url)
        return self._byte_stream(data, _HTML_MIME, url, file_name, "sitePage")

    @staticmethod
    def _byte_stream(
        data: bytes, content_type: str | None, url: str, file_name: str | None, entity_type: str
    ) -> ByteStream:
        meta = {"url": url, "file_name": file_name, "content_type": content_type, "entity_type": entity_type}
        return ByteStream(
            data=data, mime_type=content_type, meta={key: value for key, value in meta.items() if value is not None}
        )

    @staticmethod
    def _render_page_html(page: dict[str, Any]) -> bytes:
        """Build an HTML document from a sitePage's title and the inner HTML of its web parts."""
        parts: list[str] = []
        title = page.get("title") or page.get("name")
        if title:
            parts.append(f"<h1>{html.escape(title)}</h1>")

        canvas = page.get("canvasLayout") or {}
        webparts: list[dict[str, Any]] = []
        for section in canvas.get("horizontalSections") or []:
            for column in section.get("columns") or []:
                webparts.extend(column.get("webparts") or [])
        webparts.extend((canvas.get("verticalSection") or {}).get("webparts") or [])

        for webpart in webparts:
            inner_html = webpart.get("innerHtml")
            if inner_html:
                parts.append(inner_html)

        return "\n".join(parts).encode("utf-8")

    @staticmethod
    def _build_error(response: httpx.Response, url: str) -> SharePointRequestError:
        """Build a descriptive `SharePointRequestError` for an error response."""
        status = response.status_code
        if status == httpx.codes.UNAUTHORIZED:
            msg = (
                "Microsoft Graph rejected the access token (401 Unauthorized). The token may be expired, invalid, "
                "or missing the required delegated scopes (for example Files.Read.All / Sites.Read.All)."
            )
        elif status == httpx.codes.FORBIDDEN:
            msg = f"Microsoft Graph denied access to {url} (403 Forbidden): {response.text}"
        elif status == httpx.codes.NOT_FOUND:
            msg = f"Microsoft Graph could not find {url} (404 Not Found): {response.text}"
        else:
            msg = f"Microsoft Graph fetch of {url} failed with status {status}: {response.text}"

        return SharePointRequestError(msg, status_code=status)

    @staticmethod
    def _content_type(response: httpx.Response) -> str | None:
        """Extract the bare content type (without parameters) from a response."""
        content_type = response.headers.get("Content-Type", "")
        return content_type.split(";")[0].strip() or None

    @staticmethod
    def _file_name(response: httpx.Response, url: str) -> str | None:
        """Derive a file name from the `Content-Disposition` header, falling back to the URL's last path segment."""
        disposition = response.headers.get("Content-Disposition", "")
        match = _CONTENT_DISPOSITION_FILENAME.search(disposition)
        if match:
            return unquote(match.group(1)).strip() or None
        return MSSharePointFetcher._url_name(url)

    @staticmethod
    def _url_name(url: str) -> str | None:
        """Return the last path segment of a URL, unquoted."""
        name = urlparse(url).path.rsplit("/", 1)[-1]
        return unquote(name) or None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns: The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            graph_url=self.graph_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MSSharePointFetcher":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns: The deserialized component instance.
        """
        return default_from_dict(cls, data)
