# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import html
import os
import re
from typing import Any

import httpx
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret

from haystack_integrations.common.microsoft_sharepoint.errors import SharePointConfigError, SharePointRequestError
from haystack_integrations.common.microsoft_sharepoint.utils import (
    DEFAULT_GRAPH_URL,
    build_async_retrying,
    build_retrying,
    resolve_access_token,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENTITY_TYPES = ("driveItem", "listItem")
# Microsoft Graph caps a single search page at 500 results; larger `top_k` values are paginated.
_MAX_PAGE_SIZE = 500

# Search summaries come back as snippets with hit-highlight markup, e.g.
# "<c0>Contoso</c0> Detailed Design <ddd/>". These patterns turn that into plain text.
_HIGHLIGHT_PATTERN = re.compile(r"</?c\d+>")
_ELLIPSIS_PATTERN = re.compile(r"<ddd\s*/>")


@component
class MSSharePointRetriever:
    """
    Retrieves content from Microsoft SharePoint and OneDrive via the Microsoft Search (Graph) API.

    Given a query, the retriever calls `POST /search/query` and maps each hit to a Haystack `Document`
    whose `content` is the search snippet and whose `meta` carries the resource metadata (`file_name`,
    `web_url`, `entity_type`, `created_date_time`, `last_modified_date_time`, `created_by`, `last_modified_by`,
    `mime_type`, and `file_extension`), plus the SharePoint identifiers a downstream fetcher needs to read
    list items and pages by ID (`site_id`, `list_id`, `list_item_id`, `list_item_unique_id`). It does not
    download or convert the underlying files. Compose a downstream fetcher/converter (such as
    `MSSharePointFetcher`) when full content is needed.

    The retriever takes a per-user `access_token` as a run input, typically wired
    from an upstream `OAuthResolver`. The token must carry delegated Microsoft Graph permissions
    (for example `Files.Read.All` and, for site/list scoping, `Sites.Read.All`). The Search API supports
    delegated permissions only.

    ### Usage example
    ```python
    from haystack_integrations.components.retrievers.microsoft_sharepoint import (
        MSSharePointRetriever,
    )

    retriever = MSSharePointRetriever(top_k=5)

    # `access_token` is a per-user delegated Microsoft Graph bearer token.
    result = retriever.run(
        query="quarterly roadmap", access_token="my-delegated-graph-token"
    )
    documents = result["documents"]
    ```

    In a pipeline, connect an upstream component that emits a per-user `access_token` to the retriever's
    `access_token` input. See the integration documentation for a full example that obtains the token from
    an OAuth provider.
    """

    def __init__(
        self,
        *,
        entity_types: list[str] | None = None,
        top_k: int = 10,
        fields: list[str] | None = None,
        query_template: str | None = None,
        graph_url: str = DEFAULT_GRAPH_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the retriever.

        :param entity_types: The Microsoft Search entity types to query. Defaults to `["driveItem", "listItem"]`,
            which covers files, folders, SharePoint pages and news, and list items. Other valid values are
            `"list"` and `"site"`. See the supported values and combinations in the
            [Microsoft docs](https://learn.microsoft.com/en-us/graph/api/resources/searchrequest).
        :param top_k: The maximum number of documents to return. Maps to the Search API `size` and is paginated
            when it exceeds a single page.
        :param fields: Optional list of resource properties to request via the Search API `fields` selection
            (only honored for `listItem` and `driveItem` entity types). See
            [Get selected properties](https://learn.microsoft.com/en-us/graph/api/resources/search-api-overview#get-selected-properties).
        :param query_template: Optional query template used to scope the search, for example
            `'{searchTerms} path:"https://contoso.sharepoint.com/sites/Team"'`. The literal `{searchTerms}`
            placeholder is replaced by the run-time query. The template uses
            [Keyword Query Language (KQL)](https://learn.microsoft.com/en-us/sharepoint/dev/general-development/keyword-query-language-kql-syntax-reference).
        :param graph_url: The Microsoft Graph base URL. Defaults to `https://graph.microsoft.com/v1.0`.
            Override for sovereign clouds.
        :param timeout: The HTTP timeout in seconds for each request to Microsoft Graph.
        :param max_retries: The maximum number of retries for throttled (HTTP 429) or transient server errors.
        :raises SharePointConfigError: If `entity_types` is empty, `top_k` is not positive, or `max_retries` is
            negative.
        """
        if entity_types is None:
            entity_types = list(_DEFAULT_ENTITY_TYPES)
        else:
            entity_types = list(entity_types)
            if not entity_types:
                msg = "entity_types must contain at least one Microsoft Search entity type."
                raise SharePointConfigError(msg)
        if top_k <= 0:
            msg = "top_k must be a positive integer."
            raise SharePointConfigError(msg)
        if max_retries < 0:
            msg = "max_retries must be zero or a positive integer."
            raise SharePointConfigError(msg)

        self.entity_types = entity_types
        self.top_k = top_k
        self.fields = fields
        self.query_template = query_template
        self.graph_url = graph_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    @component.output_types(documents=list[Document])
    def run(self, query: str, access_token: str | Secret, top_k: int | None = None) -> dict[str, list[Document]]:
        """
        Search SharePoint and OneDrive and return the matching documents.

        :param query: The search query string. Filter results by embedding Keyword Query Language (KQL)
            operators directly in the query, for example `filetype:docx`, `author:"Jane Doe"`, or
            `path:"https://contoso.sharepoint.com/sites/Team"`. See the
            [KQL syntax reference](https://learn.microsoft.com/en-us/sharepoint/dev/general-development/keyword-query-language-kql-syntax-reference).
        :param access_token: A delegated Microsoft Graph bearer token for the user whose content is searched,
            typically wired from an upstream `OAuthResolver` (which emits a plain `str`). A `Secret` is also
            accepted and resolved internally.
        :param top_k: Overrides the `top_k` configured at initialization for this run.
        :returns: A dictionary with a `documents` key holding the list of retrieved `Document` objects.
        :raises SharePointConfigError: If `access_token` is a `Secret` that does not resolve to a string.
        :raises SharePointRequestError: If Microsoft Graph returns an error response.
        """
        token = resolve_access_token(access_token)
        limit = top_k or self.top_k
        documents: list[Document] = []
        offset = 0
        with httpx.Client(timeout=self.timeout) as client:
            while len(documents) < limit:
                size = min(_MAX_PAGE_SIZE, limit - len(documents))
                request_body = self._build_request_body(query, offset, size)
                payload = self._post(client, token, request_body)
                container = self._first_container(payload)
                hits = container.get("hits") or []
                documents.extend(self._hit_to_document(hit) for hit in hits)
                if not hits or not container.get("moreResultsAvailable"):
                    break
                offset += len(hits)

        return {"documents": documents[:limit]}

    @component.output_types(documents=list[Document])
    async def run_async(
        self, query: str, access_token: str | Secret, top_k: int | None = None
    ) -> dict[str, list[Document]]:
        """
        Asynchronously search SharePoint and OneDrive and return the matching documents.

        :param query: The search query string. Filter results by embedding Keyword Query Language (KQL)
            operators directly in the query, for example `filetype:docx`, `author:"Jane Doe"`, or
            `path:"https://contoso.sharepoint.com/sites/Team"`. See the
            [KQL syntax reference](https://learn.microsoft.com/en-us/sharepoint/dev/general-development/keyword-query-language-kql-syntax-reference).
        :param access_token: A delegated Microsoft Graph bearer token for the user whose content is searched,
            typically wired from an upstream `OAuthResolver` (which emits a plain `str`). A `Secret` is also
            accepted and resolved internally.
        :param top_k: Overrides the `top_k` configured at initialization for this run.
        :returns: A dictionary with a `documents` key holding the list of retrieved `Document` objects.
        :raises SharePointConfigError: If `access_token` is a `Secret` that does not resolve to a string.
        :raises SharePointRequestError: If Microsoft Graph returns an error response.
        """
        token = resolve_access_token(access_token)
        limit = top_k or self.top_k
        documents: list[Document] = []
        offset = 0
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while len(documents) < limit:
                size = min(_MAX_PAGE_SIZE, limit - len(documents))
                request_body = self._build_request_body(query, offset, size)
                payload = await self._post_async(client, token, request_body)
                container = self._first_container(payload)
                hits = container.get("hits") or []
                documents.extend(self._hit_to_document(hit) for hit in hits)
                if not hits or not container.get("moreResultsAvailable"):
                    break
                offset += len(hits)

        return {"documents": documents[:limit]}

    def _build_request_body(self, query: str, offset: int, size: int) -> dict[str, Any]:
        """
        Build the Microsoft Search `POST /search/query` request body for one page.

        See the searchRequest resource for the full set of supported properties:
        https://learn.microsoft.com/en-us/graph/api/resources/searchrequest
        """
        query_obj: dict[str, Any] = {"queryString": query}
        if self.query_template:
            query_obj["queryTemplate"] = self.query_template
        request: dict[str, Any] = {
            "entityTypes": self.entity_types,
            "query": query_obj,
            "from": offset,
            "size": size,
        }
        if self.fields:
            request["fields"] = self.fields

        return {"requests": [request]}

    @staticmethod
    def _first_container(payload: dict[str, Any]) -> dict[str, Any]:
        """Return the first hitsContainer from a Microsoft Search response, or an empty dict."""
        value = payload.get("value") or []
        if not value:
            return {}
        containers = value[0].get("hitsContainers") or []

        return containers[0] if containers else {}

    def _post(self, client: httpx.Client, access_token: str, body: dict[str, Any]) -> dict[str, Any]:
        """Send a search request, retrying on throttling/transient errors, and return the parsed JSON."""
        url = f"{self.graph_url}/search/query"
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        retrying = build_retrying(self.max_retries)
        response: httpx.Response = retrying(client.post, url, headers=headers, json=body)

        return self._parse_response(response)

    async def _post_async(self, client: httpx.AsyncClient, access_token: str, body: dict[str, Any]) -> dict[str, Any]:
        """Async variant of `_post`."""
        url = f"{self.graph_url}/search/query"
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        retrying = build_async_retrying(self.max_retries)
        response: httpx.Response = await retrying(client.post, url, headers=headers, json=body)

        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: httpx.Response) -> dict[str, Any]:
        """Return the parsed JSON for a successful response, or raise a descriptive error."""
        if not response.is_error:
            return response.json()

        status = response.status_code
        if status == httpx.codes.UNAUTHORIZED:
            msg = (
                "Microsoft Graph rejected the access token (401 Unauthorized). The token may be expired, invalid, "
                "or missing the required delegated scopes (for example Files.Read.All / Sites.Read.All)."
            )
        elif status == httpx.codes.FORBIDDEN:
            msg = f"Microsoft Graph denied the search request (403 Forbidden): {response.text}"
        else:
            msg = f"Microsoft Graph search request failed with status {status}: {response.text}"

        raise SharePointRequestError(msg, status_code=status)

    @staticmethod
    def _clean_summary(summary: str) -> str:
        """Strip Microsoft Search hit-highlight markup from a result summary and unescape entities."""
        if not summary:
            return ""
        text = _ELLIPSIS_PATTERN.sub("…", summary)
        text = _HIGHLIGHT_PATTERN.sub("", text)
        return html.unescape(text).strip()

    @staticmethod
    def _display_name(identity: dict[str, Any] | None) -> str | None:
        """Extract the user display name from a Microsoft Graph identitySet, if present."""
        if not identity:
            return None
        user = identity.get("user") or {}
        return user.get("displayName")

    @staticmethod
    def _hit_to_document(hit: dict[str, Any]) -> Document:
        """Map a single Microsoft Search hit to a Haystack Document (snippet content + resource metadata)."""
        resource = hit.get("resource") or {}

        name = resource.get("name") or resource.get("displayName")
        content = MSSharePointRetriever._clean_summary(hit.get("summary", "")) or name or ""

        file_facet = resource.get("file") or {}
        # Microsoft Graph has no dedicated file-extension field, so derive it from the name.
        extension = os.path.splitext(name or "")[1].lstrip(".").lower()

        parent_reference = resource.get("parentReference") or {}
        # `sharepointIds` may sit on the resource itself or under its parentReference, depending on the entity type.
        sharepoint_ids = resource.get("sharepointIds") or parent_reference.get("sharepointIds") or {}

        meta = {
            "file_name": name,
            "web_url": resource.get("webUrl"),
            "entity_type": resource.get("@odata.type"),
            "created_date_time": resource.get("createdDateTime"),
            "last_modified_date_time": resource.get("lastModifiedDateTime"),
            "created_by": MSSharePointRetriever._display_name(resource.get("createdBy")),
            "last_modified_by": MSSharePointRetriever._display_name(resource.get("lastModifiedBy")),
            "mime_type": file_facet.get("mimeType"),
            "file_extension": extension or None,
            # SharePoint identifiers a downstream fetcher needs to read list items and pages by ID
            "site_id": parent_reference.get("siteId") or sharepoint_ids.get("siteId"),
            "list_id": sharepoint_ids.get("listId"),
            "list_item_id": sharepoint_ids.get("listItemId"),
            "list_item_unique_id": sharepoint_ids.get("listItemUniqueId"),
        }
        return Document(content=content, meta={key: value for key, value in meta.items() if value is not None})

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns: The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            entity_types=self.entity_types,
            top_k=self.top_k,
            fields=self.fields,
            query_template=self.query_template,
            graph_url=self.graph_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MSSharePointRetriever":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns: The deserialized component instance.
        """
        return default_from_dict(cls, data)
