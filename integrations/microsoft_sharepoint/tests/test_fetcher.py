# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import base64
import json
import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from haystack import Document, Pipeline
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.components.fetchers.microsoft_sharepoint import (
    MSSharePointFetcher,
    SharePointConfigError,
    SharePointRequestError,
)
from haystack_integrations.components.fetchers.microsoft_sharepoint.fetcher import _encode_share_url
from haystack_integrations.components.retrievers.microsoft_sharepoint import MSSharePointRetriever

MODULE = "haystack_integrations.components.fetchers.microsoft_sharepoint.fetcher"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
FILE_URL = "https://contoso.sharepoint.com/sites/contoso-team/contoso-designs.docx"
LIST_ITEM_URL = "https://contoso.sharepoint.com/sites/team/Lists/Tasks/12_.000"
PAGE_URL = "https://contoso.sharepoint.com/sites/team/SitePages/Welcome.aspx"

LIST_ITEM = {
    "id": "12",
    "webUrl": LIST_ITEM_URL,
    "contentType": {"name": "Item"},
    "parentReference": {"siteId": "site-id"},
    "sharepointIds": {"listItemUniqueId": "item-guid"},
    "fields": {"Title": "Buy milk", "Status": "Open", "Priority": "High"},
}
PAGE_LIST_ITEM = {
    "id": "5",
    "webUrl": PAGE_URL,
    "contentType": {"name": "Site Page"},
    "parentReference": {"siteId": "site-id"},
    "sharepointIds": {"listItemUniqueId": "page-guid"},
    "fields": {"Title": "Welcome", "CanvasContent1": "<div>raw</div>"},
}
PAGE = {
    "name": "Welcome.aspx",
    "title": "Welcome",
    "canvasLayout": {
        "horizontalSections": [
            {"columns": [{"webparts": [{"innerHtml": "<p>Hello world</p>"}, {"innerHtml": "<p>Second</p>"}]}]}
        ],
        "verticalSection": {"webparts": [{"innerHtml": "<aside>Side</aside>"}]},
    },
}


def _binary_response(
    status_code: int = 200, content: bytes = b"file-bytes", headers: dict | None = None
) -> httpx.Response:
    return httpx.Response(
        status_code, content=content, headers=headers, request=httpx.Request("GET", "https://graph.microsoft.com/x")
    )


def _json_response(body: dict | None = None, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=body if body is not None else {},
        request=httpx.Request("GET", "https://graph.microsoft.com/x"),
    )


def _route(handlers: list[tuple[str, httpx.Response]]):
    """Build a `get` side effect that returns a response based on a substring of the requested URL."""

    def handler(url, *_args, **_kwargs):
        for substring, response in handlers:
            if substring in url:
                return response
        msg = f"unexpected URL: {url}"
        raise AssertionError(msg)

    return handler


def _drive_item_document(url: str = FILE_URL, **meta) -> Document:
    base = {
        "web_url": url,
        "file_name": "Contoso Detailed Design.docx",
        "entity_type": "#microsoft.graph.driveItem",
        "mime_type": DOCX_MIME,
    }
    base.update(meta)
    return Document(content="snippet", meta=base)


LIST_ITEM_IDS = {"site_id": "site-id", "list_id": "list-id", "list_item_id": "12"}
PAGE_IDS = {"site_id": "site-id", "list_id": "pages-list-id", "list_item_id": "5", "list_item_unique_id": "page-guid"}


def _list_item_document(url: str = LIST_ITEM_URL, ids: dict | None = LIST_ITEM_IDS) -> Document:
    meta = {"web_url": url, "entity_type": "#microsoft.graph.listItem", **(ids or {})}
    return Document(content="snippet", meta=meta)


def _page_document(url: str = PAGE_URL) -> Document:
    return Document(content="snippet", meta={"web_url": url, "entity_type": "#microsoft.graph.listItem", **PAGE_IDS})


class TestInit:
    def test_defaults(self):
        fetcher = MSSharePointFetcher()
        assert fetcher.graph_url == "https://graph.microsoft.com/v1.0"
        assert fetcher.timeout == 30.0
        assert fetcher.max_retries == 3
        assert fetcher.raise_on_failure is True

    def test_graph_url_trailing_slash_is_stripped(self):
        assert MSSharePointFetcher(graph_url="https://graph.microsoft.com/v1.0/").graph_url == (
            "https://graph.microsoft.com/v1.0"
        )

    def test_negative_max_retries_raises(self):
        with pytest.raises(SharePointConfigError):
            MSSharePointFetcher(max_retries=-1)


class TestSerialization:
    def test_to_dict(self):
        fetcher = MSSharePointFetcher(
            graph_url="https://graph.microsoft.us/v1.0", timeout=10.0, max_retries=1, raise_on_failure=False
        )
        assert fetcher.to_dict() == {
            "type": f"{MODULE}.MSSharePointFetcher",
            "init_parameters": {
                "graph_url": "https://graph.microsoft.us/v1.0",
                "timeout": 10.0,
                "max_retries": 1,
                "raise_on_failure": False,
            },
        }

    def test_from_dict_round_trip(self):
        fetcher = MSSharePointFetcher(timeout=5.0, raise_on_failure=False)
        restored = MSSharePointFetcher.from_dict(fetcher.to_dict())
        assert restored.to_dict() == fetcher.to_dict()


class TestShareUrlEncoding:
    def test_round_trips_to_original_url(self):
        token = _encode_share_url(FILE_URL)
        assert token.startswith("u!")
        encoded = token[2:]
        padded = encoded + "=" * (-len(encoded) % 4)
        assert base64.urlsafe_b64decode(padded).decode("utf-8") == FILE_URL

    def test_is_url_safe(self):
        token = _encode_share_url("https://contoso.sharepoint.com/sites/Team/~weird?a=b&c=d/ef")
        assert "+" not in token
        assert "/" not in token[2:]


class TestFiles:
    def test_downloads_drive_item_document(self):
        fetcher = MSSharePointFetcher()
        with patch.object(httpx.Client, "get", return_value=_binary_response(content=b"docx-bytes")):
            streams = fetcher.run(access_token="tok", targets=[_drive_item_document()])["streams"]

        assert len(streams) == 1
        stream = streams[0]
        assert isinstance(stream, ByteStream)
        assert stream.data == b"docx-bytes"
        assert stream.mime_type == DOCX_MIME
        assert stream.meta == {
            "url": FILE_URL,
            "file_name": "Contoso Detailed Design.docx",
            "content_type": DOCX_MIME,
            "entity_type": "driveItem",
        }

    def test_downloads_from_url_uses_response_headers(self):
        fetcher = MSSharePointFetcher()
        headers = {
            "Content-Type": "application/pdf; charset=binary",
            "Content-Disposition": 'attachment; filename="report.pdf"',
        }
        with patch.object(httpx.Client, "get", return_value=_binary_response(headers=headers)):
            streams = fetcher.run(access_token="tok", targets=[FILE_URL])["streams"]

        assert streams[0].meta["content_type"] == "application/pdf"
        assert streams[0].mime_type == "application/pdf"
        assert streams[0].meta["file_name"] == "report.pdf"
        assert streams[0].meta["entity_type"] == "driveItem"

    def test_builds_content_url_and_auth_header(self):
        fetcher = MSSharePointFetcher()
        with patch.object(httpx.Client, "get", return_value=_binary_response()) as mock_get:
            fetcher.run(access_token="my-token", targets=[_drive_item_document()])

        url = mock_get.call_args.args[0]
        assert url == f"https://graph.microsoft.com/v1.0/shares/{_encode_share_url(FILE_URL)}/driveItem/content"
        assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer my-token"


class TestListItems:
    def test_renders_list_item_fields_as_json(self):
        fetcher = MSSharePointFetcher()
        with patch.object(httpx.Client, "get", return_value=_json_response(LIST_ITEM)) as mock_get:
            streams = fetcher.run(access_token="tok", targets=[_list_item_document()])["streams"]

        assert len(streams) == 1
        stream = streams[0]
        assert json.loads(stream.data.decode("utf-8")) == LIST_ITEM["fields"]
        assert stream.mime_type == "application/json"
        assert stream.meta["content_type"] == "application/json"
        assert stream.meta["entity_type"] == "listItem"
        assert stream.meta["file_name"] == "Buy milk"  # falls back to the Title field
        # The list item is read by ID through the lists/items endpoint (not the shares endpoint).
        url = mock_get.call_args.args[0]
        assert url.startswith("https://graph.microsoft.com/v1.0/sites/site-id/lists/list-id/items/12?")
        assert "$expand=fields" in url

    def test_falls_back_to_shares_when_ids_missing(self):
        fetcher = MSSharePointFetcher()
        # A raw list-item URL with no IDs (e.g. `urls` input) falls back to the shares endpoint.
        with patch.object(httpx.Client, "get", return_value=_json_response(LIST_ITEM)) as mock_get:
            fetcher.run(access_token="tok", targets=[_list_item_document(ids=None)])
        assert "/shares/" in mock_get.call_args.args[0]
        assert "/listItem?" in mock_get.call_args.args[0]


class TestPages:
    def test_renders_page_web_parts_as_html(self):
        fetcher = MSSharePointFetcher()
        get = _route([("/items/", _json_response(PAGE_LIST_ITEM)), ("/pages/", _json_response(PAGE))])
        with patch.object(httpx.Client, "get", side_effect=get):
            streams = fetcher.run(access_token="tok", targets=[_page_document()])["streams"]

        assert len(streams) == 1
        stream = streams[0]
        body = stream.data.decode("utf-8")
        assert "<h1>Welcome</h1>" in body
        assert "<p>Hello world</p>" in body
        assert "<p>Second</p>" in body
        assert "<aside>Side</aside>" in body
        assert stream.mime_type == "text/html"
        assert stream.meta["entity_type"] == "sitePage"
        assert stream.meta["file_name"] == "Welcome.aspx"

    def test_calls_pages_api_with_site_and_page_id(self):
        fetcher = MSSharePointFetcher()
        get = _route([("/items/", _json_response(PAGE_LIST_ITEM)), ("/pages/", _json_response(PAGE))])
        with patch.object(httpx.Client, "get", side_effect=get) as mock_get:
            fetcher.run(access_token="tok", targets=[_page_document()])

        page_url = mock_get.call_args_list[-1].args[0]
        assert page_url == (
            "https://graph.microsoft.com/v1.0/sites/site-id/pages/page-guid/"
            "microsoft.graph.sitePage?$expand=canvasLayout"
        )

    def test_falls_back_to_fields_when_page_id_missing(self):
        fetcher = MSSharePointFetcher()
        # A page-like list item whose page id cannot be resolved: render its fields instead.
        page_without_unique_id = {**PAGE_LIST_ITEM, "parentReference": {}, "sharepointIds": {}}
        doc = _page_document()
        del doc.meta["list_item_unique_id"]  # no page id from the retriever either
        with patch.object(httpx.Client, "get", return_value=_json_response(page_without_unique_id)) as mock_get:
            streams = fetcher.run(access_token="tok", targets=[doc])["streams"]

        assert streams[0].meta["entity_type"] == "listItem"
        assert json.loads(streams[0].data.decode("utf-8")) == page_without_unique_id["fields"]
        # Only the list-item lookup happened, no Pages API call.
        assert mock_get.call_count == 1

    @pytest.mark.parametrize(
        "list_item, expected",
        [
            ({"contentType": {"name": "Site Page"}}, True),
            ({"contentType": {"name": "News Page"}}, True),
            ({"webUrl": "https://x/SitePages/Home.aspx"}, True),
            ({"fields": {"CanvasContent1": "<div/>"}}, True),
            ({"contentType": {"name": "Item"}, "webUrl": "https://x/Lists/Tasks/1"}, False),
        ],
    )
    def test_page_detection(self, list_item, expected):
        assert MSSharePointFetcher._is_page(list_item) is expected


class TestInputHandling:
    def test_probes_url_as_file_then_falls_back_to_list_item(self):
        fetcher = MSSharePointFetcher()
        # No entity-type hint: the file probe 404s, so the fetcher resolves it as a list item.
        get = _route(
            [("/driveItem/content", _binary_response(status_code=404)), ("/listItem", _json_response(LIST_ITEM))]
        )
        with patch.object(httpx.Client, "get", side_effect=get) as mock_get:
            streams = fetcher.run(access_token="tok", targets=[LIST_ITEM_URL])["streams"]

        assert streams[0].meta["entity_type"] == "listItem"
        assert mock_get.call_count == 2

    def test_dispatches_mixed_documents_and_urls(self):
        fetcher = MSSharePointFetcher()
        get = _route([("/driveItem/content", _binary_response()), ("/items/", _json_response(LIST_ITEM))])
        with patch.object(httpx.Client, "get", side_effect=get):
            # A drive-item document, a list-item document, and a raw file URL in a single list.
            streams = fetcher.run(
                access_token="tok",
                targets=[_drive_item_document(), _list_item_document(), FILE_URL],
            )["streams"]
        assert [s.meta["entity_type"] for s in streams] == ["driveItem", "listItem", "driveItem"]

    def test_targets_is_mandatory(self):
        # The data input has no default, so a pipeline will not fire the component until it is delivered;
        # calling run() without it is a TypeError.
        with pytest.raises(TypeError):
            MSSharePointFetcher().run(access_token="tok")

    def test_invalid_item_type_raises(self):
        with pytest.raises(SharePointConfigError):
            MSSharePointFetcher().run(access_token="tok", targets=[123])

    def test_empty_input_returns_no_streams(self):
        with patch.object(httpx.Client, "get") as mock_get:
            streams = MSSharePointFetcher().run(access_token="tok", targets=[])["streams"]
        assert streams == []
        mock_get.assert_not_called()

    def test_skips_container_entity(self):
        site = Document(
            content="snippet", meta={"web_url": "https://x/sites/team", "entity_type": "#microsoft.graph.site"}
        )
        with patch.object(httpx.Client, "get") as mock_get:
            streams = MSSharePointFetcher().run(access_token="tok", targets=[site])["streams"]
        assert streams == []
        mock_get.assert_not_called()

    def test_skips_document_without_web_url(self):
        doc = Document(content="snippet", meta={"file_name": "x.docx"})
        with patch.object(httpx.Client, "get") as mock_get:
            streams = MSSharePointFetcher().run(access_token="tok", targets=[doc])["streams"]
        assert streams == []
        mock_get.assert_not_called()

    def test_accepts_secret_access_token(self):
        fetcher = MSSharePointFetcher()
        with patch.object(httpx.Client, "get", return_value=_binary_response()) as mock_get:
            fetcher.run(access_token=Secret.from_token("secret-token"), targets=[FILE_URL])
        assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer secret-token"

    def test_unresolvable_secret_access_token_raises(self):
        unset_secret = Secret.from_env_var("MS_SHAREPOINT_UNSET_TOKEN", strict=False)
        with pytest.raises(SharePointConfigError):
            MSSharePointFetcher().run(access_token=unset_secret, targets=[FILE_URL])


class TestErrorHandling:
    def test_unauthorized_raises_with_status(self):
        fetcher = MSSharePointFetcher(max_retries=0)
        with patch.object(httpx.Client, "get", return_value=_binary_response(status_code=401)):
            with pytest.raises(SharePointRequestError) as exc_info:
                fetcher.run(access_token="bad", targets=[FILE_URL])
        assert exc_info.value.status_code == 401

    def test_forbidden_on_list_item_raises(self):
        fetcher = MSSharePointFetcher(max_retries=0)
        with patch.object(httpx.Client, "get", return_value=_json_response({"error": "no"}, status_code=403)):
            with pytest.raises(SharePointRequestError) as exc_info:
                fetcher.run(access_token="tok", targets=[_list_item_document()])
        assert exc_info.value.status_code == 403

    def test_raise_on_failure_false_skips_failed_items(self):
        fetcher = MSSharePointFetcher(max_retries=0, raise_on_failure=False)
        responses = [_binary_response(status_code=404), _binary_response(content=b"ok")]
        docs = [_drive_item_document(url="https://host/missing.docx"), _drive_item_document(url="https://host/ok.docx")]
        with patch.object(httpx.Client, "get", side_effect=responses):
            streams = fetcher.run(access_token="tok", targets=docs)["streams"]
        assert len(streams) == 1
        assert streams[0].data == b"ok"

    def test_retries_on_throttling_then_succeeds(self):
        fetcher = MSSharePointFetcher(max_retries=2)
        throttled = _binary_response(status_code=429, headers={"Retry-After": "0"})
        with patch.object(httpx.Client, "get", side_effect=[throttled, _binary_response(content=b"ok")]) as mock_get:
            streams = fetcher.run(access_token="tok", targets=[FILE_URL])["streams"]
        assert len(streams) == 1
        assert mock_get.call_count == 2

    def test_gives_up_after_max_retries(self):
        fetcher = MSSharePointFetcher(max_retries=1)
        throttled = _binary_response(status_code=429, headers={"Retry-After": "0"})
        with patch.object(httpx.Client, "get", return_value=throttled) as mock_get:
            with pytest.raises(SharePointRequestError) as exc_info:
                fetcher.run(access_token="tok", targets=[FILE_URL])
        assert exc_info.value.status_code == 429
        assert mock_get.call_count == 2


class TestPipeline:
    def test_runs_after_retriever_in_pipeline(self):
        files_response = {
            "value": [
                {
                    "hitsContainers": [
                        {
                            "moreResultsAvailable": False,
                            "hits": [
                                {
                                    "summary": "snippet",
                                    "resource": {
                                        "@odata.type": "#microsoft.graph.driveItem",
                                        "name": "design.docx",
                                        "webUrl": FILE_URL,
                                        "file": {"mimeType": DOCX_MIME},
                                    },
                                }
                            ],
                        }
                    ]
                }
            ]
        }
        search_response = httpx.Response(
            200, json=files_response, request=httpx.Request("POST", "https://graph.microsoft.com/v1.0/search/query")
        )

        pipeline = Pipeline()
        pipeline.add_component("retriever", MSSharePointRetriever())
        pipeline.add_component("fetcher", MSSharePointFetcher())
        pipeline.connect("retriever.documents", "fetcher.targets")

        with (
            patch.object(httpx.Client, "post", return_value=search_response),
            patch.object(httpx.Client, "get", return_value=_binary_response(content=b"docx-bytes")),
        ):
            result = pipeline.run(
                {"retriever": {"query": "design", "access_token": "tok"}, "fetcher": {"access_token": "tok"}}
            )

        streams = result["fetcher"]["streams"]
        assert len(streams) == 1
        assert streams[0].data == b"docx-bytes"
        assert streams[0].meta["url"] == FILE_URL

    def test_serialization_round_trip_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("fetcher", MSSharePointFetcher(raise_on_failure=False))
        restored = Pipeline.from_dict(pipeline.to_dict())
        assert restored.get_component("fetcher").raise_on_failure is False


@pytest.mark.asyncio
class TestRunAsync:
    async def test_downloads_drive_item(self):
        fetcher = MSSharePointFetcher()
        get = AsyncMock(return_value=_binary_response(content=b"async-bytes"))
        with patch.object(httpx.AsyncClient, "get", get):
            streams = (await fetcher.run_async(access_token="tok", targets=[_drive_item_document()]))["streams"]
        assert streams[0].data == b"async-bytes"
        assert streams[0].meta["entity_type"] == "driveItem"

    async def test_renders_list_item_fields(self):
        fetcher = MSSharePointFetcher()
        get = AsyncMock(return_value=_json_response(LIST_ITEM))
        with patch.object(httpx.AsyncClient, "get", get):
            streams = (await fetcher.run_async(access_token="tok", targets=[_list_item_document()]))["streams"]
        assert json.loads(streams[0].data.decode("utf-8")) == LIST_ITEM["fields"]
        assert streams[0].meta["entity_type"] == "listItem"

    async def test_renders_page(self):
        fetcher = MSSharePointFetcher()
        get = AsyncMock(
            side_effect=_route([("/items/", _json_response(PAGE_LIST_ITEM)), ("/pages/", _json_response(PAGE))])
        )
        with patch.object(httpx.AsyncClient, "get", get):
            streams = (await fetcher.run_async(access_token="tok", targets=[_page_document()]))["streams"]
        assert "<p>Hello world</p>" in streams[0].data.decode("utf-8")
        assert streams[0].meta["entity_type"] == "sitePage"

    async def test_retries_on_throttling_then_succeeds(self):
        fetcher = MSSharePointFetcher(max_retries=2)
        get = AsyncMock(
            side_effect=[_binary_response(status_code=429, headers={"Retry-After": "0"}), _binary_response()]
        )
        with patch.object(httpx.AsyncClient, "get", get):
            streams = (await fetcher.run_async(access_token="tok", targets=[FILE_URL]))["streams"]
        assert len(streams) == 1
        assert get.await_count == 2

    async def test_unauthorized_raises(self):
        fetcher = MSSharePointFetcher(max_retries=0)
        get = AsyncMock(return_value=_binary_response(status_code=401))
        with patch.object(httpx.AsyncClient, "get", get):
            with pytest.raises(SharePointRequestError):
                await fetcher.run_async(access_token="bad", targets=[FILE_URL])


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.environ.get("MS_SHAREPOINT_ACCESS_TOKEN") and os.environ.get("MS_SHAREPOINT_TEST_FILE_URL")),
    reason="MS_SHAREPOINT_ACCESS_TOKEN and MS_SHAREPOINT_TEST_FILE_URL not set",
)
class TestLive:
    def test_fetch_against_microsoft_graph(self):
        fetcher = MSSharePointFetcher()
        streams = fetcher.run(
            access_token=os.environ["MS_SHAREPOINT_ACCESS_TOKEN"],
            targets=[os.environ["MS_SHAREPOINT_TEST_FILE_URL"]],
        )["streams"]
        assert len(streams) == 1
        assert streams[0].data
        assert streams[0].meta["url"] == os.environ["MS_SHAREPOINT_TEST_FILE_URL"]
