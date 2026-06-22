# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from haystack import Document, Pipeline
from haystack.utils import Secret

from haystack_integrations.common.microsoft_sharepoint.utils import (
    _is_retryable_response,
    _wait_with_retry_after,
)
from haystack_integrations.components.retrievers.microsoft_sharepoint import (
    MSSharePointRetriever,
    SharePointConfigError,
    SharePointRequestError,
)

MODULE = "haystack_integrations.components.retrievers.microsoft_sharepoint.retriever"
SEARCH_URL = "https://graph.microsoft.com/v1.0/search/query"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# A driveItem hit, taken from the Microsoft Search API documentation:
# https://learn.microsoft.com/en-us/graph/search-concept-files
FILES_RESPONSE = {
    "value": [
        {
            "searchTerms": ["contoso"],
            "hitsContainers": [
                {
                    "total": 1,
                    "moreResultsAvailable": False,
                    "hits": [
                        {
                            "hitId": "FlULeN/ui/1GjLx1rUfio5UAAEl",
                            "rank": 1,
                            "summary": "<c0>Contoso</c0> Detailed Design <ddd/>",
                            "resource": {
                                "@odata.type": "#microsoft.graph.driveItem",
                                "createdDateTime": "2019-06-10T06:37:43Z",
                                "lastModifiedDateTime": "2019-06-10T06:37:43Z",
                                "name": "Contoso Detailed Design.docx",
                                "webUrl": "https://contoso.sharepoint.com/sites/contoso-team/contoso-designs.docx",
                                "file": {"mimeType": DOCX_MIME},
                                "createdBy": {"user": {"displayName": "Michaelvincent Santos;Provisioning User"}},
                                "lastModifiedBy": {"user": {"displayName": "Richard Mayer"}},
                                "parentReference": {
                                    "siteId": "m365x231305.sharepoint.com,5724d91f,c3ba25dc",
                                    "driveId": "da61a2b0-4120-4a3f-812b-0fc0d79bf16b",
                                    "sharepointIds": {"listId": "c61d1892-ca82-4f53-b16f-6bb8a379e2b2"},
                                },
                            },
                        }
                    ],
                }
            ],
        }
    ]
}

EMPTY_RESPONSE = {
    "value": [{"searchTerms": ["nothing"], "hitsContainers": [{"total": 0, "moreResultsAvailable": False}]}]
}


def _make_response(
    status_code: int = 200, json_body: dict | None = None, headers: dict | None = None
) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=json_body if json_body is not None else {},
        headers=headers,
        request=httpx.Request("POST", SEARCH_URL),
    )


def _hit(hit_id: str, rank: int, name: str) -> dict:
    return {
        "hitId": hit_id,
        "rank": rank,
        "summary": "",
        "resource": {"@odata.type": "#microsoft.graph.driveItem", "name": name, "webUrl": f"https://x/{hit_id}"},
    }


def _page(hits: list, more: bool) -> dict:
    return {
        "value": [{"searchTerms": ["q"], "hitsContainers": [{"total": 99, "moreResultsAvailable": more, "hits": hits}]}]
    }


class TestInit:
    def test_defaults(self):
        retriever = MSSharePointRetriever()
        assert retriever.entity_types == ["driveItem", "listItem"]
        assert retriever.top_k == 10
        assert retriever.fields is None
        assert retriever.query_template is None
        assert retriever.graph_url == "https://graph.microsoft.com/v1.0"
        assert retriever.timeout == 30.0
        assert retriever.max_retries == 3

    def test_graph_url_trailing_slash_is_stripped(self):
        retriever = MSSharePointRetriever(graph_url="https://graph.microsoft.com/v1.0/")
        assert retriever.graph_url == "https://graph.microsoft.com/v1.0"

    def test_empty_entity_types_raises(self):
        with pytest.raises(SharePointConfigError):
            MSSharePointRetriever(entity_types=[])

    def test_non_positive_top_k_raises(self):
        with pytest.raises(SharePointConfigError):
            MSSharePointRetriever(top_k=0)

    def test_negative_max_retries_raises(self):
        with pytest.raises(SharePointConfigError):
            MSSharePointRetriever(max_retries=-1)


class TestSerialization:
    def test_to_dict(self):
        retriever = MSSharePointRetriever(
            entity_types=["driveItem", "listItem"],
            top_k=5,
            fields=["title"],
            query_template='{searchTerms} path:"https://x"',
            graph_url="https://graph.microsoft.us/v1.0",
            timeout=10.0,
            max_retries=1,
        )
        data = retriever.to_dict()
        assert data == {
            "type": f"{MODULE}.MSSharePointRetriever",
            "init_parameters": {
                "entity_types": ["driveItem", "listItem"],
                "top_k": 5,
                "fields": ["title"],
                "query_template": '{searchTerms} path:"https://x"',
                "graph_url": "https://graph.microsoft.us/v1.0",
                "timeout": 10.0,
                "max_retries": 1,
            },
        }

    def test_from_dict_round_trip(self):
        retriever = MSSharePointRetriever(entity_types=["site"], top_k=7, query_template="{searchTerms}")
        restored = MSSharePointRetriever.from_dict(retriever.to_dict())
        assert restored.to_dict() == retriever.to_dict()


class TestRun:
    def test_maps_hit_to_document(self):
        retriever = MSSharePointRetriever()
        with patch.object(httpx.Client, "post", return_value=_make_response(json_body=FILES_RESPONSE)):
            documents = retriever.run(query="contoso", access_token="tok")["documents"]

        assert len(documents) == 1
        doc = documents[0]
        assert isinstance(doc, Document)
        # Highlight markup is stripped, <ddd/> becomes an ellipsis.
        assert doc.content == "Contoso Detailed Design …"
        assert doc.meta["file_name"] == "Contoso Detailed Design.docx"
        assert doc.meta["web_url"] == "https://contoso.sharepoint.com/sites/contoso-team/contoso-designs.docx"
        assert doc.meta["entity_type"] == "#microsoft.graph.driveItem"
        assert doc.meta["created_date_time"] == "2019-06-10T06:37:43Z"
        assert doc.meta["last_modified_date_time"] == "2019-06-10T06:37:43Z"
        assert doc.meta["created_by"] == "Michaelvincent Santos;Provisioning User"
        assert doc.meta["last_modified_by"] == "Richard Mayer"
        assert doc.meta["mime_type"] == DOCX_MIME
        # file_extension is derived from the name.
        assert doc.meta["file_extension"] == "docx"
        # Document.score is unset because Microsoft Search returns a rank, not a relevance score.
        assert doc.score is None

    def test_maps_sharepoint_ids(self):
        # A listItem hit carries the IDs a downstream fetcher needs to read it by ID.
        hit = {
            "summary": "",
            "resource": {
                "@odata.type": "#microsoft.graph.listItem",
                "name": "Task 1",
                "webUrl": "https://contoso.sharepoint.com/sites/team/Lists/Tasks/DispForm.aspx?ID=1",
                "parentReference": {"siteId": "site-1"},
                "sharepointIds": {"listId": "list-1", "listItemId": "1", "listItemUniqueId": "guid-1"},
            },
        }
        retriever = MSSharePointRetriever()
        with patch.object(httpx.Client, "post", return_value=_make_response(json_body=_page([hit], more=False))):
            doc = retriever.run(query="q", access_token="tok")["documents"][0]
        assert doc.meta["site_id"] == "site-1"
        assert doc.meta["list_id"] == "list-1"
        assert doc.meta["list_item_id"] == "1"
        assert doc.meta["list_item_unique_id"] == "guid-1"

    def test_builds_request_body_and_auth_header(self):
        retriever = MSSharePointRetriever(
            entity_types=["driveItem", "listItem"], top_k=5, fields=["title"], query_template="{searchTerms} foo"
        )
        with patch.object(httpx.Client, "post", return_value=_make_response(json_body=EMPTY_RESPONSE)) as mock_post:
            retriever.run(query="contoso", access_token="my-token")

        url = mock_post.call_args.args[0]
        body = mock_post.call_args.kwargs["json"]
        headers = mock_post.call_args.kwargs["headers"]
        assert url == SEARCH_URL
        assert headers["Authorization"] == "Bearer my-token"
        request = body["requests"][0]
        assert request["entityTypes"] == ["driveItem", "listItem"]
        assert request["query"] == {"queryString": "contoso", "queryTemplate": "{searchTerms} foo"}
        assert request["from"] == 0
        assert request["size"] == 5
        assert request["fields"] == ["title"]

    def test_top_k_override_sets_size(self):
        retriever = MSSharePointRetriever(top_k=10)
        with patch.object(httpx.Client, "post", return_value=_make_response(json_body=EMPTY_RESPONSE)) as mock_post:
            retriever.run(query="contoso", access_token="tok", top_k=3)
        assert mock_post.call_args.kwargs["json"]["requests"][0]["size"] == 3

    def test_accepts_secret_access_token(self):
        retriever = MSSharePointRetriever()
        with patch.object(httpx.Client, "post", return_value=_make_response(json_body=EMPTY_RESPONSE)) as mock_post:
            retriever.run(query="contoso", access_token=Secret.from_token("secret-token"))
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer secret-token"

    def test_unresolvable_secret_access_token_raises(self):
        retriever = MSSharePointRetriever()
        # A non-strict env var Secret resolves to None when the variable is unset.
        unset_secret = Secret.from_env_var("MS_SHAREPOINT_UNSET_TOKEN", strict=False)
        with pytest.raises(SharePointConfigError):
            retriever.run(query="contoso", access_token=unset_secret)

    def test_no_query_template_omits_key(self):
        retriever = MSSharePointRetriever()
        with patch.object(httpx.Client, "post", return_value=_make_response(json_body=EMPTY_RESPONSE)) as mock_post:
            retriever.run(query="contoso", access_token="tok")
        assert "queryTemplate" not in mock_post.call_args.kwargs["json"]["requests"][0]["query"]
        assert "fields" not in mock_post.call_args.kwargs["json"]["requests"][0]

    def test_empty_results(self):
        retriever = MSSharePointRetriever()
        with patch.object(httpx.Client, "post", return_value=_make_response(json_body=EMPTY_RESPONSE)):
            documents = retriever.run(query="nothing", access_token="tok")["documents"]
        assert documents == []

    def test_omits_missing_meta_fields(self):
        # _hit() has no file facet, no timestamps, and a name without an extension.
        retriever = MSSharePointRetriever()
        with patch.object(
            httpx.Client, "post", return_value=_make_response(json_body=_page([_hit("h1", 1, "notes")], more=False))
        ):
            doc = retriever.run(query="q", access_token="tok")["documents"][0]
        assert doc.meta["file_name"] == "notes"
        assert "mime_type" not in doc.meta
        assert "file_extension" not in doc.meta
        assert "created_date_time" not in doc.meta

    def test_pagination(self):
        retriever = MSSharePointRetriever(top_k=3)
        page1 = _make_response(json_body=_page([_hit("a", 1, "A"), _hit("b", 2, "B")], more=True))
        page2 = _make_response(json_body=_page([_hit("c", 3, "C")], more=False))
        with patch.object(httpx.Client, "post", side_effect=[page1, page2]) as mock_post:
            documents = retriever.run(query="q", access_token="tok")["documents"]

        assert [d.meta["file_name"] for d in documents] == ["A", "B", "C"]
        assert mock_post.call_count == 2
        # The second page is requested from the new offset with the remaining size.
        second_request = mock_post.call_args_list[1].kwargs["json"]["requests"][0]
        assert second_request["from"] == 2
        assert second_request["size"] == 1

    def test_respects_top_k_when_more_available(self):
        retriever = MSSharePointRetriever(top_k=2)
        page = _make_response(json_body=_page([_hit("a", 1, "A"), _hit("b", 2, "B")], more=True))
        with patch.object(httpx.Client, "post", return_value=page) as mock_post:
            documents = retriever.run(query="q", access_token="tok")["documents"]
        assert len(documents) == 2
        assert mock_post.call_count == 1


class TestErrorHandling:
    def test_unauthorized_raises_with_status(self):
        retriever = MSSharePointRetriever(max_retries=0)
        with patch.object(httpx.Client, "post", return_value=_make_response(status_code=401)):
            with pytest.raises(SharePointRequestError) as exc_info:
                retriever.run(query="q", access_token="bad")
        assert exc_info.value.status_code == 401

    def test_client_error_raises(self):
        retriever = MSSharePointRetriever(max_retries=0)
        with patch.object(
            httpx.Client, "post", return_value=_make_response(status_code=400, json_body={"error": "bad"})
        ):
            with pytest.raises(SharePointRequestError) as exc_info:
                retriever.run(query="q", access_token="tok")
        assert exc_info.value.status_code == 400

    def test_retries_on_throttling_then_succeeds(self):
        retriever = MSSharePointRetriever(max_retries=2)
        throttled = _make_response(status_code=429, headers={"Retry-After": "0"})
        ok = _make_response(json_body=FILES_RESPONSE)
        with patch.object(httpx.Client, "post", side_effect=[throttled, ok]) as mock_post:
            documents = retriever.run(query="contoso", access_token="tok")["documents"]
        assert len(documents) == 1
        assert mock_post.call_count == 2

    def test_gives_up_after_max_retries(self):
        retriever = MSSharePointRetriever(max_retries=1)
        throttled = _make_response(status_code=429, headers={"Retry-After": "0"})
        with patch.object(httpx.Client, "post", return_value=throttled) as mock_post:
            with pytest.raises(SharePointRequestError) as exc_info:
                retriever.run(query="q", access_token="tok")
        assert exc_info.value.status_code == 429
        # Initial attempt + 1 retry.
        assert mock_post.call_count == 2


class TestRetryStrategy:
    def _state(self, response: httpx.Response, attempt: int = 1) -> SimpleNamespace:
        outcome = SimpleNamespace(failed=False, result=lambda: response)
        return SimpleNamespace(outcome=outcome, attempt_number=attempt)

    @pytest.mark.parametrize("status_code, expected", [(429, True), (503, True), (200, False), (401, False)])
    def test_is_retryable_response(self, status_code, expected):
        assert _is_retryable_response(_make_response(status_code=status_code)) is expected

    def test_wait_honors_numeric_retry_after(self):
        response = _make_response(status_code=429, headers={"Retry-After": "7"})
        assert _wait_with_retry_after(self._state(response)) == 7.0

    def test_wait_falls_back_to_exponential_without_header(self):
        # No Retry-After header -> exponential backoff: 2 ** (attempt - 1).
        response = _make_response(status_code=429)
        assert _wait_with_retry_after(self._state(response, attempt=1)) == 1.0
        assert _wait_with_retry_after(self._state(response, attempt=3)) == 4.0

    def test_wait_falls_back_when_retry_after_is_not_numeric(self):
        # An HTTP-date Retry-After is not parsed as a float and falls back to exponential backoff.
        response = _make_response(status_code=429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})
        assert _wait_with_retry_after(self._state(response, attempt=2)) == 2.0


class TestPipeline:
    def test_runs_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("retriever", MSSharePointRetriever())
        with patch.object(httpx.Client, "post", return_value=_make_response(json_body=FILES_RESPONSE)):
            result = pipeline.run({"retriever": {"query": "contoso", "access_token": "tok"}})
        assert len(result["retriever"]["documents"]) == 1

    def test_serialization_round_trip_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("retriever", MSSharePointRetriever(top_k=4))
        restored = Pipeline.from_dict(pipeline.to_dict())
        assert restored.get_component("retriever").top_k == 4


@pytest.mark.asyncio
class TestRunAsync:
    async def test_maps_hit_to_document(self):
        retriever = MSSharePointRetriever()
        post = AsyncMock(return_value=_make_response(json_body=FILES_RESPONSE))
        with patch.object(httpx.AsyncClient, "post", post):
            documents = (await retriever.run_async(query="contoso", access_token="tok"))["documents"]
        assert len(documents) == 1
        assert documents[0].content == "Contoso Detailed Design …"

    async def test_retries_on_throttling_then_succeeds(self):
        retriever = MSSharePointRetriever(max_retries=2)
        post = AsyncMock(
            side_effect=[
                _make_response(status_code=429, headers={"Retry-After": "0"}),
                _make_response(json_body=FILES_RESPONSE),
            ]
        )
        with patch.object(httpx.AsyncClient, "post", post):
            documents = (await retriever.run_async(query="contoso", access_token="tok"))["documents"]
        assert len(documents) == 1
        assert post.await_count == 2

    async def test_unauthorized_raises(self):
        retriever = MSSharePointRetriever(max_retries=0)
        post = AsyncMock(return_value=_make_response(status_code=401))
        with patch.object(httpx.AsyncClient, "post", post):
            with pytest.raises(SharePointRequestError):
                await retriever.run_async(query="q", access_token="bad")

    async def test_accepts_secret_access_token(self):
        retriever = MSSharePointRetriever()
        post = AsyncMock(return_value=_make_response(json_body=EMPTY_RESPONSE))
        with patch.object(httpx.AsyncClient, "post", post):
            await retriever.run_async(query="contoso", access_token=Secret.from_token("secret-token"))
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer secret-token"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("MS_SHAREPOINT_ACCESS_TOKEN"),
    reason="MS_SHAREPOINT_ACCESS_TOKEN not set",
)
class TestLive:
    def test_run_against_microsoft_graph(self):
        retriever = MSSharePointRetriever(top_k=3)
        query = os.environ.get("MS_SHAREPOINT_TEST_QUERY", "test")
        documents = retriever.run(query=query, access_token=os.environ["MS_SHAREPOINT_ACCESS_TOKEN"])["documents"]
        assert isinstance(documents, list)
        for doc in documents:
            assert isinstance(doc, Document)
            assert "web_url" in doc.meta
