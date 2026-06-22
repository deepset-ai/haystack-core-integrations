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

from haystack_integrations.components.retrievers.google_drive import (
    GoogleDriveConfigError,
    GoogleDriveRequestError,
    GoogleDriveRetriever,
)
from haystack_integrations.components.retrievers.google_drive.retriever import (
    _is_retryable_response,
    _wait_with_retry_after,
)

MODULE = "haystack_integrations.components.retrievers.google_drive.retriever"
FILES_URL = "https://www.googleapis.com/drive/v3/files"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"

# A `files.list` hit shaped like a real Drive API response (a binary .docx file).
FILES_RESPONSE = {
    "files": [
        {
            "id": "1AbCdEf",
            "name": "Contoso Detailed Design.docx",
            "mimeType": DOCX_MIME,
            "webViewLink": "https://drive.google.com/file/d/1AbCdEf/view",
            "description": "Contoso detailed design document.",
            "fileExtension": "docx",
            "createdTime": "2019-06-10T06:37:43.000Z",
            "modifiedTime": "2019-06-11T08:20:00.000Z",
            "owners": [{"displayName": "Michaelvincent Santos"}],
            "lastModifyingUser": {"displayName": "Richard Mayer"},
        }
    ]
}

# A native Google Doc (no description) used to exercise content export.
GOOGLE_DOC_RESPONSE = {
    "files": [
        {
            "id": "doc1",
            "name": "Quarterly Roadmap",
            "mimeType": GOOGLE_DOC_MIME,
            "webViewLink": "https://docs.google.com/document/d/doc1/edit",
        }
    ]
}

EMPTY_RESPONSE = {"files": []}


def _make_response(status_code=200, json_body=None, text_body=None, headers=None):
    if text_body is not None:
        return httpx.Response(status_code, text=text_body, headers=headers, request=httpx.Request("GET", FILES_URL))
    return httpx.Response(
        status_code,
        json=json_body if json_body is not None else {},
        headers=headers,
        request=httpx.Request("GET", FILES_URL),
    )


def _file(file_id, name):
    return {"id": file_id, "name": name, "mimeType": DOCX_MIME, "webViewLink": f"https://x/{file_id}"}


def _page(files, next_token=None):
    page = {"files": files}
    if next_token:
        page["nextPageToken"] = next_token
    return page


class TestInit:
    def test_defaults(self):
        retriever = GoogleDriveRetriever()
        assert retriever.include_content is False
        assert retriever.top_k == 10
        assert retriever.query_filter is None
        assert retriever.include_shared_drives is False
        assert retriever.order_by is None
        assert retriever.fields is None
        assert retriever.api_base_url == "https://www.googleapis.com/drive/v3"
        assert retriever.timeout == 30.0
        assert retriever.max_retries == 3

    def test_api_base_url_trailing_slash_is_stripped(self):
        retriever = GoogleDriveRetriever(api_base_url="https://www.googleapis.com/drive/v3/")
        assert retriever.api_base_url == "https://www.googleapis.com/drive/v3"

    def test_non_positive_top_k_raises(self):
        with pytest.raises(GoogleDriveConfigError):
            GoogleDriveRetriever(top_k=0)

    def test_negative_max_retries_raises(self):
        with pytest.raises(GoogleDriveConfigError):
            GoogleDriveRetriever(max_retries=-1)


class TestSerialization:
    def test_to_dict(self):
        retriever = GoogleDriveRetriever(
            include_content=True,
            top_k=5,
            query_filter="mimeType != 'application/vnd.google-apps.folder'",
            include_shared_drives=True,
            order_by="modifiedTime desc",
            fields=["id", "name"],
            api_base_url="https://www.googleapis.com/drive/v3",
            timeout=10.0,
            max_retries=1,
        )
        data = retriever.to_dict()
        assert data == {
            "type": f"{MODULE}.GoogleDriveRetriever",
            "init_parameters": {
                "include_content": True,
                "top_k": 5,
                "query_filter": "mimeType != 'application/vnd.google-apps.folder'",
                "include_shared_drives": True,
                "order_by": "modifiedTime desc",
                "fields": ["id", "name"],
                "api_base_url": "https://www.googleapis.com/drive/v3",
                "timeout": 10.0,
                "max_retries": 1,
            },
        }

    def test_from_dict_round_trip(self):
        retriever = GoogleDriveRetriever(top_k=7, query_filter="'folder' in parents")
        restored = GoogleDriveRetriever.from_dict(retriever.to_dict())
        assert restored.to_dict() == retriever.to_dict()


class TestRun:
    def test_maps_file_to_document(self):
        retriever = GoogleDriveRetriever()
        with patch.object(httpx.Client, "get", return_value=_make_response(json_body=FILES_RESPONSE)):
            documents = retriever.run(query="contoso", access_token="tok")["documents"]

        assert len(documents) == 1
        doc = documents[0]
        assert isinstance(doc, Document)
        # Default include_content=False → content is the file description (falls back to name when absent).
        assert doc.content == "Contoso detailed design document."
        assert doc.meta["file_name"] == "Contoso Detailed Design.docx"
        assert doc.meta["file_id"] == "1AbCdEf"
        assert doc.meta["web_url"] == "https://drive.google.com/file/d/1AbCdEf/view"
        assert doc.meta["mime_type"] == DOCX_MIME
        assert doc.meta["file_extension"] == "docx"
        assert doc.meta["created_date_time"] == "2019-06-10T06:37:43.000Z"
        assert doc.meta["last_modified_date_time"] == "2019-06-11T08:20:00.000Z"
        assert doc.meta["created_by"] == "Michaelvincent Santos"
        assert doc.meta["last_modified_by"] == "Richard Mayer"
        # Drive returns no relevance score.
        assert doc.score is None

    def test_builds_request_params_and_auth_header(self):
        retriever = GoogleDriveRetriever(top_k=5)
        with patch.object(httpx.Client, "get", return_value=_make_response(json_body=EMPTY_RESPONSE)) as mock_get:
            retriever.run(query="contoso", access_token="my-token")

        url = mock_get.call_args.args[0]
        params = mock_get.call_args.kwargs["params"]
        headers = mock_get.call_args.kwargs["headers"]
        assert url == FILES_URL
        assert headers["Authorization"] == "Bearer my-token"
        assert params["q"] == "fullText contains 'contoso'"
        assert params["pageSize"] == 5
        assert params["fields"].startswith("nextPageToken, files(")
        # Optional params are absent by default.
        assert "pageToken" not in params
        assert "orderBy" not in params
        assert "corpora" not in params

    def test_query_filter_is_anded_into_q(self):
        retriever = GoogleDriveRetriever(query_filter="mimeType != 'application/vnd.google-apps.folder'")
        with patch.object(httpx.Client, "get", return_value=_make_response(json_body=EMPTY_RESPONSE)) as mock_get:
            retriever.run(query="contoso", access_token="tok")

        q = mock_get.call_args.kwargs["params"]["q"]
        assert q == "fullText contains 'contoso' and (mimeType != 'application/vnd.google-apps.folder')"

    def test_shared_drive_and_order_params(self):
        retriever = GoogleDriveRetriever(include_shared_drives=True, order_by="modifiedTime desc")
        with patch.object(httpx.Client, "get", return_value=_make_response(json_body=EMPTY_RESPONSE)) as mock_get:
            retriever.run(query="contoso", access_token="tok")

        params = mock_get.call_args.kwargs["params"]
        assert params["includeItemsFromAllDrives"] is True
        assert params["supportsAllDrives"] is True
        assert params["corpora"] == "allDrives"
        assert params["orderBy"] == "modifiedTime desc"

    def test_escapes_quotes_in_query(self):
        retriever = GoogleDriveRetriever()
        with patch.object(httpx.Client, "get", return_value=_make_response(json_body=EMPTY_RESPONSE)) as mock_get:
            retriever.run(query="o'brien", access_token="tok")

        assert mock_get.call_args.kwargs["params"]["q"] == "fullText contains 'o\\'brien'"

    def test_top_k_override_sets_page_size(self):
        retriever = GoogleDriveRetriever(top_k=10)
        with patch.object(httpx.Client, "get", return_value=_make_response(json_body=EMPTY_RESPONSE)) as mock_get:
            retriever.run(query="contoso", access_token="tok", top_k=3)
        assert mock_get.call_args.kwargs["params"]["pageSize"] == 3

    def test_accepts_secret_access_token(self):
        retriever = GoogleDriveRetriever()
        with patch.object(httpx.Client, "get", return_value=_make_response(json_body=EMPTY_RESPONSE)) as mock_get:
            retriever.run(query="contoso", access_token=Secret.from_token("secret-token"))
        assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer secret-token"

    def test_unresolvable_secret_access_token_raises(self):
        retriever = GoogleDriveRetriever()
        # A non-strict env var Secret resolves to None when the variable is unset.
        unset_secret = Secret.from_env_var("GOOGLE_DRIVE_UNSET_TOKEN", strict=False)
        with pytest.raises(GoogleDriveConfigError):
            retriever.run(query="contoso", access_token=unset_secret)

    def test_empty_results(self):
        retriever = GoogleDriveRetriever()
        with patch.object(httpx.Client, "get", return_value=_make_response(json_body=EMPTY_RESPONSE)):
            documents = retriever.run(query="nothing", access_token="tok")["documents"]
        assert documents == []

    def test_omits_missing_meta_fields(self):
        # A file with only id + name: no mimeType, no timestamps, and a name without an extension.
        retriever = GoogleDriveRetriever()
        page = _page([{"id": "x1", "name": "notes"}])
        with patch.object(httpx.Client, "get", return_value=_make_response(json_body=page)):
            doc = retriever.run(query="q", access_token="tok")["documents"][0]
        assert doc.meta["file_name"] == "notes"
        assert doc.content == "notes"
        assert "mime_type" not in doc.meta
        assert "file_extension" not in doc.meta
        assert "created_date_time" not in doc.meta
        assert "created_by" not in doc.meta

    def test_pagination(self):
        retriever = GoogleDriveRetriever(top_k=3)
        page1 = _make_response(json_body=_page([_file("a", "A"), _file("b", "B")], next_token="tok2"))
        page2 = _make_response(json_body=_page([_file("c", "C")]))
        with patch.object(httpx.Client, "get", side_effect=[page1, page2]) as mock_get:
            documents = retriever.run(query="q", access_token="tok")["documents"]

        assert [d.meta["file_name"] for d in documents] == ["A", "B", "C"]
        assert mock_get.call_count == 2
        # The second page is requested with the returned token and the remaining size.
        second_params = mock_get.call_args_list[1].kwargs["params"]
        assert second_params["pageToken"] == "tok2"
        assert second_params["pageSize"] == 1

    def test_respects_top_k_when_more_available(self):
        retriever = GoogleDriveRetriever(top_k=2)
        page = _make_response(json_body=_page([_file("a", "A"), _file("b", "B")], next_token="more"))
        with patch.object(httpx.Client, "get", return_value=page) as mock_get:
            documents = retriever.run(query="q", access_token="tok")["documents"]
        assert len(documents) == 2
        assert mock_get.call_count == 1


class TestIncludeContent:
    def test_exports_native_google_doc(self):
        retriever = GoogleDriveRetriever(include_content=True)
        list_resp = _make_response(json_body=GOOGLE_DOC_RESPONSE)
        export_resp = _make_response(text_body="Exported roadmap body.")
        with patch.object(httpx.Client, "get", side_effect=[list_resp, export_resp]) as mock_get:
            documents = retriever.run(query="roadmap", access_token="tok")["documents"]

        assert documents[0].content == "Exported roadmap body."
        assert mock_get.call_count == 2
        export_call = mock_get.call_args_list[1]
        assert export_call.args[0] == "https://www.googleapis.com/drive/v3/files/doc1/export"
        assert export_call.kwargs["params"] == {"mimeType": "text/plain"}

    def test_binary_file_not_exported(self):
        retriever = GoogleDriveRetriever(include_content=True)
        with patch.object(httpx.Client, "get", return_value=_make_response(json_body=FILES_RESPONSE)) as mock_get:
            documents = retriever.run(query="contoso", access_token="tok")["documents"]
        # Binary files are never exported; content falls back to description/name and no export call is made.
        assert mock_get.call_count == 1
        assert documents[0].content == "Contoso detailed design document."

    def test_export_error_falls_back_to_metadata(self):
        retriever = GoogleDriveRetriever(include_content=True)
        list_resp = _make_response(json_body=GOOGLE_DOC_RESPONSE)
        export_err = _make_response(status_code=403, json_body={"error": "export too large"})
        with patch.object(httpx.Client, "get", side_effect=[list_resp, export_err]):
            documents = retriever.run(query="roadmap", access_token="tok")["documents"]
        # No raise; content falls back to the file name (this fixture has no description).
        assert documents[0].content == "Quarterly Roadmap"

    def test_export_network_error_falls_back_to_metadata(self):
        retriever = GoogleDriveRetriever(include_content=True)
        list_resp = _make_response(json_body=GOOGLE_DOC_RESPONSE)
        with patch.object(httpx.Client, "get", side_effect=[list_resp, httpx.ConnectError("boom")]):
            documents = retriever.run(query="roadmap", access_token="tok")["documents"]
        # A transport error during export is swallowed; content falls back to the file name.
        assert documents[0].content == "Quarterly Roadmap"

    def test_content_skipped_when_file_lacks_id_or_mime(self):
        # A file without a mimeType cannot be exported, so no export call is made.
        retriever = GoogleDriveRetriever(include_content=True)
        page = _page([{"id": "x", "name": "notes"}])
        with patch.object(httpx.Client, "get", return_value=_make_response(json_body=page)) as mock_get:
            documents = retriever.run(query="q", access_token="tok")["documents"]
        assert mock_get.call_count == 1
        assert documents[0].content == "notes"


class TestErrorHandling:
    def test_unauthorized_raises_with_status(self):
        retriever = GoogleDriveRetriever(max_retries=0)
        with patch.object(httpx.Client, "get", return_value=_make_response(status_code=401)):
            with pytest.raises(GoogleDriveRequestError) as exc_info:
                retriever.run(query="q", access_token="bad")
        assert exc_info.value.status_code == 401

    def test_client_error_raises(self):
        retriever = GoogleDriveRetriever(max_retries=0)
        response = _make_response(status_code=400, json_body={"error": "bad"})
        with patch.object(httpx.Client, "get", return_value=response):
            with pytest.raises(GoogleDriveRequestError) as exc_info:
                retriever.run(query="q", access_token="tok")
        assert exc_info.value.status_code == 400

    def test_retries_on_throttling_then_succeeds(self):
        retriever = GoogleDriveRetriever(max_retries=2)
        throttled = _make_response(status_code=429, headers={"Retry-After": "0"})
        ok = _make_response(json_body=FILES_RESPONSE)
        with patch.object(httpx.Client, "get", side_effect=[throttled, ok]) as mock_get:
            documents = retriever.run(query="contoso", access_token="tok")["documents"]
        assert len(documents) == 1
        assert mock_get.call_count == 2

    def test_gives_up_after_max_retries(self):
        retriever = GoogleDriveRetriever(max_retries=1)
        throttled = _make_response(status_code=429, headers={"Retry-After": "0"})
        with patch.object(httpx.Client, "get", return_value=throttled) as mock_get:
            with pytest.raises(GoogleDriveRequestError) as exc_info:
                retriever.run(query="q", access_token="tok")
        assert exc_info.value.status_code == 429
        # Initial attempt + 1 retry.
        assert mock_get.call_count == 2

    def test_forbidden_raises_with_status(self):
        retriever = GoogleDriveRetriever(max_retries=0)
        response = _make_response(status_code=403, json_body={"error": "forbidden"})
        with patch.object(httpx.Client, "get", return_value=response):
            with pytest.raises(GoogleDriveRequestError) as exc_info:
                retriever.run(query="q", access_token="tok")
        assert exc_info.value.status_code == 403


class TestRetryStrategy:
    def _state(self, response: httpx.Response, attempt: int = 1) -> SimpleNamespace:
        outcome = SimpleNamespace(failed=False, result=lambda: response)
        return SimpleNamespace(outcome=outcome, attempt_number=attempt)

    @pytest.mark.parametrize("status_code, expected", [(429, True), (503, True), (200, False), (401, False)])
    def test_is_retryable_response(self, status_code, expected):
        assert _is_retryable_response(_make_response(status_code=status_code)) is expected

    def test_wait_honors_numeric_retry_after(self):
        response = _make_response(status_code=429, headers={"Retry-After": "5"})
        assert _wait_with_retry_after(self._state(response)) == 5.0

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
        pipeline.add_component("retriever", GoogleDriveRetriever())
        with patch.object(httpx.Client, "get", return_value=_make_response(json_body=FILES_RESPONSE)):
            result = pipeline.run({"retriever": {"query": "contoso", "access_token": "tok"}})
        assert len(result["retriever"]["documents"]) == 1

    def test_serialization_round_trip_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("retriever", GoogleDriveRetriever(top_k=4))
        restored = Pipeline.from_dict(pipeline.to_dict())
        assert restored.get_component("retriever").top_k == 4


@pytest.mark.asyncio
class TestRunAsync:
    async def test_maps_file_to_document(self):
        retriever = GoogleDriveRetriever()
        get = AsyncMock(return_value=_make_response(json_body=FILES_RESPONSE))
        with patch.object(httpx.AsyncClient, "get", get):
            documents = (await retriever.run_async(query="contoso", access_token="tok"))["documents"]
        assert len(documents) == 1
        assert documents[0].meta["file_name"] == "Contoso Detailed Design.docx"

    async def test_respects_top_k_when_more_available(self):
        retriever = GoogleDriveRetriever(top_k=2)
        page = _make_response(json_body=_page([_file("a", "A"), _file("b", "B")], next_token="more"))
        get = AsyncMock(return_value=page)
        with patch.object(httpx.AsyncClient, "get", get):
            documents = (await retriever.run_async(query="q", access_token="tok"))["documents"]
        assert len(documents) == 2
        assert get.await_count == 1

    async def test_retries_on_throttling_then_succeeds(self):
        retriever = GoogleDriveRetriever(max_retries=2)
        get = AsyncMock(
            side_effect=[
                _make_response(status_code=429, headers={"Retry-After": "0"}),
                _make_response(json_body=FILES_RESPONSE),
            ]
        )
        with patch.object(httpx.AsyncClient, "get", get):
            documents = (await retriever.run_async(query="contoso", access_token="tok"))["documents"]
        assert len(documents) == 1
        assert get.await_count == 2

    async def test_unauthorized_raises(self):
        retriever = GoogleDriveRetriever(max_retries=0)
        get = AsyncMock(return_value=_make_response(status_code=401))
        with patch.object(httpx.AsyncClient, "get", get):
            with pytest.raises(GoogleDriveRequestError):
                await retriever.run_async(query="q", access_token="bad")

    async def test_accepts_secret_access_token(self):
        retriever = GoogleDriveRetriever()
        get = AsyncMock(return_value=_make_response(json_body=EMPTY_RESPONSE))
        with patch.object(httpx.AsyncClient, "get", get):
            await retriever.run_async(query="contoso", access_token=Secret.from_token("secret-token"))
        assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer secret-token"


@pytest.mark.asyncio
class TestIncludeContentAsync:
    async def test_exports_native_google_doc(self):
        retriever = GoogleDriveRetriever(include_content=True)
        get = AsyncMock(
            side_effect=[
                _make_response(json_body=GOOGLE_DOC_RESPONSE),
                _make_response(text_body="Exported roadmap body."),
            ]
        )
        with patch.object(httpx.AsyncClient, "get", get):
            documents = (await retriever.run_async(query="roadmap", access_token="tok"))["documents"]
        assert documents[0].content == "Exported roadmap body."
        assert get.await_count == 2
        export_call = get.await_args_list[1]
        assert export_call.args[0] == "https://www.googleapis.com/drive/v3/files/doc1/export"
        assert export_call.kwargs["params"] == {"mimeType": "text/plain"}

    async def test_binary_file_not_exported(self):
        retriever = GoogleDriveRetriever(include_content=True)
        get = AsyncMock(return_value=_make_response(json_body=FILES_RESPONSE))
        with patch.object(httpx.AsyncClient, "get", get):
            documents = (await retriever.run_async(query="contoso", access_token="tok"))["documents"]
        assert get.await_count == 1
        assert documents[0].content == "Contoso detailed design document."

    async def test_content_skipped_when_file_lacks_id_or_mime(self):
        retriever = GoogleDriveRetriever(include_content=True)
        get = AsyncMock(return_value=_make_response(json_body=_page([{"id": "x", "name": "notes"}])))
        with patch.object(httpx.AsyncClient, "get", get):
            documents = (await retriever.run_async(query="q", access_token="tok"))["documents"]
        assert get.await_count == 1
        assert documents[0].content == "notes"

    async def test_export_http_error_falls_back_to_metadata(self):
        retriever = GoogleDriveRetriever(include_content=True)
        get = AsyncMock(
            side_effect=[
                _make_response(json_body=GOOGLE_DOC_RESPONSE),
                _make_response(status_code=403, json_body={"error": "export too large"}),
            ]
        )
        with patch.object(httpx.AsyncClient, "get", get):
            documents = (await retriever.run_async(query="roadmap", access_token="tok"))["documents"]
        assert documents[0].content == "Quarterly Roadmap"

    async def test_export_network_error_falls_back_to_metadata(self):
        retriever = GoogleDriveRetriever(include_content=True)
        get = AsyncMock(
            side_effect=[
                _make_response(json_body=GOOGLE_DOC_RESPONSE),
                httpx.ConnectError("boom"),
            ]
        )
        with patch.object(httpx.AsyncClient, "get", get):
            documents = (await retriever.run_async(query="roadmap", access_token="tok"))["documents"]
        assert documents[0].content == "Quarterly Roadmap"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("GOOGLE_DRIVE_ACCESS_TOKEN"),
    reason="GOOGLE_DRIVE_ACCESS_TOKEN not set",
)
class TestLive:
    def test_run_against_google_drive(self):
        retriever = GoogleDriveRetriever(top_k=3)
        query = os.environ.get("GOOGLE_DRIVE_TEST_QUERY", "test")
        documents = retriever.run(query=query, access_token=os.environ["GOOGLE_DRIVE_ACCESS_TOKEN"])["documents"]
        assert isinstance(documents, list)
        for doc in documents:
            assert isinstance(doc, Document)
            assert "web_url" in doc.meta
