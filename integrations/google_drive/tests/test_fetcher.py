# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from haystack import Document, Pipeline
from haystack.dataclasses import ByteStream
from haystack.utils import Secret

from haystack_integrations.components.fetchers.google_drive import (
    GoogleDriveConfigError,
    GoogleDriveFetcher,
    GoogleDriveRequestError,
)
from haystack_integrations.components.fetchers.google_drive.fetcher import _resolve_file_id
from haystack_integrations.components.retrievers.google_drive import GoogleDriveRetriever

MODULE = "haystack_integrations.components.fetchers.google_drive.fetcher"
BASE = "https://www.googleapis.com/drive/v3"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
GDOC = "application/vnd.google-apps.document"
FOLDER = "application/vnd.google-apps.folder"
WEB_URL = "https://drive.google.com/file/d/FILE1/view"


def _binary(status_code: int = 200, content: bytes = b"bytes", content_type: str | None = "application/pdf"):
    headers = {"Content-Type": content_type} if content_type else {}
    return httpx.Response(status_code, content=content, headers=headers, request=httpx.Request("GET", f"{BASE}/files"))


def _json(body: dict | None = None, status_code: int = 200):
    return httpx.Response(status_code, json=body if body is not None else {}, request=httpx.Request("GET", f"{BASE}/x"))


def _router(metadata=None, media=None, export=None, files_list=None):
    """Build a `get` side effect that returns a response based on the request URL/params."""

    def handler(url, *_args, **kwargs):
        params = kwargs.get("params") or {}
        if url.endswith("/export"):
            return export
        if params.get("alt") == "media":
            return media
        if url.endswith("/files"):
            return files_list
        return metadata

    return handler


def _drive_doc(file_id: str = "FILE1", mime_type: str | None = "application/pdf", **extra) -> Document:
    meta = {"file_id": file_id, "mime_type": mime_type, "file_name": "report.pdf", "web_url": WEB_URL}
    meta.update(extra)
    return Document(content="snippet", meta=meta)


class TestInit:
    def test_defaults(self):
        fetcher = GoogleDriveFetcher()
        assert fetcher.api_base_url == BASE
        assert fetcher.timeout == 30.0
        assert fetcher.max_retries == 3
        assert fetcher.raise_on_failure is True
        assert fetcher.export_mime_types is None
        assert fetcher._export_map[GDOC] == DOCX

    def test_api_base_url_trailing_slash_is_stripped(self):
        assert GoogleDriveFetcher(api_base_url=f"{BASE}/").api_base_url == BASE

    def test_negative_max_retries_raises(self):
        with pytest.raises(GoogleDriveConfigError):
            GoogleDriveFetcher(max_retries=-1)

    def test_custom_export_mapping_replaces_default(self):
        fetcher = GoogleDriveFetcher(export_mime_types={GDOC: "application/pdf"})
        assert fetcher._export_map == {GDOC: "application/pdf"}


class TestSerialization:
    def test_to_dict(self):
        fetcher = GoogleDriveFetcher(timeout=10.0, max_retries=1, raise_on_failure=False)
        assert fetcher.to_dict() == {
            "type": f"{MODULE}.GoogleDriveFetcher",
            "init_parameters": {
                "api_base_url": BASE,
                "timeout": 10.0,
                "max_retries": 1,
                "raise_on_failure": False,
                "export_mime_types": None,
            },
        }

    def test_from_dict_round_trip(self):
        fetcher = GoogleDriveFetcher(export_mime_types={GDOC: "application/pdf"}, raise_on_failure=False)
        restored = GoogleDriveFetcher.from_dict(fetcher.to_dict())
        assert restored.to_dict() == fetcher.to_dict()
        assert restored._export_map == {GDOC: "application/pdf"}


class TestFileIdParsing:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("https://drive.google.com/file/d/1AbC_dEf-9/view?usp=sharing", "1AbC_dEf-9"),
            ("https://docs.google.com/document/d/XYZ987/edit", "XYZ987"),
            ("https://drive.google.com/open?id=OPEN42", "OPEN42"),
            ("https://drive.google.com/uc?export=download&id=UC_7", "UC_7"),
            ("BARE_ID_7", "BARE_ID_7"),
        ],
    )
    def test_resolve_file_id(self, value, expected):
        assert _resolve_file_id(value) == expected


class TestBinaryFiles:
    def test_downloads_binary_document(self):
        fetcher = GoogleDriveFetcher()
        with patch.object(httpx.Client, "get", return_value=_binary(content=b"pdf-bytes")):
            streams = fetcher.run(access_token="tok", targets=[_drive_doc()])["streams"]

        assert len(streams) == 1
        stream = streams[0]
        assert isinstance(stream, ByteStream)
        assert stream.data == b"pdf-bytes"
        assert stream.mime_type == "application/pdf"
        assert stream.meta == {
            "file_id": "FILE1",
            "web_url": WEB_URL,
            "file_name": "report.pdf",
            "content_type": "application/pdf",
        }

    def test_content_type_falls_back_to_known_mime(self):
        fetcher = GoogleDriveFetcher()
        # No Content-Type on the response -> fall back to the document's mime hint.
        with patch.object(httpx.Client, "get", return_value=_binary(content_type=None)):
            streams = fetcher.run(access_token="tok", targets=[_drive_doc(mime_type="image/png")])["streams"]
        assert streams[0].meta["content_type"] == "image/png"
        assert streams[0].mime_type == "image/png"

    def test_builds_media_url_and_auth_header(self):
        fetcher = GoogleDriveFetcher()
        with patch.object(httpx.Client, "get", return_value=_binary()) as mock_get:
            fetcher.run(access_token="my-token", targets=[_drive_doc()])

        assert mock_get.call_args.args[0] == f"{BASE}/files/FILE1"
        assert mock_get.call_args.kwargs["params"] == {"alt": "media", "supportsAllDrives": True}
        assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer my-token"

    def test_downloads_from_raw_file_id_with_metadata_lookup(self):
        fetcher = GoogleDriveFetcher()
        metadata = _json({"id": "FILE1", "name": "paper.pdf", "mimeType": "application/pdf", "webViewLink": WEB_URL})
        get = _router(metadata=metadata, media=_binary(content=b"pdf"))
        with patch.object(httpx.Client, "get", side_effect=get):
            streams = fetcher.run(access_token="tok", targets=["FILE1"])["streams"]

        assert streams[0].data == b"pdf"
        assert streams[0].meta["file_name"] == "paper.pdf"  # learned from metadata
        assert streams[0].meta["web_url"] == WEB_URL

    def test_downloads_from_drive_url(self):
        fetcher = GoogleDriveFetcher()
        metadata = _json({"id": "1AbC", "name": "doc.pdf", "mimeType": "application/pdf"})
        get = _router(metadata=metadata, media=_binary(content=b"x"))
        with patch.object(httpx.Client, "get", side_effect=get) as mock_get:
            fetcher.run(access_token="tok", targets=["https://drive.google.com/file/d/1AbC/view"])

        # The parsed file id is used for the metadata lookup.
        assert mock_get.call_args_list[0].args[0] == f"{BASE}/files/1AbC"


class TestNativeExport:
    def test_exports_google_doc_as_docx(self):
        fetcher = GoogleDriveFetcher()
        get = _router(export=_binary(content=b"docx-bytes", content_type=None))
        with patch.object(httpx.Client, "get", side_effect=get):
            streams = fetcher.run(access_token="tok", targets=[_drive_doc(mime_type=GDOC)])["streams"]

        assert streams[0].data == b"docx-bytes"
        assert streams[0].mime_type == DOCX
        assert streams[0].meta["content_type"] == DOCX

    def test_builds_export_url_and_params(self):
        fetcher = GoogleDriveFetcher()
        get = _router(export=_binary(content=b"x"))
        with patch.object(httpx.Client, "get", side_effect=get) as mock_get:
            fetcher.run(access_token="tok", targets=[_drive_doc(mime_type=GDOC)])

        call = mock_get.call_args
        assert call.args[0] == f"{BASE}/files/FILE1/export"
        assert call.kwargs["params"] == {"mimeType": DOCX, "supportsAllDrives": True}

    def test_export_uses_configured_mapping(self):
        fetcher = GoogleDriveFetcher(export_mime_types={GDOC: "application/pdf"})
        get = _router(export=_binary(content=b"x"))
        with patch.object(httpx.Client, "get", side_effect=get) as mock_get:
            streams = fetcher.run(access_token="tok", targets=[_drive_doc(mime_type=GDOC)])["streams"]

        assert mock_get.call_args.kwargs["params"]["mimeType"] == "application/pdf"
        assert streams[0].meta["content_type"] == "application/pdf"


class TestSkips:
    def test_skips_folder(self):
        with patch.object(httpx.Client, "get") as mock_get:
            streams = GoogleDriveFetcher().run(access_token="tok", targets=[_drive_doc(mime_type=FOLDER)])["streams"]
        assert streams == []
        mock_get.assert_not_called()

    def test_skips_unexportable_google_type(self):
        form = _drive_doc(mime_type="application/vnd.google-apps.form")
        with patch.object(httpx.Client, "get") as mock_get:
            streams = GoogleDriveFetcher().run(access_token="tok", targets=[form])["streams"]
        assert streams == []
        mock_get.assert_not_called()

    def test_skips_document_without_file_id(self):
        doc = Document(content="snippet", meta={"file_name": "x.pdf"})
        with patch.object(httpx.Client, "get") as mock_get:
            streams = GoogleDriveFetcher().run(access_token="tok", targets=[doc])["streams"]
        assert streams == []
        mock_get.assert_not_called()


class TestInputHandling:
    def test_targets_is_mandatory(self):
        with pytest.raises(TypeError):
            GoogleDriveFetcher().run(access_token="tok")

    def test_invalid_item_type_raises(self):
        with pytest.raises(GoogleDriveConfigError):
            GoogleDriveFetcher().run(access_token="tok", targets=[123])

    def test_empty_input_returns_no_streams(self):
        with patch.object(httpx.Client, "get") as mock_get:
            streams = GoogleDriveFetcher().run(access_token="tok", targets=[])["streams"]
        assert streams == []
        mock_get.assert_not_called()

    def test_dispatches_mixed_documents_and_raw_ids(self):
        fetcher = GoogleDriveFetcher()
        metadata = _json({"id": "RAW", "name": "raw.pdf", "mimeType": "application/pdf"})
        get = _router(metadata=metadata, media=_binary(content=b"x"))
        with patch.object(httpx.Client, "get", side_effect=get):
            streams = fetcher.run(access_token="tok", targets=[_drive_doc(), "RAW"])["streams"]
        assert [s.meta["file_id"] for s in streams] == ["FILE1", "RAW"]

    def test_accepts_secret_access_token(self):
        fetcher = GoogleDriveFetcher()
        with patch.object(httpx.Client, "get", return_value=_binary()) as mock_get:
            fetcher.run(access_token=Secret.from_token("secret-token"), targets=[_drive_doc()])
        assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer secret-token"

    def test_unresolvable_secret_access_token_raises(self):
        unset = Secret.from_env_var("GOOGLE_DRIVE_UNSET_TOKEN", strict=False)
        with pytest.raises(GoogleDriveConfigError):
            GoogleDriveFetcher().run(access_token=unset, targets=[_drive_doc()])


class TestErrorHandling:
    def test_unauthorized_raises_with_status(self):
        fetcher = GoogleDriveFetcher(max_retries=0)
        with patch.object(httpx.Client, "get", return_value=_binary(status_code=401)):
            with pytest.raises(GoogleDriveRequestError) as exc_info:
                fetcher.run(access_token="bad", targets=[_drive_doc()])
        assert exc_info.value.status_code == 401

    def test_forbidden_on_export_raises(self):
        fetcher = GoogleDriveFetcher(max_retries=0)
        get = _router(export=_json({"error": "no"}, status_code=403))
        with patch.object(httpx.Client, "get", side_effect=get):
            with pytest.raises(GoogleDriveRequestError) as exc_info:
                fetcher.run(access_token="tok", targets=[_drive_doc(mime_type=GDOC)])
        assert exc_info.value.status_code == 403

    def test_not_found_raises(self):
        fetcher = GoogleDriveFetcher(max_retries=0)
        with patch.object(httpx.Client, "get", return_value=_binary(status_code=404)):
            with pytest.raises(GoogleDriveRequestError) as exc_info:
                fetcher.run(access_token="tok", targets=[_drive_doc()])
        assert exc_info.value.status_code == 404

    def test_raise_on_failure_false_skips_failed_files(self):
        fetcher = GoogleDriveFetcher(max_retries=0, raise_on_failure=False)
        responses = [_binary(status_code=404), _binary(content=b"ok")]
        docs = [_drive_doc(file_id="MISSING"), _drive_doc(file_id="OK")]
        with patch.object(httpx.Client, "get", side_effect=responses):
            streams = fetcher.run(access_token="tok", targets=docs)["streams"]
        assert len(streams) == 1
        assert streams[0].data == b"ok"

    def test_retries_on_throttling_then_succeeds(self):
        fetcher = GoogleDriveFetcher(max_retries=2)
        throttled = _binary(status_code=429, content_type=None)
        throttled.headers["Retry-After"] = "0"
        with patch.object(httpx.Client, "get", side_effect=[throttled, _binary(content=b"ok")]) as mock_get:
            streams = fetcher.run(access_token="tok", targets=[_drive_doc()])["streams"]
        assert len(streams) == 1
        assert mock_get.call_count == 2

    def test_gives_up_after_max_retries(self):
        fetcher = GoogleDriveFetcher(max_retries=1)
        throttled = _binary(status_code=429, content_type=None)
        throttled.headers["Retry-After"] = "0"
        with patch.object(httpx.Client, "get", return_value=throttled) as mock_get:
            with pytest.raises(GoogleDriveRequestError) as exc_info:
                fetcher.run(access_token="tok", targets=[_drive_doc()])
        assert exc_info.value.status_code == 429
        assert mock_get.call_count == 2


class TestPipeline:
    def test_runs_after_retriever_in_pipeline(self):
        files_list = _json(
            {
                "files": [
                    {
                        "id": "FILE1",
                        "name": "report.pdf",
                        "mimeType": "application/pdf",
                        "webViewLink": WEB_URL,
                    }
                ]
            }
        )
        get = _router(files_list=files_list, media=_binary(content=b"pdf-bytes"))

        pipeline = Pipeline()
        pipeline.add_component("retriever", GoogleDriveRetriever())
        pipeline.add_component("fetcher", GoogleDriveFetcher())
        pipeline.connect("retriever.documents", "fetcher.targets")

        with patch.object(httpx.Client, "get", side_effect=get):
            result = pipeline.run(
                {"retriever": {"query": "report", "access_token": "tok"}, "fetcher": {"access_token": "tok"}}
            )

        streams = result["fetcher"]["streams"]
        assert len(streams) == 1
        assert streams[0].data == b"pdf-bytes"
        assert streams[0].meta["file_id"] == "FILE1"

    def test_serialization_round_trip_in_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("fetcher", GoogleDriveFetcher(raise_on_failure=False))
        restored = Pipeline.from_dict(pipeline.to_dict())
        assert restored.get_component("fetcher").raise_on_failure is False


@pytest.mark.asyncio
class TestRunAsync:
    async def test_downloads_binary(self):
        fetcher = GoogleDriveFetcher()
        get = AsyncMock(return_value=_binary(content=b"async-bytes"))
        with patch.object(httpx.AsyncClient, "get", get):
            streams = (await fetcher.run_async(access_token="tok", targets=[_drive_doc()]))["streams"]
        assert streams[0].data == b"async-bytes"

    async def test_exports_native(self):
        fetcher = GoogleDriveFetcher()
        get = AsyncMock(side_effect=_router(export=_binary(content=b"docx", content_type=None)))
        with patch.object(httpx.AsyncClient, "get", get):
            streams = (await fetcher.run_async(access_token="tok", targets=[_drive_doc(mime_type=GDOC)]))["streams"]
        assert streams[0].mime_type == DOCX

    async def test_unauthorized_raises(self):
        fetcher = GoogleDriveFetcher(max_retries=0)
        get = AsyncMock(return_value=_binary(status_code=401))
        with patch.object(httpx.AsyncClient, "get", get):
            with pytest.raises(GoogleDriveRequestError):
                await fetcher.run_async(access_token="bad", targets=[_drive_doc()])


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.environ.get("GOOGLE_DRIVE_ACCESS_TOKEN") and os.environ.get("GOOGLE_DRIVE_TEST_FILE")),
    reason="GOOGLE_DRIVE_ACCESS_TOKEN and GOOGLE_DRIVE_TEST_FILE not set",
)
class TestLive:
    def test_fetch_against_google_drive(self):
        fetcher = GoogleDriveFetcher()
        streams = fetcher.run(
            access_token=os.environ["GOOGLE_DRIVE_ACCESS_TOKEN"],
            targets=[os.environ["GOOGLE_DRIVE_TEST_FILE"]],
        )["streams"]
        assert len(streams) == 1
        assert streams[0].data
        assert streams[0].meta["file_id"]
