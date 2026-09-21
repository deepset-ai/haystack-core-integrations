# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

import httpx
import pytest
from haystack.dataclasses import Document, SparseEmbedding
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    DocumentStoreBaseExtendedTests,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
)
from haystack.utils import Secret

from haystack_integrations.document_stores.solr import (
    SolrDocumentStore,
    SolrDocumentStoreConfigError,
    SolrDocumentStoreError,
)

from .test_document_store_common import SolrDocumentStoreTestMixin


def _system_info(version: str = "10.0.0") -> dict:
    return {"responseHeader": {"status": 0}, "lucene": {"solr-spec-version": version}}


def _bootstrap_aware(query_handler):
    """
    Wrap `query_handler` so the bootstrap round-trips are answered for it.

    Keeps each test focused on the one request it cares about instead of restating the version check,
    the config overlay read and the schema read.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/admin/info/system"):
            return httpx.Response(200, json=_system_info())
        if path.endswith("/config/overlay"):
            return httpx.Response(
                200,
                json={
                    "responseHeader": {"status": 0},
                    "overlay": {"userProps": {"update.autoCreateFields": "false"}},
                },
            )
        if path.endswith("/schema"):
            return httpx.Response(200, json={"schema": {}, "responseHeader": {"status": 0}})
        return query_handler(request)

    return handler


def _mock_store(handler, **kwargs) -> SolrDocumentStore:
    """
    A store whose HTTP calls are served by `handler`.

    Injecting a `MockTransport` rather than patching methods keeps the real request construction -
    URLs, query params, JSON bodies - under test.
    """
    store = SolrDocumentStore(url="http://solr.test/solr", core="unit", auth=None, **kwargs)
    store._solr_client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return store


def _mock_store_async(handler, **kwargs) -> SolrDocumentStore:
    """The async counterpart of `_mock_store`. `MockTransport` serves both client kinds."""
    store = SolrDocumentStore(url="http://solr.test/solr", core="unit", auth=None, **kwargs)
    store._solr_client._async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return store


class TestSerialization:
    def test_to_dict(self):
        store = SolrDocumentStore(url="http://solr.test/solr", core="docs", embedding_dim=4, auth=None)
        data = store.to_dict()
        assert data["type"] == "haystack_integrations.document_stores.solr.document_store.SolrDocumentStore"
        assert data["init_parameters"]["url"] == "http://solr.test/solr"
        assert data["init_parameters"]["core"] == "docs"
        assert data["init_parameters"]["embedding_dim"] == 4
        assert data["init_parameters"]["auth"] is None

    def test_to_dict_serializes_secret_auth(self):
        store = SolrDocumentStore(
            auth=(
                Secret.from_env_var("SOLR_USERNAME", strict=False),
                Secret.from_env_var("SOLR_PASSWORD", strict=False),
            )
        )
        auth = store.to_dict()["init_parameters"]["auth"]
        assert [entry["type"] for entry in auth] == ["env_var", "env_var"]

    def test_to_dict_does_not_leak_plain_credentials(self):
        """A raw username/password pair is dropped rather than written into the pipeline YAML."""
        store = SolrDocumentStore(auth=("admin", "hunter2"))
        assert store.to_dict()["init_parameters"]["auth"] is None

    def test_round_trip(self):
        store = SolrDocumentStore(
            url="http://solr.test/solr",
            core="docs",
            embedding_dim=16,
            similarity_function="dot_product",
            return_embedding=True,
            batch_size=7,
            auth=(
                Secret.from_env_var("SOLR_USERNAME", strict=False),
                Secret.from_env_var("SOLR_PASSWORD", strict=False),
            ),
        )
        restored = SolrDocumentStore.from_dict(store.to_dict())
        assert restored.to_dict() == store.to_dict()
        assert restored._core == "docs"
        assert restored._embedding_dim == 16
        assert restored._similarity_function == "dot_product"
        assert restored._return_embedding is True
        assert restored._batch_size == 7
        assert all(isinstance(value, Secret) for value in restored._auth)

    def test_extra_kwargs_reach_the_http_client(self):
        store = SolrDocumentStore(auth=None, headers={"X-Trace": "1"})
        assert store._solr_client._client_kwargs()["headers"] == {"X-Trace": "1"}

    def test_extra_kwargs_round_trip(self):
        store = SolrDocumentStore(auth=None, headers={"X-Trace": "1"})
        assert store.to_dict()["init_parameters"]["headers"] == {"X-Trace": "1"}
        restored = SolrDocumentStore.from_dict(store.to_dict())
        assert restored._kwargs == {"headers": {"X-Trace": "1"}}

    def test_url_falls_back_to_environment(self, monkeypatch):
        monkeypatch.setenv("SOLR_URL", "http://from-env:8983/solr")
        assert SolrDocumentStore(auth=None)._url == "http://from-env:8983/solr"

    def test_url_default(self, monkeypatch):
        monkeypatch.delenv("SOLR_URL", raising=False)
        assert SolrDocumentStore(auth=None)._url == "http://localhost:8983/solr"


class TestAuth:
    def test_no_auth(self):
        assert SolrDocumentStore(auth=None)._solr_client.resolved_auth() is None

    def test_unresolved_secrets_mean_no_auth(self, monkeypatch):
        monkeypatch.delenv("SOLR_USERNAME", raising=False)
        monkeypatch.delenv("SOLR_PASSWORD", raising=False)
        assert SolrDocumentStore()._solr_client.resolved_auth() is None

    def test_both_credentials(self, monkeypatch):
        monkeypatch.setenv("SOLR_USERNAME", "admin")
        monkeypatch.setenv("SOLR_PASSWORD", "secret")
        assert SolrDocumentStore()._solr_client.resolved_auth() == ("admin", "secret")

    def test_only_one_credential_is_an_error(self, monkeypatch):
        monkeypatch.setenv("SOLR_USERNAME", "admin")
        monkeypatch.delenv("SOLR_PASSWORD", raising=False)
        with pytest.raises(SolrDocumentStoreError, match="both username and password"):
            SolrDocumentStore()._solr_client.resolved_auth()


class TestVersionGate:
    @pytest.mark.parametrize("version", ["9.6.0", "9.10.1", "10.0.0", "11.0.0"])
    def test_supported_versions(self, version):
        SolrDocumentStore(auth=None)._check_version(_system_info(version))

    @pytest.mark.parametrize("version", ["9.0.0", "9.5.0", "8.11.4"])
    def test_unsupported_versions(self, version):
        with pytest.raises(SolrDocumentStoreConfigError, match=r"requires Solr 9\.6 or newer"):
            SolrDocumentStore(auth=None)._check_version(_system_info(version))

    def test_snapshot_suffix_is_tolerated(self):
        SolrDocumentStore(auth=None)._check_version(_system_info("10.1.0-SNAPSHOT"))

    def test_missing_version(self):
        with pytest.raises(SolrDocumentStoreConfigError, match="Could not determine the Solr version"):
            SolrDocumentStore(auth=None)._check_version({"lucene": {}})

    def test_unparseable_version(self):
        with pytest.raises(SolrDocumentStoreConfigError, match="Could not parse the Solr version"):
            SolrDocumentStore(auth=None)._check_version(_system_info("not-a-version"))


class TestCoreAndSchemaChecks:
    def test_core_exists(self):
        assert SolrDocumentStore._core_exists({"status": {"docs": {"name": "docs"}}}, "docs") is True

    def test_missing_core_reports_an_empty_dict(self):
        """Solr answers STATUS for an unknown core with `{}` rather than an error."""
        assert SolrDocumentStore._core_exists({"status": {"docs": {}}}, "docs") is False

    def test_vector_dimension_mismatch_is_rejected(self):
        store = SolrDocumentStore(embedding_dim=768, auth=None)
        schema = {
            "schema": {
                "fields": [{"name": "embedding", "type": "haystack_knn_4"}],
                "fieldTypes": [{"name": "haystack_knn_4", "class": "solr.DenseVectorField", "vectorDimension": 4}],
            }
        }
        with pytest.raises(SolrDocumentStoreConfigError, match="cannot be changed after"):
            store._verify_vector_field(schema)

    def test_matching_vector_dimension_is_accepted(self):
        store = SolrDocumentStore(embedding_dim=4, auth=None)
        store._verify_vector_field(
            {
                "schema": {
                    "fields": [{"name": "embedding", "type": "haystack_knn_4"}],
                    "fieldTypes": [{"name": "haystack_knn_4", "class": "solr.DenseVectorField", "vectorDimension": 4}],
                }
            }
        )

    def test_absent_embedding_field_is_fine(self):
        SolrDocumentStore(auth=None)._verify_vector_field({"schema": {"fields": [], "fieldTypes": []}})

    def test_schemaless_guessing_is_disabled(self):
        assert SolrDocumentStore._disable_schemaless_payload() == {
            "set-user-property": {"update.autoCreateFields": "false"}
        }

    @pytest.mark.parametrize(
        ("overlay", "expected"),
        [
            ({"overlay": {"userProps": {"update.autoCreateFields": "false"}}}, True),
            ({"overlay": {"userProps": {"update.autoCreateFields": "true"}}}, False),
            ({"overlay": {"userProps": {}}}, False),
            ({"overlay": {}}, False),
            ({}, False),
        ],
    )
    def test_schemaless_disable_is_skipped_when_already_set(self, overlay, expected):
        """
        A core built from a shared configset stores its overlay in the configset, not the core.

        Writing it unconditionally would keep rewriting state shared with every other core using that
        configset, so the store reads it first.
        """
        assert SolrDocumentStore._schemaless_already_disabled(overlay) is expected


class TestRequestConstruction:
    """Checks the requests the store actually puts on the wire."""

    def test_bm25_query_uses_edismax(self):
        captured = {}

        def on_query(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"responseHeader": {"status": 0}, "response": {"docs": []}})

        store = _mock_store(_bootstrap_aware(on_query))
        store._bm25_retrieval("apache solr", top_k=5, all_terms_must_match=True)

        assert captured["body"]["params"]["defType"] == "edismax"
        assert captured["body"]["params"]["qf"] == "content"
        assert captured["body"]["params"]["mm"] == "100%"
        assert captured["body"]["limit"] == 5
        assert "score" in captured["body"]["fields"]

    def test_knn_query_goes_into_query_not_filter(self):
        """Inside `fq` Solr applies no implicit graph pre-filter, so `{!knn}` must be the main query."""
        captured = {}

        def on_query(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"responseHeader": {"status": 0}, "response": {"docs": []}})

        store = _mock_store(_bootstrap_aware(on_query))
        store._embedding_retrieval([0.1, 0.2], filters={"field": "meta.page", "operator": "==", "value": "1"}, top_k=3)

        assert captured["body"]["query"].startswith("{!knn f=embedding topK=3}[")
        assert captured["body"]["filter"] == ['meta_s_page:"1"']
        assert captured["body"]["limit"] == 3

    def test_embeddings_are_left_off_the_wire_by_default(self):
        store = SolrDocumentStore(auth=None, return_embedding=False)
        assert "embedding" not in store._requested_fields()
        assert "meta_*" in store._requested_fields()

    def test_embeddings_are_requested_when_wanted(self):
        store = SolrDocumentStore(auth=None, return_embedding=True)
        assert "embedding" in store._requested_fields()

    def test_solr_errors_are_surfaced_with_their_message(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"msg": "undefined field nope", "code": 400}})

        store = _mock_store(handler)
        with pytest.raises(SolrDocumentStoreError, match="undefined field nope"):
            store.count_documents()

    def test_unreachable_server(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            msg = "refused"
            raise httpx.ConnectError(msg)

        store = _mock_store(handler)
        with pytest.raises(SolrDocumentStoreError, match="Could not reach Solr"):
            store.count_documents()


#: One canned payload answering every Solr endpoint the store talks to. Solr responses are dicts and
#: each method reads only the keys it cares about, so one payload carrying all of them serves every
#: call. `nextCursorMark` equals the cursor the store opens with, which is what ends a paging loop.
_FAKE_SOLR: dict = {
    "responseHeader": {"status": 0},
    "response": {
        "numFound": 2,
        "docs": [
            {"id": "1", "content": "first", "meta_l_page": 1, "score": 3.5},
            {"id": "2", "content": "second", "meta_s_title": "t", "score": 1.5},
        ],
    },
    "nextCursorMark": "*",
    "facets": {"min_0": 1, "max_0": 9, "f0_0": {"numBuckets": 3}},  # min/max, then unique counts
    "facet_counts": {"facet_fields": {"meta_l_page": ["1", 2, "5", 1]}},  # classic: value, count, ...
    "fields": {"meta_l_page": {}, "meta_s_title": {}, "_version_": {}},  # Luke
    "doc": None,  # real-time get of an absent id
}

#: The same, for a core holding nothing at all.
_EMPTY_SOLR: dict = {**_FAKE_SOLR, "response": {"numFound": 0, "docs": []}} | {
    "facets": {},
    "facet_counts": {"facet_fields": {}},
    "fields": {},
}


def _serving(payload: dict):
    """A handler answering the bootstrap round-trips, then `payload` for everything else."""
    return _bootstrap_aware(lambda _request: httpx.Response(200, json=payload))


def _result(value):
    """Compare document lists by id, so a table row below stays a one-liner."""
    if isinstance(value, list) and value and isinstance(value[0], Document):
        return [document.id for document in value]
    return value


_FILTERS = {"field": "meta.page", "operator": "==", "value": 1}
_DOC = [Document(id="1", content="x")]

#: (method, args, endpoint it has to reach, result). The async twin is the same name plus `_async`.
#:
#: Every store method issues a request as its first act, so a unit test only gets past that line by
#: being served a fake Solr. Driving the whole surface from one table keeps that cheap, and pins both
#: the endpoint each call uses and what it makes of the response.
_CALLS = [
    ("count_documents", (), "/query", 2),
    ("count_documents_by_filter", (_FILTERS,), "/query", 2),
    ("filter_documents", (), "/query", ["1", "2"]),
    ("filter_documents", (_FILTERS,), "/query", ["1", "2"]),
    ("write_documents", (_DOC, DuplicatePolicy.OVERWRITE), "/update", 1),
    # A deduplicating write consults the real-time get handler, which reports id "1" as present.
    ("write_documents", (_DOC, DuplicatePolicy.SKIP), "/get", 0),
    ("delete_documents", (["1"],), "/update", None),
    ("delete_all_documents", (), "/update", None),
    ("delete_by_filter", (_FILTERS,), "/update", 2),
    ("update_by_filter", (_FILTERS, {"flag": True}), "/update", 2),
    ("get_metadata_fields_info", (), "/admin/luke", {"page": {"type": "int"}, "title": {"type": "str"}}),
    ("get_metadata_field_min_max", ("meta.page",), "/query", {"min": 1, "max": 9}),
    ("count_unique_metadata_by_filter", (_FILTERS, ["page"]), "/query", {"page": 3}),
    # Only classic facets can filter buckets by substring, so this one has to use /select.
    ("get_metadata_field_unique_values", ("page",), "/select", ([1, 5], 2)),
    ("_bm25_retrieval", ("apache solr",), "/query", ["1", "2"]),
    ("_embedding_retrieval", ([0.1, 0.2],), "/query", ["1", "2"]),
]

#: The same table against a store with nothing in it, where every row is an early return. A `None`
#: endpoint means the call answers without reaching Solr at all.
_EMPTY_CALLS = [
    ("write_documents", ([],), None, 0),
    ("write_documents", (_DOC, DuplicatePolicy.FAIL), "/update", 1),  # nothing exists, so nothing is dropped
    ("delete_documents", ([],), None, None),
    ("delete_by_filter", (_FILTERS,), "/query", 0),
    ("update_by_filter", (_FILTERS, {"flag": True}), "/query", 0),
    ("get_metadata_fields_info", (), "/admin/luke", {}),
    ("count_unique_metadata_by_filter", (_FILTERS, ["page"]), "/admin/luke", {"page": 0}),
    ("get_metadata_field_min_max", ("page",), "/admin/luke", {"min": None, "max": None}),
    ("get_metadata_field_unique_values", ("page",), "/admin/luke", ([], 0)),
]

_ALL_CALLS = [(_FAKE_SOLR, *row) for row in _CALLS] + [(_EMPTY_SOLR, *row) for row in _EMPTY_CALLS]
_ALL_IDS = [f"full-{index}-{row[0]}" for index, row in enumerate(_CALLS)] + [
    f"empty-{index}-{row[0]}" for index, row in enumerate(_EMPTY_CALLS)
]


class TestEveryCallAgainstAFakeSolr:
    """Drives the whole store surface over a fake Solr, sync and async."""

    @staticmethod
    def _recording(payload):
        paths: list[str] = []
        handler = _serving(payload)

        def record(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return handler(request)

        return record, paths

    @pytest.mark.parametrize(("payload", "method", "args", "endpoint", "expected"), _ALL_CALLS, ids=_ALL_IDS)
    def test_sync(self, payload, method, args, endpoint, expected):
        handler, paths = self._recording(payload)
        store = _mock_store(handler)
        assert _result(getattr(store, method)(*args)) == expected
        assert endpoint is None or any(path.endswith(endpoint) for path in paths), f"{endpoint} not in {paths}"

    @pytest.mark.parametrize(("payload", "method", "args", "endpoint", "expected"), _ALL_CALLS, ids=_ALL_IDS)
    async def test_async(self, payload, method, args, endpoint, expected):
        handler, paths = self._recording(payload)
        store = _mock_store_async(handler)
        assert _result(await getattr(store, f"{method}_async")(*args)) == expected
        assert endpoint is None or any(path.endswith(endpoint) for path in paths), f"{endpoint} not in {paths}"


class TestPagination:
    """`cursorMark` paging has to follow the cursor Solr hands back, not stop at the first page."""

    @staticmethod
    def _two_pages():
        cursors = iter(["page2", "page2"])  # the repeated cursor is what ends the loop

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/update"):
                return httpx.Response(200, json={"responseHeader": {"status": 0}})
            first = json.loads(request.content)["params"]["cursorMark"] == "*"
            return httpx.Response(
                200,
                json={
                    "responseHeader": {"status": 0},
                    "response": {"numFound": 2, "docs": [{"id": "1" if first else "2", "content": "x"}]},
                    "nextCursorMark": next(cursors),
                },
            )

        return _bootstrap_aware(handler)

    async def test_both_paginating_readers_follow_the_cursor(self):
        """`update_by_filter` reads with its own helper, so that one has to page as well."""
        assert [d.id for d in _mock_store(self._two_pages()).filter_documents()] == ["1", "2"]
        assert _mock_store(self._two_pages()).update_by_filter(_FILTERS, {"flag": True}) == 2

        documents = await _mock_store_async(self._two_pages()).filter_documents_async()
        assert [document.id for document in documents] == ["1", "2"]
        assert await _mock_store_async(self._two_pages()).update_by_filter_async(_FILTERS, {"flag": True}) == 2


class TestPureHelpers:
    """The branchy pure helpers, whose alternatives one trip through a public method cannot reach."""

    @pytest.mark.parametrize(
        ("type_code", "raw", "expected"),
        [
            ("l", "10", 10),
            ("ls", "10", 10),
            ("d", "1.5", 1.5),
            ("ds", "1.5", 1.5),
            ("b", "true", True),
            ("bs", "false", False),
            ("s", "text", "text"),
        ],
    )
    def test_classic_facet_values_are_decoded_by_type_code(self, type_code, raw, expected):
        """Classic facets render every bucket as a string, so the type code restores the type."""
        decoded = SolrDocumentStore._decode_facet_value(raw, type_code)
        assert decoded == expected
        assert isinstance(decoded, type(expected))

    @pytest.mark.parametrize(
        ("type_code", "expected"),
        [("s", "str"), ("l", "int"), ("d", "float"), ("b", "bool"), ("ss", "list[str]"), ("j", "object")],
    )
    def test_python_type_names_per_type_code(self, type_code, expected):
        assert SolrDocumentStore._python_type_for_code(type_code) == expected

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({}, {"commit": True}),
            ({"commit": False}, {}),
            ({"commit_within_ms": 500}, {"commitWithin": 500, "commit": True}),
            ({"commit": False, "commit_within_ms": 500}, {"commitWithin": 500}),
        ],
    )
    def test_update_params(self, kwargs, expected):
        assert SolrDocumentStore(auth=None, **kwargs)._update_params() == expected

    @pytest.mark.parametrize(
        ("facets", "expected"),
        [
            # Solr omits min/max entirely when nothing matched, so a missing value means "no value".
            ({"min_0": None, "max_0": None}, {"min": None, "max": None}),
            # A key stored as both int and float lives in two fields, and the span covers both.
            ({"min_0": 5, "max_0": 5, "min_1": 0.5, "max_1": 9.5}, {"min": 0.5, "max": 9.5}),
        ],
    )
    def test_min_max_spans_every_numeric_field_of_a_key(self, facets, expected):
        assert SolrDocumentStore._min_max_from_response({"facets": facets}) == expected

    def test_only_numeric_fields_can_be_spanned(self):
        fields = {"meta_l_a": "l", "meta_d_b": "d", "meta_s_c": "s", "meta_b_d": "b"}
        assert sorted(SolrDocumentStore._numeric_fields(fields)) == ["meta_d_b", "meta_l_a"]

    def test_a_key_stored_under_several_type_codes_reports_the_first(self):
        payload = {"fields": {"meta_s_page": {}, "meta_l_page": {}}}
        assert SolrDocumentStore()._fields_info_from_luke(payload) == {"page": {"type": "int"}}

    def test_unique_value_params(self):
        params = SolrDocumentStore._unique_values_params("meta_s_a", _FILTERS, "needle")
        assert params["facet.contains"] == "needle"
        assert params["facet.contains.ignoreCase"] is True
        assert params["fq"] == "meta_l_page:1"
        assert "facet.contains" not in SolrDocumentStore._unique_values_params("meta_s_a", None, None)

    def test_no_facets_means_every_field_counts_zero(self):
        assert SolrDocumentStore._unique_counts_from_response({}, {"page": ["f0_0"]}) == {"page": 0}

    def test_merging_meta_keeps_the_document_id(self):
        """The rewrite has to preserve identity, or `update_by_filter` would orphan the original."""
        (merged,) = SolrDocumentStore._merge_meta([Document(id="fixed", content="x", meta={"a": 1})], {"b": 2})
        assert (merged.id, merged.meta) == ("fixed", {"a": 1, "b": 2})

    def test_documents_are_split_into_batches(self):
        documents = [Document(id=str(index), content="x") for index in range(5)]
        assert [len(batch) for batch in SolrDocumentStore(auth=None, batch_size=2)._batches(documents)] == [2, 2, 1]

    def test_a_field_that_is_not_a_vector_field_is_not_checked(self):
        """Only a `DenseVectorField` has a dimension to disagree about."""
        SolrDocumentStore(embedding_dim=768, auth=None)._verify_vector_field(
            {
                "schema": {
                    "fields": [{"name": "embedding", "type": "string"}],
                    "fieldTypes": [{"name": "string", "class": "solr.StrField"}],
                }
            }
        )


class TestLifecycle:
    async def test_close_resets_the_bootstrap(self):
        """The next call has to bootstrap again, because the connection is gone."""
        store = _mock_store(_serving(_FAKE_SOLR))
        store.count_documents()
        store.close()
        assert store._initialized is False
        assert store._solr_client._client is None

        store = _mock_store_async(_serving(_FAKE_SOLR))
        await store.count_documents_async()
        await store.close_async()
        assert store._async_initialized is False
        assert store._solr_client._async_client is None

    @pytest.mark.parametrize(
        ("call", "message"),
        [
            (lambda store: store._bm25_retrieval_async(""), "query must be a non empty string"),
            (lambda store: store._embedding_retrieval_async([]), "must be a non-empty list of floats"),
        ],
    )
    async def test_async_retrieval_validates_its_input(self, call, message):
        with pytest.raises(ValueError, match=message):
            await call(SolrDocumentStore(auth=None))

    @pytest.mark.parametrize(("from_", "size", "expected"), [(0, 1, [1]), (1, 1, [5]), (2, 1, []), (1, 10, [5])])
    def test_unique_values_are_paginated_but_the_total_is_not(self, from_, size, expected):
        """The count has to cover every matching value, not just the page being returned."""
        store = _mock_store(_serving(_FAKE_SOLR))
        assert store.get_metadata_field_unique_values("page", from_=from_, size=size) == (expected, 2)

    async def test_json_encoded_metadata_cannot_be_faceted(self):
        """
        A JSON fallback field is opaque to Solr and not indexed, so it has no buckets to list.

        The fake offers buckets for it anyway, so a store that forgot to skip the field would return
        them and fail here rather than passing on an empty response.
        """
        handler = _serving(
            {
                **_FAKE_SOLR,
                "fields": {"meta_j_nested": {}},
                "facet_counts": {"facet_fields": {"meta_j_nested": ['{"x": 1}', 3]}},
            }
        )
        assert _mock_store(handler).get_metadata_field_unique_values("nested") == ([], 0)
        assert await _mock_store_async(handler).get_metadata_field_unique_values_async("nested") == ([], 0)


class TestBootstrap:
    """The sequence of calls the store makes on first use."""

    @staticmethod
    def _recording_handler(paths, *, core_exists=True, schemaless_off=True):
        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            paths.append(f"{request.method} {path}")
            if path.endswith("/admin/info/system"):
                return httpx.Response(200, json=_system_info())
            if path.endswith("/admin/cores"):
                action = request.url.params.get("action")
                if action == "STATUS":
                    status = {"unit": {"name": "unit"} if core_exists else {}}
                    return httpx.Response(200, json={"responseHeader": {"status": 0}, "status": status})
                return httpx.Response(200, json={"responseHeader": {"status": 0}, "core": "unit"})
            if path.endswith("/config/overlay"):
                return httpx.Response(
                    200,
                    json={
                        "responseHeader": {"status": 0},
                        # `autoCreateFields: false` is what "schemaless guessing is already off" looks like.
                        "overlay": {"userProps": {"update.autoCreateFields": "false" if schemaless_off else "true"}},
                    },
                )
            if path.endswith("/schema"):
                return httpx.Response(200, json={"schema": {}, "responseHeader": {"status": 0}})
            return httpx.Response(200, json={"responseHeader": {"status": 0}, "response": {"numFound": 0}})

        return handler

    def test_schemaless_guessing_is_turned_off_when_still_on(self):
        """The overlay is only written when it does not already say what the store needs."""
        paths: list[str] = []
        store = _mock_store(self._recording_handler(paths, schemaless_off=False))
        store.count_documents()
        assert "POST /solr/unit/config" in paths

    def test_schemaless_guessing_is_left_alone_when_already_off(self):
        paths: list[str] = []
        store = _mock_store(self._recording_handler(paths, schemaless_off=True))
        store.count_documents()
        assert "POST /solr/unit/config" not in paths

    async def test_the_async_bootstrap_creates_the_core_and_the_schema(self):
        """The async bootstrap is a separate code path and has to do the same work as the sync one."""
        paths: list[str] = []
        store = _mock_store_async(
            self._recording_handler(paths, core_exists=False, schemaless_off=False),
            create_core=True,
        )
        await store.count_documents_async()

        assert sum("/admin/cores" in path for path in paths) == 2  # STATUS, then CREATE
        assert "POST /solr/unit/config" in paths
        assert any(path.endswith("/schema") for path in paths)

    async def test_the_async_bootstrap_runs_only_once(self):
        paths: list[str] = []
        store = _mock_store_async(self._recording_handler(paths))
        await store.count_documents_async()
        await store.count_documents_async()
        assert sum(path.endswith("/admin/info/system") for path in paths) == 1

    def test_manage_schema_false_touches_no_schema_endpoints(self):
        """Opting out has to mean the store leaves the core's configuration completely alone."""
        paths: list[str] = []
        store = _mock_store(self._recording_handler(paths), manage_schema=False)
        store.count_documents()
        assert not any("/schema" in path or "/config" in path for path in paths)

    def test_schema_is_provisioned_by_default(self):
        paths: list[str] = []
        store = _mock_store(self._recording_handler(paths))
        store.count_documents()
        assert any(path.endswith("/config/overlay") for path in paths)
        assert any(path.endswith("/schema") for path in paths)

    def test_core_is_not_created_by_default(self):
        paths: list[str] = []
        store = _mock_store(self._recording_handler(paths))
        store.count_documents()
        assert not any("/admin/cores" in path for path in paths)

    def test_missing_core_is_created_when_asked(self):
        paths: list[str] = []
        store = _mock_store(self._recording_handler(paths, core_exists=False), create_core=True)
        store.count_documents()
        assert sum("/admin/cores" in path for path in paths) == 2  # STATUS, then CREATE

    def test_existing_core_is_left_alone(self):
        paths: list[str] = []
        store = _mock_store(self._recording_handler(paths, core_exists=True), create_core=True)
        store.count_documents()
        assert sum("/admin/cores" in path for path in paths) == 1  # STATUS only

    def test_bootstrap_runs_only_once(self):
        paths: list[str] = []
        store = _mock_store(self._recording_handler(paths))
        store.count_documents()
        store.count_documents()
        assert sum(path.endswith("/admin/info/system") for path in paths) == 1


class TestInputValidation:
    def test_write_documents_rejects_non_list(self):
        with pytest.raises(ValueError, match="Documents must be a list"):
            SolrDocumentStore._validate_documents("not a list")

    def test_write_documents_rejects_non_documents(self):
        with pytest.raises(ValueError, match="must contain a list of objects of type Document"):
            SolrDocumentStore._validate_documents(["not a document"])

    def test_bm25_rejects_empty_query(self):
        with pytest.raises(ValueError, match="query must be a non empty string"):
            SolrDocumentStore(auth=None)._bm25_retrieval("")

    def test_embedding_retrieval_rejects_empty_embedding(self):
        with pytest.raises(ValueError, match="must be a non-empty list of floats"):
            SolrDocumentStore(auth=None)._embedding_retrieval([])

    def test_none_policy_becomes_fail(self):
        store = SolrDocumentStore(auth=None)
        assert store._resolve_policy(DuplicatePolicy.NONE) == DuplicatePolicy.FAIL
        assert store._resolve_policy(DuplicatePolicy.SKIP) == DuplicatePolicy.SKIP

    def test_sparse_embeddings_warn(self, caplog):
        documents = [Document(content="x", sparse_embedding=SparseEmbedding(indices=[0], values=[1.0]))]
        SolrDocumentStore._warn_about_sparse_embeddings(documents)
        assert "no sparse vector field" in caplog.text

    def test_metadata_keys_are_validated_on_write(self):
        store = _mock_store(lambda _request: httpx.Response(200, json={"responseHeader": {"status": 0}}))
        store._initialized = True
        with pytest.raises(ValueError, match="Metadata keys must contain only"):
            store.write_documents([Document(content="x", meta={"bad key": 1})], DuplicatePolicy.OVERWRITE)


#: Ids that a query-based duplicate check mangles: Lucene syntax characters, whitespace, and the
#: comma that a `{!terms}` clause uses to separate its values.
NASTY_IDS = ["a-b", "a b", "a:b", "a,b", "a\\b", 'a"b', "a\nb", "a+b", "a/b", "plain"]


class TestDuplicateDetection:
    """The existing-id lookup has to reach Solr in a form that cannot mangle the ids."""

    def test_ids_are_passed_through_verbatim(self):
        """The real-time get handler parses nothing, so an escaped id would simply never match."""
        assert SolrDocumentStore._existing_ids_payload(NASTY_IDS) == {"params": {"id": NASTY_IDS, "fl": "id"}}

    def test_the_id_list_is_copied(self):
        ids = ["1"]
        payload = SolrDocumentStore._existing_ids_payload(ids)
        ids.append("2")
        assert payload["params"]["id"] == ["1"]

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ({"response": {"docs": [{"id": "1"}, {"id": "2"}]}}, {"1", "2"}),
            ({"response": {"numFound": 0, "docs": []}}, set()),
            ({"response": {}}, set()),
            # A lookup of exactly one id answers with a bare doc instead of a response block.
            ({"doc": {"id": "1"}}, {"1"}),
            ({"doc": None}, set()),
            ({}, set()),
        ],
    )
    def test_both_response_shapes_are_understood(self, payload, expected):
        assert SolrDocumentStore._existing_ids_from_response(payload) == expected

    def test_the_lookup_uses_the_real_time_get_handler(self):
        """A search would miss uncommitted writes, so the check must not go to `/query`."""
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            if request.url.path.endswith("/get"):
                return httpx.Response(200, json={"doc": {"id": "a-b"}})
            return httpx.Response(200, json={"responseHeader": {"status": 0}})

        store = _mock_store(_bootstrap_aware(handler))
        with pytest.raises(DuplicateDocumentError):
            store.write_documents([Document(id="a-b", content="x")], DuplicatePolicy.FAIL)

        assert any(path.endswith("/get") for path in paths)
        assert not any(path.endswith("/query") for path in paths)


class TestFuzziness:
    def test_zero_leaves_the_query_untouched(self):
        assert SolrDocumentStore._apply_fuzziness('apache "solr search"', 0) == 'apache "solr search"'

    def test_bare_terms_get_a_fuzzy_suffix(self):
        assert SolrDocumentStore._apply_fuzziness("apache solr", 1) == "apache~1 solr~1"

    def test_quoted_phrases_are_left_alone(self):
        """`"a b"~1` is a proximity slop, not a fuzzy phrase, so a phrase must not be suffixed."""
        assert SolrDocumentStore._apply_fuzziness('"apache solr" other', 1) == '"apache solr" other~1'

    def test_a_prefixed_phrase_is_one_token(self):
        """`+`/`-` glued to a phrase belongs to the phrase, so the whole token stays unsuffixed."""
        assert SolrDocumentStore._apply_fuzziness('+"apache solr" -cooking', 2) == '+"apache solr" -cooking~2'

    def test_field_qualified_terms_still_get_a_suffix(self):
        assert SolrDocumentStore._apply_fuzziness("content:apache", 1) == "content:apache~1"

    def test_an_unbalanced_quote_is_not_dropped(self):
        """Substituting rather than re-joining is what keeps a malformed query's characters intact."""
        assert SolrDocumentStore._apply_fuzziness('"apache solr', 1) == '"apache~1 solr~1'


class TestScoreScaling:
    def test_scale_score_is_actually_applied(self):
        """The scaled score has to land back on the documents the caller receives."""
        documents = [Document(content="a", score=8.0), Document(content="b", score=None)]
        scaled = SolrDocumentStore._scale_scores(documents)
        assert scaled[0].score == pytest.approx(1 / (1 + pow(2.718281828459045, -1.0)))
        assert 0 < scaled[0].score < 1
        assert scaled[1].score is None

    def test_scaling_is_monotonic(self):
        documents = SolrDocumentStore._scale_scores(
            [Document(content="a", score=1.0), Document(content="b", score=10.0)]
        )
        assert documents[0].score < documents[1].score


@pytest.mark.integration
class TestSolrDocumentStore(
    SolrDocumentStoreTestMixin,
    DocumentStoreBaseExtendedTests,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
):
    """The shared document store suites, run against a real Solr core."""

    @pytest.fixture
    def document_store(self, document_store):
        # Pull in the conftest fixture, overriding the plain one the base classes declare.
        return document_store


@pytest.mark.integration
class TestSolrSpecificBehaviour:
    def test_metadata_types_survive_a_round_trip(self, document_store):
        document = Document(
            id="1",
            content="typed metadata",
            meta={
                "page": "100",
                "number": 2,
                "rating": 0.5,
                "flag": True,
                "tags": ["a", "b"],
                "nested": {"x": 1},
            },
        )
        document_store.write_documents([document], DuplicatePolicy.OVERWRITE)
        restored = document_store.filter_documents()[0]
        assert restored.meta == document.meta
        assert isinstance(restored.meta["page"], str)
        assert isinstance(restored.meta["number"], int)
        assert isinstance(restored.meta["rating"], float)
        assert isinstance(restored.meta["flag"], bool)

    def test_empty_content_is_not_dropped(self, document_store):
        """Solr's schemaless chain strips blank values; the store must have disabled it."""
        document_store.write_documents([Document(id="1", content="")], DuplicatePolicy.OVERWRITE)
        assert document_store.filter_documents()[0].content == ""

    def test_numeric_looking_strings_stay_strings(self, document_store):
        """Solr's `parse-long` processor would turn "100" into 100 if it were still active."""
        document_store.write_documents([Document(id="1", content="x", meta={"page": "100"})], DuplicatePolicy.OVERWRITE)
        assert document_store.filter_documents()[0].meta["page"] == "100"

    def test_embeddings_round_trip(self, document_store):
        embedding = [0.1] * 768
        document_store.write_documents([Document(id="1", content="x", embedding=embedding)], DuplicatePolicy.OVERWRITE)
        restored = document_store.filter_documents()[0]
        assert restored.embedding == pytest.approx(embedding, abs=1e-5)

    def test_embeddings_are_omitted_by_default(self, document_store_no_embedding_returned):
        """`return_embedding=False` keeps vectors off the wire entirely."""
        store = document_store_no_embedding_returned
        store.write_documents([Document(id="1", content="x", embedding=[0.1] * 768)], DuplicatePolicy.OVERWRITE)
        assert store.filter_documents()[0].embedding is None

    def test_pagination_beyond_one_page(self, document_store):
        """`filter_documents` pages with cursorMark, so it must not stop at the page size."""
        document_store._query_page_size = 10
        documents = [Document(id=str(index), content=f"doc {index}") for index in range(35)]
        document_store.write_documents(documents, DuplicatePolicy.OVERWRITE)
        assert len(document_store.filter_documents()) == 35

    def test_bm25_retrieval(self, document_store):
        document_store.write_documents(
            [
                Document(id="1", content="Apache Solr is a search platform"),
                Document(id="2", content="Completely unrelated text about cooking"),
            ],
            DuplicatePolicy.OVERWRITE,
        )
        results = document_store._bm25_retrieval("Apache Solr search", top_k=10)
        assert results[0].id == "1"
        assert results[0].score is not None

    def test_bm25_scale_score(self, document_store):
        document_store.write_documents(
            [Document(id="1", content="Apache Solr search platform")], DuplicatePolicy.OVERWRITE
        )
        scaled = document_store._bm25_retrieval("Apache Solr", top_k=1, scale_score=True)
        assert 0 < scaled[0].score < 1

    def test_bm25_all_terms_must_match(self, document_store):
        document_store.write_documents(
            [
                Document(id="1", content="alpha beta gamma"),
                Document(id="2", content="alpha only"),
            ],
            DuplicatePolicy.OVERWRITE,
        )
        loose = document_store._bm25_retrieval("alpha gamma", top_k=10)
        strict = document_store._bm25_retrieval("alpha gamma", top_k=10, all_terms_must_match=True)
        assert {document.id for document in loose} == {"1", "2"}
        assert {document.id for document in strict} == {"1"}

    def test_embedding_retrieval_respects_filters(self, document_store):
        document_store.write_documents(
            [
                Document(id="1", content="a", meta={"group": "x"}, embedding=[1.0] + [0.0] * 767),
                Document(id="2", content="b", meta={"group": "y"}, embedding=[1.0] + [0.0] * 767),
            ],
            DuplicatePolicy.OVERWRITE,
        )
        results = document_store._embedding_retrieval(
            [1.0] + [0.0] * 767,
            filters={"field": "meta.group", "operator": "==", "value": "x"},
            top_k=10,
        )
        assert [document.id for document in results] == ["1"]

    def test_unique_values_keeps_distinct_types_apart(self, document_store):
        """The int 1, the str "1", the float 1.0 and True occupy four different Solr fields."""
        document_store.write_documents(
            [
                Document(id="1", content="a", meta={"priority": 1}),
                Document(id="2", content="b", meta={"priority": "1"}),
                Document(id="3", content="c", meta={"priority": 1.0}),
                Document(id="4", content="d", meta={"priority": True}),
            ],
            DuplicatePolicy.OVERWRITE,
        )
        values, total = document_store.get_metadata_field_unique_values("priority")
        assert total == 4
        assert {type(value) for value in values} == {int, str, float, bool}

    def test_update_by_filter_preserves_embeddings(self, document_store):
        """The rewrite has to carry the embedding along, or updating metadata would drop the vector."""
        embedding = [0.25] * 768
        document_store.write_documents(
            [Document(id="1", content="x", meta={"kind": "a"}, embedding=embedding)],
            DuplicatePolicy.OVERWRITE,
        )
        document_store.update_by_filter({"field": "meta.kind", "operator": "==", "value": "a"}, {"updated": True})
        restored = document_store.filter_documents()[0]
        assert restored.embedding == pytest.approx(embedding, abs=1e-5)
        assert restored.meta == {"kind": "a", "updated": True}

    def test_update_by_filter_handles_a_value_changing_type(self, document_store):
        """
        Changing a metadata value's type must not leave the old value behind.

        The type code is part of the field name, so `1` lives in `meta_l_n` and `"one"` in `meta_s_n`.
        An atomic update would set the latter and leave the former in place; rewriting the whole
        document replaces it outright.
        """
        document_store.write_documents([Document(id="1", content="x", meta={"n": 1})], DuplicatePolicy.OVERWRITE)
        document_store.update_by_filter({"field": "id", "operator": "==", "value": "1"}, {"n": "one"})
        assert document_store.filter_documents()[0].meta == {"n": "one"}

    def test_reconnects_after_close(self, document_store):
        document_store.write_documents([Document(id="1", content="x")], DuplicatePolicy.OVERWRITE)
        document_store.close()
        assert document_store.count_documents() == 1

    @pytest.mark.parametrize("document_id", NASTY_IDS)
    def test_duplicate_detection_survives_special_characters_in_ids(self, document_store, document_id):
        """
        Lucene syntax, whitespace or a comma in an id must not defeat the duplicate check.

        A query-based lookup has to escape those characters, and the escaping is exactly what breaks
        it: the ids stop matching, so `FAIL` overwrites without raising and `SKIP` replaces the
        stored document instead of leaving it alone.
        """
        document_store.write_documents([Document(id=document_id, content="ORIGINAL")], DuplicatePolicy.OVERWRITE)

        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents([Document(id=document_id, content="CLOBBER")], DuplicatePolicy.FAIL)

        written = document_store.write_documents([Document(id=document_id, content="CLOBBER")], DuplicatePolicy.SKIP)
        assert written == 0
        assert document_store.filter_documents()[0].content == "ORIGINAL"

    def test_duplicate_detection_across_a_batch(self, document_store):
        """Several ids answer with a `response` block where a single one answers with a bare doc."""
        ids = ["a-b", "a b", "a,b", "plain"]
        document_store.write_documents(
            [Document(id=document_id, content="ORIGINAL") for document_id in ids], DuplicatePolicy.OVERWRITE
        )

        assert (
            document_store.write_documents(
                [Document(id=document_id, content="CLOBBER") for document_id in ids], DuplicatePolicy.SKIP
            )
            == 0
        )
        # A mixed batch writes the new document and skips the one that is already there.
        assert (
            document_store.write_documents(
                [Document(id="a-b", content="CLOBBER"), Document(id="fresh", content="NEW")],
                DuplicatePolicy.SKIP,
            )
            == 1
        )
        assert {document.id: document.content for document in document_store.filter_documents()} == {
            "a-b": "ORIGINAL",
            "a b": "ORIGINAL",
            "a,b": "ORIGINAL",
            "plain": "ORIGINAL",
            "fresh": "NEW",
        }

    def test_duplicate_detection_sees_uncommitted_documents(self, solr_store):
        """
        With `commit=False` a search cannot see the previous write, but a real-time get can.

        That is the configuration a deployment which does not want to commit on every batch runs in,
        and duplicate detection has to keep working there.
        """
        store = solr_store(commit=False)
        store.write_documents([Document(id="1", content="ORIGINAL")], DuplicatePolicy.OVERWRITE)

        with pytest.raises(DuplicateDocumentError):
            store.write_documents([Document(id="1", content="CLOBBER")], DuplicatePolicy.FAIL)
        assert store.write_documents([Document(id="1", content="CLOBBER")], DuplicatePolicy.SKIP) == 0
