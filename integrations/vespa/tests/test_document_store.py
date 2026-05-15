# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import random
from unittest.mock import Mock, patch

import pytest
from haystack import Document, default_from_dict
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    DocumentStoreBaseExtendedTests,
    GetMetadataFieldsInfoTest,
)
from haystack.utils import Secret

from haystack_integrations.document_stores.vespa import VespaDocumentStore
from haystack_integrations.document_stores.vespa.errors import VespaDocumentStoreError
from haystack_integrations.document_stores.vespa.filters import _normalize_filters

VESPA_TEST_EMBEDDING_DIM = 3


def _random_embeddings_vespa() -> list[float]:
    return [random.random() for _ in range(VESPA_TEST_EMBEDDING_DIM)]  # noqa: S311


VESPA_TEST_EMBEDDING_1 = _random_embeddings_vespa()
VESPA_TEST_EMBEDDING_2 = _random_embeddings_vespa()


def create_vespa_filterable_docs() -> list[Document]:
    """Haystack `create_filterable_docs` shape with Vespa-compatible 3-dimensional embeddings."""

    documents: list[Document] = []

    def emb() -> list[float]:
        return _random_embeddings_vespa()

    for i in range(3):
        documents.extend(
            [
                Document(
                    content=f"A Foo Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "100",
                        "chapter": "intro",
                        "number": 2,
                        "date": "1969-07-21T20:17:40",
                    },
                    embedding=emb(),
                ),
                Document(
                    content=f"A Bar Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "123",
                        "chapter": "abstract",
                        "number": -2,
                        "date": "1972-12-11T19:54:58",
                    },
                    embedding=emb(),
                ),
                Document(
                    content=f"A Foobar Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "90",
                        "chapter": "conclusion",
                        "number": -10,
                        "date": "1989-11-09T17:53:00",
                    },
                    embedding=emb(),
                ),
                Document(
                    content=f"Document {i} without embedding",
                    meta={"name": f"name_{i}", "no_embedding": True, "chapter": "conclusion"},
                ),
                Document(
                    content=f"Doc {i} with zeros emb", meta={"name": "zeros_doc"}, embedding=VESPA_TEST_EMBEDDING_1
                ),
                Document(content=f"Doc {i} with ones emb", meta={"name": "ones_doc"}, embedding=VESPA_TEST_EMBEDDING_2),
            ]
        )
    return documents


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def is_successful(self):
        return 200 <= self.status_code < 300

    def get_json(self):
        return self._payload


@pytest.fixture
def store():
    document_store = VespaDocumentStore(
        url="http://localhost",
        schema="docs",
        namespace="docs",
        metadata_fields=["category", "author"],
    )
    document_store._app = Mock()
    return document_store


def test_to_dict_from_dict():
    document_store = VespaDocumentStore(
        url="http://localhost",
        cert=Secret.from_env_var("VESPA_CERT"),
        key=Secret.from_env_var("VESPA_KEY"),
        vespa_cloud_secret_token=Secret.from_env_var("VESPA_CLOUD_SECRET_TOKEN"),
        additional_headers={"X-Custom-Header": "test"},
        schema="docs",
        namespace="docs",
        metadata_fields=["category"],
    )

    restored = default_from_dict(VespaDocumentStore, document_store.to_dict())

    assert restored.to_dict() == document_store.to_dict()


def test_count_documents(store):
    store._app.query.return_value = DummyResponse({"root": {"fields": {"totalCount": 7}}})

    assert store.count_documents() == 7


def test_app_initializes_pyvespa_with_auth_parameters():
    document_store = VespaDocumentStore(
        url="https://example.vespa-app.cloud",
        port=443,
        cert=Secret.from_token("/path/to/cert.pem"),
        key=Secret.from_token("/path/to/key.pem"),
        vespa_cloud_secret_token=Secret.from_token("token"),
        additional_headers={"X-Custom-Header": "test"},
    )

    with patch("haystack_integrations.document_stores.vespa.document_store.Vespa") as vespa:
        _ = document_store.app

    vespa.assert_called_once_with(
        url="https://example.vespa-app.cloud",
        port=443,
        cert="/path/to/cert.pem",
        key="/path/to/key.pem",
        vespa_cloud_secret_token="token",
        additional_headers={"X-Custom-Header": "test"},
    )


def test_string_equality_filters_use_contains():
    yql_filter = _normalize_filters(
        {"field": "meta.category", "operator": "==", "value": "news"},
        content_field="content",
    )

    assert yql_filter == 'category contains "news"'


def test_string_relational_filters_require_iso_dates():
    with pytest.raises(FilterError):
        _normalize_filters({"field": "meta.number", "operator": ">", "value": "1"}, content_field="content")


def test_normalize_filters_multi_condition_not_clause():
    yql_filter = _normalize_filters(
        {
            "operator": "NOT",
            "conditions": [
                {"field": "meta.number", "operator": "==", "value": 100},
                {"field": "meta.name", "operator": "==", "value": "name_0"},
            ],
        },
        content_field="content",
    )

    assert yql_filter == '!( ( number = 100 and name contains "name_0" ) )'


def test_write_documents(store):
    store._app.feed_data_point.return_value = DummyResponse({})
    store._app.get_data.return_value = DummyResponse({}, status_code=404)

    written = store.write_documents(
        [Document(id="1", content="hello", embedding=[0.1, 0.2], meta={"category": "news", "ignored": "x"})]
    )

    assert written == 1
    _, kwargs = store._app.feed_data_point.call_args
    assert kwargs["schema"] == "docs"
    assert kwargs["data_id"] == "1"
    assert kwargs["fields"]["content"] == "hello"
    assert kwargs["fields"]["embedding"] == [0.1, 0.2]
    assert kwargs["fields"]["category"] == "news"
    assert "id" not in kwargs["fields"]
    assert "ignored" not in kwargs["fields"]


def test_write_documents_duplicate_skip(store):
    store._app.get_data.return_value = DummyResponse({"fields": {"id": "1"}})

    written = store.write_documents([Document(id="1", content="hello")], policy=DuplicatePolicy.SKIP)

    assert written == 0
    store._app.feed_data_point.assert_not_called()


def test_write_documents_duplicate_policy_fail_raises_duplicate_document_error(store):
    store._app.feed_data_point.return_value = DummyResponse({})
    store._app.get_data.side_effect = [
        DummyResponse({}, status_code=404),
        DummyResponse({"fields": {"id": "1"}}, status_code=200),
    ]

    assert store.write_documents([Document(id="1", content="first")]) == 1

    with pytest.raises(DuplicateDocumentError):
        store.write_documents([Document(id="1", content="second")], policy=DuplicatePolicy.FAIL)


def test_write_documents_duplicate_check_surfaces_backend_error(store):
    store._app.get_data.return_value = DummyResponse({"message": "boom"}, status_code=500)

    with pytest.raises(VespaDocumentStoreError):
        store.write_documents([Document(id="1", content="hello")], policy=DuplicatePolicy.SKIP)


def test_write_documents_invalid_inputs_raise(store):
    store._app.get_data.return_value = DummyResponse({}, status_code=404)

    with pytest.raises(ValueError, match="Please provide a list of Documents"):
        store.write_documents("not a list")  # type:ignore[arg-type]

    with pytest.raises(ValueError, match="Please provide a list of Documents"):
        store.write_documents(["bad"])  # type:ignore[arg-type]


def test_filter_documents(store):
    store._app.query.return_value = DummyResponse(
        {
            "root": {
                "children": [
                    {
                        "id": "id:docs:docs::1",
                        "relevance": 3.5,
                        "fields": {
                            "id": "1",
                            "content": "hello",
                            "embedding": [0.1, 0.2],
                            "category": "news",
                        },
                    }
                ]
            }
        }
    )

    documents = store.filter_documents(filters={"field": "meta.category", "operator": "==", "value": "news"})

    assert len(documents) == 1
    assert documents[0].id == "1"
    assert documents[0].score == 3.5
    assert documents[0].meta == {"category": "news"}


def test_filter_documents_with_none_value_uses_python_fallback(store):
    store._query_documents = Mock(  # type:ignore[method-assign]
        return_value=[
            Document(id="1", content="with number", meta={"number": 1}),
            Document(id="2", content="without number"),
        ]
    )

    documents = store.filter_documents(filters={"field": "meta.number", "operator": "==", "value": None})

    assert [document.id for document in documents] == ["2"]
    store._query_documents.assert_called_once_with(where="true", top_k=store.query_limit)


def test_filter_documents_with_iso_date_comparison_uses_python_fallback(store):
    store._query_documents = Mock(  # type:ignore[method-assign]
        return_value=[
            Document(id="1", content="old", meta={"date": "1969-07-21T20:17:40"}),
            Document(id="2", content="new", meta={"date": "1989-11-09T17:53:00"}),
        ]
    )

    documents = store.filter_documents(filters={"field": "meta.date", "operator": ">", "value": "1972-12-11T19:54:58"})

    assert [document.id for document in documents] == ["2"]
    store._query_documents.assert_called_once_with(where="true", top_k=store.query_limit)


def test_bm25_retrieval_uses_bm25_ranking_by_default(store):
    store._app.query.return_value = DummyResponse({"root": {"children": []}})

    store._bm25_retrieval(query="hello")

    _, kwargs = store._app.query.call_args
    assert kwargs["body"]["ranking"] == "bm25"


def test_embedding_retrieval_uses_semantic_ranking_by_default(store):
    store._app.query.return_value = DummyResponse({"root": {"children": []}})

    store._embedding_retrieval(query_embedding=[0.1, 0.2, 0.3])

    _, kwargs = store._app.query.call_args
    assert kwargs["body"]["ranking"] == "semantic"
    assert "targetHits:10" in kwargs["body"]["yql"]


def test_get_documents_by_id(store):
    store._app.get_data.return_value = DummyResponse({"fields": {"id": "1", "content": "hello", "author": "sam"}})

    documents = store.get_documents_by_id(["1"])

    assert [doc.id for doc in documents] == ["1"]
    assert documents[0].meta == {"author": "sam"}


def test_delete_by_filter(store):
    store._app.query.side_effect = [
        DummyResponse(
            {
                "root": {
                    "children": [
                        {"id": "id:docs:docs::1", "fields": {"content": "hello"}},
                        {"id": "id:docs:docs::2", "fields": {"content": "world"}},
                    ]
                }
            }
        ),
        DummyResponse({"root": {"children": []}}),
    ]
    store._app.delete_data.return_value = DummyResponse({})

    deleted = store.delete_by_filter(filters={"field": "meta.category", "operator": "==", "value": "news"})

    assert deleted == 2
    assert store._app.delete_data.call_count == 2


def test_delete_all_documents(store):
    store._app.delete_all_docs.return_value = None

    store.delete_all_documents()

    store._app.delete_all_docs.assert_called_once_with(
        content_cluster_name="content",
        schema="docs",
        namespace="docs",
    )


@pytest.mark.integration
class TestVespaDocumentStoreBase(
    DocumentStoreBaseExtendedTests,
    CountDocumentsByFilterTest,
    GetMetadataFieldsInfoTest,
):
    """
    Runs Haystack `DocumentStoreBaseExtendedTests` (+ count-by-filter / metadata-info) against Dockerized Vespa.
    Requires `VESPA_RUN_INTEGRATION_TESTS=1` and `docker compose up` (see project CI workflow).
    """

    @pytest.fixture
    def document_store(self, deployed_vespa_app, request):  # noqa: ARG002
        """Override the inherited base fixture with a Docker-backed Vespa store."""
        metadata_fields = [
            "category",
            "author",
            "name",
            "page",
            "chapter",
            "number",
            "date",
            "no_embedding",
            "year",
            "status",
            "updated",
            "extra_field",
            "featured",
            "priority",
            "rating",
            "age",
        ]
        store = VespaDocumentStore(
            url="http://localhost",
            schema="doc",
            namespace="doc",
            content_field="content",
            embedding_field="embedding",
            metadata_fields=metadata_fields,
        )
        store.delete_all_documents()
        yield store
        store.delete_all_documents()

    @pytest.fixture
    def filterable_docs(self) -> list[Document]:
        """Vespa embeddings are restricted to tensor dimension 3 in the test schema."""
        return create_vespa_filterable_docs()

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]) -> None:
        assert len(received) == len(expected)
        rs = sorted(received, key=lambda d: str(d.id))
        es = sorted(expected, key=lambda d: str(d.id))
        for r, e in zip(rs, es, strict=True):
            assert str(r.id) == str(e.id)
            assert r.content == e.content
            received_meta = dict(r.meta)
            for key in ("featured", "no_embedding", "updated"):
                if key not in e.meta and received_meta.get(key) is False:
                    received_meta.pop(key)
            assert received_meta == e.meta

    def test_write_documents(self, document_store):  # type:ignore[override]
        docs = [Document(id="1", content="test doc")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)
