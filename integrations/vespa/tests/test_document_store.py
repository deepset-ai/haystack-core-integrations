# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import random
from unittest.mock import patch

import pytest
from haystack import Document, default_from_dict
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    DocumentStoreBaseExtendedTests,
    GetMetadataFieldsInfoTest,
)
from haystack.utils import Secret

from haystack_integrations.document_stores.vespa import VespaDocumentStore
from haystack_integrations.document_stores.vespa.errors import VespaDocumentStoreError

from .conftest import VESPA_URL, DummyResponse

VESPA_TEST_EMBEDDING_DIM = 3


def _random_embeddings_vespa() -> list[float]:
    return [random.random() for _ in range(VESPA_TEST_EMBEDDING_DIM)]  # noqa: S311


VESPA_TEST_EMBEDDING_1 = _random_embeddings_vespa()
VESPA_TEST_EMBEDDING_2 = _random_embeddings_vespa()


def create_vespa_filterable_docs() -> list[Document]:
    """`haystack.testing.document_store.create_filterable_docs` with Vespa-compatible 3-d embeddings."""
    documents: list[Document] = []
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
                    embedding=_random_embeddings_vespa(),
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
                    embedding=_random_embeddings_vespa(),
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
                    embedding=_random_embeddings_vespa(),
                ),
                Document(
                    content=f"Document {i} without embedding",
                    meta={"name": f"name_{i}", "no_embedding": True, "chapter": "conclusion"},
                ),
                Document(
                    content=f"Doc {i} with zeros emb",
                    meta={"name": "zeros_doc"},
                    embedding=VESPA_TEST_EMBEDDING_1,
                ),
                Document(
                    content=f"Doc {i} with ones emb",
                    meta={"name": "ones_doc"},
                    embedding=VESPA_TEST_EMBEDDING_2,
                ),
            ]
        )
    return documents


class TestVespaDocumentStoreUnit:
    """Unit tests against a `VespaDocumentStore` with a mocked pyvespa app."""

    def test_to_dict(self):
        document_store = VespaDocumentStore(
            url="http://localhost",
            schema="docs",
            namespace="docs",
            metadata_fields=["category"],
        )

        assert document_store.to_dict() == {
            "type": "haystack_integrations.document_stores.vespa.document_store.VespaDocumentStore",
            "init_parameters": {
                "url": "http://localhost",
                "port": 8080,
                "cert": None,
                "key": None,
                "vespa_cloud_secret_token": {
                    "type": "env_var",
                    "env_vars": ["VESPA_CLOUD_SECRET_TOKEN"],
                    "strict": False,
                },
                "additional_headers": None,
                "content_cluster_name": "content",
                "schema": "docs",
                "namespace": "docs",
                "groupname": None,
                "content_field": "content",
                "embedding_field": "embedding",
                "id_field": "id",
                "metadata_fields": ["category"],
                "query_limit": 400,
            },
        }

    def test_to_dict_from_dict_roundtrip(self):
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

    def test_count_documents(self, mock_store):
        mock_store._app.query.return_value = DummyResponse({"root": {"fields": {"totalCount": 7}}})

        assert mock_store.count_documents() == 7

    def test_app_initializes_pyvespa_with_auth_parameters(self):
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

    def test_write_documents(self, mock_store):
        mock_store._app.feed_data_point.return_value = DummyResponse({})
        mock_store._app.get_data.return_value = DummyResponse({}, status_code=404)

        written = mock_store.write_documents(
            [Document(id="1", content="hello", embedding=[0.1, 0.2], meta={"category": "news", "ignored": "x"})]
        )

        assert written == 1
        _, kwargs = mock_store._app.feed_data_point.call_args
        assert kwargs["schema"] == "docs"
        assert kwargs["data_id"] == "1"
        assert kwargs["fields"]["content"] == "hello"
        assert kwargs["fields"]["embedding"] == [0.1, 0.2]
        assert kwargs["fields"]["category"] == "news"
        assert "id" not in kwargs["fields"]
        assert "ignored" not in kwargs["fields"]

    def test_write_documents_duplicate_skip(self, mock_store):
        mock_store._app.get_data.return_value = DummyResponse({"fields": {"id": "1"}})

        written = mock_store.write_documents([Document(id="1", content="hello")], policy=DuplicatePolicy.SKIP)

        assert written == 0
        mock_store._app.feed_data_point.assert_not_called()

    def test_write_documents_duplicate_policy_fail_raises_duplicate_document_error(self, mock_store):
        mock_store._app.feed_data_point.return_value = DummyResponse({})
        mock_store._app.get_data.side_effect = [
            DummyResponse({}, status_code=404),
            DummyResponse({"fields": {"id": "1"}}, status_code=200),
        ]

        assert mock_store.write_documents([Document(id="1", content="first")]) == 1

        with pytest.raises(DuplicateDocumentError):
            mock_store.write_documents([Document(id="1", content="second")], policy=DuplicatePolicy.FAIL)

    def test_write_documents_duplicate_check_surfaces_backend_error(self, mock_store):
        mock_store._app.get_data.return_value = DummyResponse({"message": "boom"}, status_code=500)

        with pytest.raises(VespaDocumentStoreError):
            mock_store.write_documents([Document(id="1", content="hello")], policy=DuplicatePolicy.SKIP)

    def test_write_documents_invalid_inputs_raise(self, mock_store):
        mock_store._app.get_data.return_value = DummyResponse({}, status_code=404)

        with pytest.raises(ValueError, match="Please provide a list of Documents"):
            mock_store.write_documents("not a list")  # type:ignore[arg-type]

        with pytest.raises(ValueError, match="Please provide a list of Documents"):
            mock_store.write_documents(["bad"])  # type:ignore[arg-type]

    def test_bm25_retrieval_uses_bm25_ranking_by_default(self, mock_store):
        mock_store._app.query.return_value = DummyResponse({"root": {"children": []}})

        mock_store._bm25_retrieval(query="hello")

        _, kwargs = mock_store._app.query.call_args
        assert kwargs["body"]["ranking"] == "bm25"

    def test_embedding_retrieval_uses_semantic_ranking_by_default(self, mock_store):
        mock_store._app.query.return_value = DummyResponse({"root": {"children": []}})

        mock_store._embedding_retrieval(query_embedding=[0.1, 0.2, 0.3])

        _, kwargs = mock_store._app.query.call_args
        assert kwargs["body"]["ranking"] == "semantic"
        assert "targetHits:10" in kwargs["body"]["yql"]

    def test_get_documents_by_id(self, mock_store):
        mock_store._app.get_data.return_value = DummyResponse(
            {"fields": {"id": "1", "content": "hello", "author": "sam"}}
        )

        documents = mock_store.get_documents_by_id(["1"])

        assert [doc.id for doc in documents] == ["1"]
        assert documents[0].meta == {"author": "sam"}

    def test_delete_by_filter(self, mock_store):
        mock_store._app.query.side_effect = [
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
        mock_store._app.delete_data.return_value = DummyResponse({})

        deleted = mock_store.delete_by_filter(filters={"field": "meta.category", "operator": "==", "value": "news"})

        assert deleted == 2
        assert mock_store._app.delete_data.call_count == 2

    def test_delete_all_documents(self, mock_store):
        mock_store._app.delete_all_docs.return_value = None

        mock_store.delete_all_documents()

        mock_store._app.delete_all_docs.assert_called_once_with(
            content_cluster_name="content",
            schema="docs",
            namespace="docs",
        )


@pytest.mark.integration
class TestVespaDocumentStoreIntegration(
    DocumentStoreBaseExtendedTests,
    CountDocumentsByFilterTest,
    GetMetadataFieldsInfoTest,
):

    @pytest.fixture
    def document_store(self):
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
            url=VESPA_URL,
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
        # Compare by ID only — matches the qdrant pattern. Vespa returns extra boolean meta defaults
        # (e.g. `featured: False`) for attributes declared in the schema but absent from the input
        # document, so a strict meta comparison would fail spuriously without testing anything useful
        # about filter correctness.
        assert {doc.id for doc in received} == {doc.id for doc in expected}

    def test_write_documents(self, document_store):  # type:ignore[override]
        # The base `WriteDocumentsTest.test_write_documents` is abstract (raises NotImplementedError);
        # every DS integration must override it to declare its default duplicate-handling policy.
        # Vespa's default policy is FAIL, so writing the same document twice raises.
        docs = [Document(id="1", content="test doc")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)
