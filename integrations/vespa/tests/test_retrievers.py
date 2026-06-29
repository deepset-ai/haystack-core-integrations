# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import time
from unittest.mock import Mock

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict

from haystack_integrations.components.retrievers.vespa import VespaEmbeddingRetriever, VespaKeywordRetriever
from haystack_integrations.document_stores.vespa import VespaDocumentStore

KEYWORD_RETRIEVER_TYPE = "haystack_integrations.components.retrievers.vespa.keyword_retriever.VespaKeywordRetriever"
EMBEDDING_RETRIEVER_TYPE = (
    "haystack_integrations.components.retrievers.vespa.embedding_retriever.VespaEmbeddingRetriever"
)


def _document_store() -> VespaDocumentStore:
    return VespaDocumentStore(url="http://localhost", schema="docs", namespace="docs")


def _doc_store_dict() -> dict:
    return _document_store().to_dict()


class TestVespaKeywordRetriever:
    def test_init_default(self):
        document_store = _document_store()
        retriever = VespaKeywordRetriever(document_store=document_store)
        assert retriever.document_store is document_store
        assert retriever.filters is None
        assert retriever.top_k == 10
        assert retriever.ranking == "bm25"

    def test_init(self):
        document_store = _document_store()
        retriever = VespaKeywordRetriever(
            document_store=document_store,
            filters={"field": "meta.category", "operator": "==", "value": "news"},
            top_k=7,
            ranking="custom_bm25",
        )
        assert retriever.filters == {"field": "meta.category", "operator": "==", "value": "news"}
        assert retriever.top_k == 7
        assert retriever.ranking == "custom_bm25"

    def test_init_rejects_invalid_document_store(self):
        with pytest.raises(ValueError, match="VespaDocumentStore"):
            VespaKeywordRetriever(document_store="not a store")  # type:ignore[arg-type]

    def test_to_dict(self):
        retriever = VespaKeywordRetriever(
            document_store=_document_store(),
            filters={"field": "meta.category", "operator": "==", "value": "news"},
            top_k=7,
            ranking="bm25",
        )

        assert component_to_dict(retriever, "retriever") == {
            "type": KEYWORD_RETRIEVER_TYPE,
            "init_parameters": {
                "document_store": _doc_store_dict(),
                "filters": {"field": "meta.category", "operator": "==", "value": "news"},
                "top_k": 7,
                "ranking": "bm25",
            },
        }

    def test_from_dict(self):
        data = {
            "type": KEYWORD_RETRIEVER_TYPE,
            "init_parameters": {
                "document_store": _doc_store_dict(),
                "filters": {"field": "meta.category", "operator": "==", "value": "news"},
                "top_k": 7,
                "ranking": "bm25",
            },
        }
        retriever = component_from_dict(VespaKeywordRetriever, data, "retriever")
        assert isinstance(retriever.document_store, VespaDocumentStore)
        assert retriever.filters == {"field": "meta.category", "operator": "==", "value": "news"}
        assert retriever.top_k == 7
        assert retriever.ranking == "bm25"

    def test_run_forwards_init_arguments(self):
        document_store = _document_store()
        document_store._bm25_retrieval = Mock(return_value=[Document(id="1", content="hello")])

        retriever = VespaKeywordRetriever(
            document_store=document_store,
            filters={"field": "meta.category", "operator": "==", "value": "news"},
            top_k=5,
            ranking="bm25",
        )
        result = retriever.run("hello")

        assert result["documents"][0].id == "1"
        document_store._bm25_retrieval.assert_called_once_with(
            query="hello",
            filters={"field": "meta.category", "operator": "==", "value": "news"},
            top_k=5,
            ranking="bm25",
        )

    def test_run_overrides_with_run_kwargs(self):
        document_store = _document_store()
        document_store._bm25_retrieval = Mock(return_value=[])

        retriever = VespaKeywordRetriever(
            document_store=document_store,
            filters={"field": "meta.category", "operator": "==", "value": "news"},
            top_k=10,
        )
        retriever.run(
            "hello",
            filters={"field": "meta.author", "operator": "==", "value": "deepset"},
            top_k=3,
        )

        _, kwargs = document_store._bm25_retrieval.call_args
        assert kwargs["filters"] == {"field": "meta.author", "operator": "==", "value": "deepset"}
        assert kwargs["top_k"] == 3

    def test_run_defaults_to_bm25_ranking(self):
        document_store = _document_store()
        document_store._bm25_retrieval = Mock(return_value=[])

        VespaKeywordRetriever(document_store=document_store).run("hello")

        _, kwargs = document_store._bm25_retrieval.call_args
        assert kwargs["ranking"] == "bm25"


class TestVespaEmbeddingRetriever:
    def test_init_default(self):
        document_store = _document_store()
        retriever = VespaEmbeddingRetriever(document_store=document_store)
        assert retriever.document_store is document_store
        assert retriever.filters is None
        assert retriever.top_k == 10
        assert retriever.ranking == "semantic"
        assert retriever.query_tensor_name == "query_embedding"
        assert retriever.target_hits is None

    def test_init(self):
        document_store = _document_store()
        retriever = VespaEmbeddingRetriever(
            document_store=document_store,
            filters={"field": "meta.category", "operator": "==", "value": "news"},
            top_k=7,
            ranking="semantic",
            query_tensor_name="q",
            target_hits=40,
        )
        assert retriever.filters == {"field": "meta.category", "operator": "==", "value": "news"}
        assert retriever.top_k == 7
        assert retriever.ranking == "semantic"
        assert retriever.query_tensor_name == "q"
        assert retriever.target_hits == 40

    def test_init_rejects_invalid_document_store(self):
        with pytest.raises(ValueError, match="VespaDocumentStore"):
            VespaEmbeddingRetriever(document_store="not a store")  # type:ignore[arg-type]

    def test_to_dict(self):
        retriever = VespaEmbeddingRetriever(
            document_store=_document_store(),
            filters={"field": "meta.category", "operator": "==", "value": "news"},
            top_k=7,
            ranking="semantic",
            query_tensor_name="query_embedding",
            target_hits=40,
        )

        assert component_to_dict(retriever, "retriever") == {
            "type": EMBEDDING_RETRIEVER_TYPE,
            "init_parameters": {
                "document_store": _doc_store_dict(),
                "filters": {"field": "meta.category", "operator": "==", "value": "news"},
                "top_k": 7,
                "ranking": "semantic",
                "query_tensor_name": "query_embedding",
                "target_hits": 40,
            },
        }

    def test_from_dict(self):
        data = {
            "type": EMBEDDING_RETRIEVER_TYPE,
            "init_parameters": {
                "document_store": _doc_store_dict(),
                "filters": {"field": "meta.category", "operator": "==", "value": "news"},
                "top_k": 7,
                "ranking": "semantic",
                "query_tensor_name": "query_embedding",
                "target_hits": 40,
            },
        }
        retriever = component_from_dict(VespaEmbeddingRetriever, data, "retriever")
        assert isinstance(retriever.document_store, VespaDocumentStore)
        assert retriever.filters == {"field": "meta.category", "operator": "==", "value": "news"}
        assert retriever.top_k == 7
        assert retriever.ranking == "semantic"
        assert retriever.query_tensor_name == "query_embedding"
        assert retriever.target_hits == 40

    def test_run_forwards_init_arguments(self):
        document_store = _document_store()
        document_store._embedding_retrieval = Mock(return_value=[Document(id="1", content="hello")])

        retriever = VespaEmbeddingRetriever(
            document_store=document_store,
            filters={"field": "meta.category", "operator": "==", "value": "news"},
            top_k=5,
            ranking="semantic",
            query_tensor_name="q",
            target_hits=25,
        )
        result = retriever.run([0.1, 0.2, 0.3])

        assert result["documents"][0].id == "1"
        document_store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            filters={"field": "meta.category", "operator": "==", "value": "news"},
            top_k=5,
            ranking="semantic",
            query_tensor_name="q",
            target_hits=25,
        )

    def test_run_overrides_with_run_kwargs(self):
        document_store = _document_store()
        document_store._embedding_retrieval = Mock(return_value=[])

        retriever = VespaEmbeddingRetriever(
            document_store=document_store,
            filters={"field": "meta.category", "operator": "==", "value": "news"},
            top_k=10,
        )
        retriever.run(
            [0.1, 0.2, 0.3],
            filters={"field": "meta.author", "operator": "==", "value": "deepset"},
            top_k=3,
        )

        _, kwargs = document_store._embedding_retrieval.call_args
        assert kwargs["filters"] == {"field": "meta.author", "operator": "==", "value": "deepset"}
        assert kwargs["top_k"] == 3

    def test_run_defaults_to_semantic_ranking(self):
        document_store = _document_store()
        document_store._embedding_retrieval = Mock(return_value=[])

        VespaEmbeddingRetriever(document_store=document_store).run([0.1, 0.2, 0.3])

        _, kwargs = document_store._embedding_retrieval.call_args
        assert kwargs["ranking"] == "semantic"


def _wait_until_documents_are_visible(document_store, expected_count: int) -> None:
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        if document_store.count_documents() == expected_count:
            return
        time.sleep(0.5)
    msg = f"Expected {expected_count} documents to become visible in Vespa"
    raise AssertionError(msg)


@pytest.mark.integration
class TestVespaRetrieversIntegration:
    def test_keyword_and_embedding_retrieval(self, document_store):
        written = document_store.write_documents(
            [
                Document(
                    id="1",
                    content="Haystack integrates with Vespa for search.",
                    embedding=[1.0, 0.0, 0.0],
                    meta={"category": "docs", "author": "deepset"},
                ),
                Document(
                    id="2",
                    content="Vespa supports lexical and vector retrieval.",
                    embedding=[0.0, 1.0, 0.0],
                    meta={"category": "docs", "author": "vespa"},
                ),
                Document(
                    id="3",
                    content="This note is about something else entirely.",
                    embedding=[0.0, 0.0, 1.0],
                    meta={"category": "misc", "author": "someone"},
                ),
            ]
        )

        assert written == 3
        _wait_until_documents_are_visible(document_store, 3)

        keyword_retriever = VespaKeywordRetriever(
            document_store=document_store,
            top_k=2,
            filters={"field": "meta.category", "operator": "==", "value": "docs"},
        )
        keyword_result = keyword_retriever.run(query="vector retrieval")

        assert keyword_result["documents"]
        assert keyword_result["documents"][0].id == "2"

        embedding_retriever = VespaEmbeddingRetriever(document_store=document_store, top_k=1)
        embedding_result = embedding_retriever.run(query_embedding=[1.0, 0.0, 0.0])

        assert embedding_result["documents"]
        assert embedding_result["documents"][0].id == "1"
