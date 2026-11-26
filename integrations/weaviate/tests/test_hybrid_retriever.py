# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, create_autospec

import pytest
from haystack import Document, Pipeline
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.weaviate import WeaviateHybridRetriever
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["dense", "multivector", "combined"])
def test_named_vector_hybrid_retrieval(multivector_document_store, mode):
    store = multivector_document_store
    store.write_documents(
        [
            Document(
                id="match",
                content="matching document",
                embedding=[1.0, 0.0],
                multivector_embedding=[[1.0, 0.0]],
                meta={"category": "keep"},
            ),
            Document(
                id="other",
                content="other document",
                embedding=[0.0, 1.0],
                multivector_embedding=[[0.0, 1.0]],
                meta={"category": "other"},
            ),
        ]
    )
    inputs = {}
    if mode in ("dense", "combined"):
        inputs["query_embedding"] = [1.0, 0.0]
    if mode in ("multivector", "combined"):
        inputs["query_multivector_embedding"] = [[1.0, 0.0]]
    retriever = WeaviateHybridRetriever(
        document_store=store, top_k=1, vector_weights={"default": 0.2, "multi_vector": 0.8}
    )
    docs = retriever.run(query="matching", **inputs)["documents"]
    assert [doc.id for doc in docs] == ["match"]
    assert docs[0].embedding == [1.0, 0.0]
    assert docs[0].multivector_embedding == [[1.0, 0.0]]
    assert docs[0].score is not None
    docs = retriever.run(
        query="document",
        **inputs,
        filters={"field": "meta.category", "operator": "==", "value": "other"},
    )["documents"]
    assert [doc.id for doc in docs] == ["other"]


@pytest.mark.parametrize("override", [False, True])
def test_combined_inputs(override):
    store = create_autospec(WeaviateDocumentStore, instance=True)
    store._hybrid_retrieval.return_value = [Document(content="result")]
    initial = {"default": 0.2, "multi_vector": 0.8}
    runtime = {"default": 0.8, "multi_vector": 0.2}
    retriever = WeaviateHybridRetriever(document_store=store, vector_weights=initial)
    result = retriever.run(
        query="test",
        query_embedding=[1.0, 0.0],
        query_multivector_embedding=[[0.0, 1.0]],
        vector_weights=runtime if override else None,
    )
    store._hybrid_retrieval.assert_called_once_with(
        query="test",
        query_embedding=[1.0, 0.0],
        query_multivector_embedding=[[0.0, 1.0]],
        vector_weights=runtime if override else initial,
        filters={},
        top_k=10,
        alpha=0.7,
        max_vector_distance=None,
    )
    assert result == {"documents": store._hybrid_retrieval.return_value}


def test_requires_embedding():
    store = create_autospec(WeaviateDocumentStore, instance=True)
    with pytest.raises(ValueError, match="Provide query_embedding"):
        WeaviateHybridRetriever(document_store=store).run(query="test")
    store._hybrid_retrieval.assert_not_called()


def test_close():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    retriever = WeaviateHybridRetriever(document_store=mock_document_store)

    retriever.close()

    mock_document_store.close.assert_called_once_with()
    assert retriever._document_store is mock_document_store


def test_init_default():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    retriever = WeaviateHybridRetriever(document_store=mock_document_store)
    assert retriever._document_store == mock_document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._alpha == 0.7
    assert retriever._max_vector_distance is None
    assert retriever._filter_policy == FilterPolicy.REPLACE

    retriever = WeaviateHybridRetriever(document_store=mock_document_store, filter_policy="replace")
    assert retriever._filter_policy == FilterPolicy.REPLACE

    with pytest.raises(ValueError):
        WeaviateHybridRetriever(document_store=mock_document_store, filter_policy="keep_all")


def test_to_dict():
    store = WeaviateDocumentStore()
    retriever = WeaviateHybridRetriever(document_store=store)
    assert retriever.to_dict() == {
        "type": "haystack_integrations.components.retrievers.weaviate.hybrid_retriever.WeaviateHybridRetriever",
        "init_parameters": {
            "filters": {},
            "top_k": 10,
            "alpha": 0.7,
            "max_vector_distance": None,
            "vector_weights": None,
            "filter_policy": "replace",
            "document_store": store.to_dict(),
        },
    }


def test_run_with_runtime_filters():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    mock_document_store._hybrid_retrieval.return_value = [Mock(content="Filtered document", score=0.8)]

    # Test with REPLACE policy (default)
    retriever = WeaviateHybridRetriever(document_store=mock_document_store, filters={"init": "filter"})
    retriever.run(query="test query", query_embedding=[0.1, 0.2, 0.3], filters={"runtime": "filter"})

    mock_document_store._hybrid_retrieval.assert_called_once_with(
        query="test query",
        query_embedding=[0.1, 0.2, 0.3],
        filters={"runtime": "filter"},
        top_k=10,
        alpha=0.7,
        max_vector_distance=None,
        query_multivector_embedding=None,
        vector_weights=None,
    )


def test_run_with_init_and_runtime_parameters():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    mock_document_store._hybrid_retrieval.return_value = [Mock(content="Init and runtime params document", score=0.5)]

    retriever = WeaviateHybridRetriever(
        document_store=mock_document_store, top_k=20, alpha=0.3, max_vector_distance=0.9
    )
    retriever.run(query="test query", query_embedding=[0.1, 0.2, 0.3], top_k=15, alpha=0.6, max_vector_distance=0.8)

    # Runtime parameters should override init parameters
    mock_document_store._hybrid_retrieval.assert_called_once_with(
        query="test query",
        query_embedding=[0.1, 0.2, 0.3],
        filters={},
        top_k=15,
        alpha=0.6,
        max_vector_distance=0.8,
        query_multivector_embedding=None,
        vector_weights=None,
    )


def test_run_empty_query():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    mock_document_store._hybrid_retrieval.return_value = []

    retriever = WeaviateHybridRetriever(document_store=mock_document_store)
    result = retriever.run(query="", query_embedding=[0.1, 0.2, 0.3])

    assert "documents" in result
    assert len(result["documents"]) == 0
    mock_document_store._hybrid_retrieval.assert_called_once_with(
        query="",
        query_embedding=[0.1, 0.2, 0.3],
        filters={},
        top_k=10,
        alpha=0.7,
        max_vector_distance=None,
        query_multivector_embedding=None,
        vector_weights=None,
    )


def test_from_dict_without_filter_policy_or_vector_weights():
    retriever = WeaviateHybridRetriever(document_store=WeaviateDocumentStore())
    data = retriever.to_dict()
    del data["init_parameters"]["filter_policy"]
    del data["init_parameters"]["vector_weights"]
    assert WeaviateHybridRetriever.from_dict(data).to_dict() == retriever.to_dict()


@pytest.mark.parametrize("override", [False, True])
def test_zero_thresholds(override):
    store = create_autospec(WeaviateDocumentStore, instance=True)
    thresholds = {"alpha": 0.0, "max_vector_distance": 0.0}
    retriever = WeaviateHybridRetriever(document_store=store, **({} if override else thresholds))
    retriever.run(query="q", query_embedding=[0.1, 0.2], **(thresholds if override else {}))
    kwargs = store._hybrid_retrieval.call_args.kwargs
    assert kwargs["alpha"] == 0.0
    assert kwargs["max_vector_distance"] == 0.0


def test_init_with_invalid_alpha():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    with pytest.raises(ValueError, match=r"alpha \(-0.1\) must be in the range \[0.0, 1.0\]"):
        WeaviateHybridRetriever(document_store=mock_document_store, alpha=-0.1)
    with pytest.raises(ValueError, match=r"alpha \(1.5\) must be in the range \[0.0, 1.0\]"):
        WeaviateHybridRetriever(document_store=mock_document_store, alpha=1.5)


def test_run_with_invalid_alpha():
    mock_document_store = create_autospec(WeaviateDocumentStore, instance=True)
    retriever = WeaviateHybridRetriever(document_store=mock_document_store)
    with pytest.raises(ValueError, match=r"alpha \(-0.1\) must be in the range \[0.0, 1.0\]"):
        retriever.run(query="q", query_embedding=[0.1], alpha=-0.1)
    with pytest.raises(ValueError, match=r"alpha \(1.5\) must be in the range \[0.0, 1.0\]"):
        retriever.run(query="q", query_embedding=[0.1], alpha=1.5)


def test_pipeline_serde():
    """A pipeline holding the retriever must survive a dumps/loads round trip, nested store included."""
    document_store = WeaviateDocumentStore(url="http://localhost:8080")
    pipeline = Pipeline()
    retriever = WeaviateHybridRetriever(
        document_store=document_store,
        top_k=3,
        alpha=0.3,
        max_vector_distance=0.6,
        filters={"field": "content", "operator": "==", "value": "test"},
        filter_policy=FilterPolicy.MERGE,
        vector_weights={"default": 0.2, "multi_vector": 0.8},
    )
    pipeline.add_component("retriever", retriever)
    restored = Pipeline.loads(pipeline.dumps()).get_component("retriever")
    assert restored.to_dict() == retriever.to_dict()
    assert restored._top_k == 3
    assert restored._alpha == 0.3
    assert restored._max_vector_distance == 0.6
    assert restored._filters == retriever._filters
    assert restored._filter_policy == FilterPolicy.MERGE
    assert restored._vector_weights == {"default": 0.2, "multi_vector": 0.8}
