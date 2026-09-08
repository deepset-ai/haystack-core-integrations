import pytest
from haystack import Document, Pipeline
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.query import QueryExpander
from haystack.components.rankers import LLMRanker
from haystack.components.retrievers import InMemoryBM25Retriever, MultiQueryTextRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from retrieval import RetrievalEvaluationCase, RetrievalHarnessEvaluator
from retrieval.harness_evaluator import (
    documents_exit_point,
    query_entry_points,
    query_reporters,
)

from haystack_integrations.agent_pack.dataclasses import RunRecord
from haystack_integrations.agent_pack.optimization import dump_pipeline, load_pipeline

QUESTION = "What is CRISPR used for?"
EXPANSION = '{"queries": ["CRISPR gene editing", "hereditary blindness treatment"]}'


@pytest.fixture
def store():
    documents = [
        Document(content="CRISPR gene editing can correct hereditary blindness mutations."),
        Document(content="A quantum computer demonstrated error-corrected logical qubits."),
    ]
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)
    return document_store


def retrieval_pipeline(store, *, top_k=2, expansions=2, model="expander"):
    pipeline = Pipeline()
    pipeline.add_component(
        "expander",
        QueryExpander(
            chat_generator=MockChatGenerator(
                EXPANSION, model=model, meta={"usage": {"prompt_tokens": 30, "completion_tokens": 5}}
            ),
            n_expansions=expansions,
        ),
    )
    pipeline.add_component(
        "retriever", MultiQueryTextRetriever(retriever=InMemoryBM25Retriever(document_store=store, top_k=top_k))
    )
    pipeline.connect("expander.queries", "retriever.queries")
    return pipeline


def runs():
    return [RunRecord(run_id="case-0", inputs={"query": QUESTION}, outputs={})]


def case(store, **overrides):
    expected = frozenset({store.filter_documents()[0].id})
    return RetrievalEvaluationCase(question=QUESTION, expected_document_ids=expected, **overrides)


def test_sockets_are_found_by_name_not_by_component_name(store):
    pipeline = retrieval_pipeline(store)
    assert query_entry_points(pipeline=pipeline) == {"expander"}
    assert documents_exit_point(pipeline=pipeline) == "retriever"
    assert query_reporters(pipeline=pipeline) == {"expander"}


def test_a_renamed_and_reranked_pipeline_is_still_drivable(store):
    """The optimizer may rename components and append a ranker; the harness must follow the sockets."""
    pipeline = Pipeline()
    pipeline.add_component("rewrite", QueryExpander(chat_generator=MockChatGenerator(EXPANSION), n_expansions=2))
    pipeline.add_component(
        "search", MultiQueryTextRetriever(retriever=InMemoryBM25Retriever(document_store=store, top_k=2))
    )
    pipeline.add_component(
        "rerank", LLMRanker(chat_generator=MockChatGenerator('{"documents": [{"index": 1}]}'), top_k=1)
    )
    pipeline.connect("rewrite.queries", "search.queries")
    pipeline.connect("search.documents", "rerank.documents")

    assert query_entry_points(pipeline=pipeline) == {"rewrite", "rerank"}
    assert documents_exit_point(pipeline=pipeline) == "rerank"
    RetrievalHarnessEvaluator(cases=[case(store)]).validate_pipeline(pipeline)


def test_validation_rejects_a_pipeline_the_harness_cannot_read(store):
    pipeline = Pipeline()
    pipeline.add_component("expander", QueryExpander(chat_generator=MockChatGenerator(EXPANSION)))
    evaluator = RetrievalHarnessEvaluator(cases=[case(store)])
    with pytest.raises(ValueError, match="documents"):
        evaluator.validate_pipeline(pipeline)


def test_recall_precision_and_issued_queries_are_measured(store):
    evaluator = RetrievalHarnessEvaluator(cases=[case(store)])
    metrics = evaluator.evaluate(target=retrieval_pipeline(store), reference_runs=runs())

    assert metrics.quality == 1.0
    assert metrics.details["mean_recall"] == 1.0
    assert metrics.details["cases"][0]["passed"] is True
    # The expander's own queries are read off a connected edge, which a pipeline result omits by default.
    assert metrics.details["cases"][0]["queries"]
    assert metrics.model_usage["expander"].input_tokens == 30


def test_a_missed_document_is_named_so_a_failure_can_be_diagnosed(store):
    absent = RetrievalEvaluationCase(question=QUESTION, expected_document_ids=frozenset({"not-in-the-store"}))
    metrics = RetrievalHarnessEvaluator(cases=[absent]).evaluate(
        target=retrieval_pipeline(store), reference_runs=runs()
    )

    assert metrics.quality == 0.0
    assert metrics.details["cases"][0]["failures"] == ["recall_below_1"]
    assert metrics.details["cases"][0]["missed_document_ids"] == ["not-in-the-store"]


def test_expanding_without_limit_is_charged_against_a_query_budget(store):
    metrics = RetrievalHarnessEvaluator(cases=[case(store, max_queries=1)]).evaluate(
        target=retrieval_pipeline(store), reference_runs=runs()
    )

    assert metrics.quality == 0.0
    assert "queries_over_budget:3" in metrics.details["cases"][0]["failures"]


def test_the_pipeline_round_trips_through_yaml_and_still_measures(store):
    reloaded = load_pipeline(dump_pipeline(retrieval_pipeline(store)))
    metrics = RetrievalHarnessEvaluator(cases=[case(store)]).evaluate(target=reloaded, reference_runs=runs())

    assert metrics.quality == 1.0


def test_an_unlabelled_question_is_refused_rather_than_scored(store):
    other = [RunRecord(run_id="case-0", inputs={"query": "something else"}, outputs={})]
    with pytest.raises(ValueError, match="No labelled retrieval case"):
        RetrievalHarnessEvaluator(cases=[case(store)]).evaluate(target=retrieval_pipeline(store), reference_runs=other)


def test_returning_most_of_the_corpus_earns_nothing_for_the_documents_past_the_limit(store):
    """
    Recall on its own is maximized by returning everything, so only the first `max_retrieved` are scored. That
    removes the degenerate optimum without turning one document too many into a total loss.
    """
    both = frozenset(document.id for document in store.filter_documents())
    wide = RetrievalEvaluationCase(question=QUESTION, expected_document_ids=both, max_retrieved=1)
    metrics = RetrievalHarnessEvaluator(cases=[wide]).evaluate(
        target=retrieval_pipeline(store, top_k=2), reference_runs=runs()
    )

    scored_case = metrics.details["cases"][0]
    assert any(f.startswith("retrieved_over_budget:") for f in scored_case["failures"])
    # Both expected documents come back, but the second sits past the limit and earns nothing.
    assert scored_case["retrieved"] == 2
    assert metrics.quality == 0.5


def test_ranking_a_wide_candidate_set_down_satisfies_the_document_budget(store):
    """The cap is on what the pipeline returns, so widening then reranking complies while widening alone does not."""
    wide = retrieval_pipeline(store, top_k=2)
    wide_metrics = RetrievalHarnessEvaluator(cases=[case(store, max_retrieved=1)]).evaluate(
        target=wide, reference_runs=runs()
    )

    ranked = Pipeline()
    ranked.add_component("expander", QueryExpander(chat_generator=MockChatGenerator(EXPANSION), n_expansions=2))
    ranked.add_component(
        "retriever", MultiQueryTextRetriever(retriever=InMemoryBM25Retriever(document_store=store, top_k=2))
    )
    ranked.add_component(
        "ranker", LLMRanker(chat_generator=MockChatGenerator('{"documents": [{"index": 1}]}'), top_k=1)
    )
    ranked.connect("expander.queries", "retriever.queries")
    ranked.connect("retriever.documents", "ranker.documents")
    ranked_metrics = RetrievalHarnessEvaluator(cases=[case(store, max_retrieved=1)]).evaluate(
        target=ranked, reference_runs=runs()
    )

    assert wide_metrics.details["cases"][0]["retrieved"] > 1
    assert ranked_metrics.details["cases"][0]["retrieved"] == 1
    assert not any(f.startswith("retrieved_over_budget:") for f in ranked_metrics.details["cases"][0]["failures"])


def test_a_component_that_degrades_instead_of_failing_is_reported(store):
    """The exact case that scored as an improvement while its ranker was returning documents unranked."""

    def explode(*_args, **_kwargs):
        message = "Error code: 400 - temperature does not support 0.0 with this model"
        raise RuntimeError(message)

    pipeline = Pipeline()
    pipeline.add_component("expander", QueryExpander(chat_generator=MockChatGenerator(EXPANSION), n_expansions=2))
    pipeline.add_component(
        "retriever", MultiQueryTextRetriever(retriever=InMemoryBM25Retriever(document_store=store, top_k=2))
    )
    # raise_on_failure is False by default, which is what makes the failure invisible to the score.
    pipeline.add_component("ranker", LLMRanker(chat_generator=MockChatGenerator(response_fn=explode), top_k=1))
    pipeline.connect("expander.queries", "retriever.queries")
    pipeline.connect("retriever.documents", "ranker.documents")

    metrics = RetrievalHarnessEvaluator(cases=[case(store)]).evaluate(target=pipeline, reference_runs=runs())

    warnings = metrics.details["warnings"]
    assert any("LLMRanker failed during chat generation" in entry["message"] for entry in warnings)
    assert any("temperature does not support 0.0" in entry["message"] for entry in warnings)
    # The ranker returned its input untouched, so the run looks ordinary to every other measurement.
    assert metrics.details["cases"][0]["retrieved"] > 1


def test_partial_recall_scores_partially_but_a_broken_budget_scores_nothing(store):
    """Recall over a few documents moves in halves and thirds; a threshold would report those steps as no change."""
    documents = store.filter_documents()
    both = frozenset(d.id for d in documents)

    half = RetrievalEvaluationCase(question=QUESTION, expected_document_ids=both)
    scored = RetrievalHarnessEvaluator(cases=[half]).evaluate(
        target=retrieval_pipeline(store, top_k=1), reference_runs=runs()
    )
    assert scored.details["cases"][0]["passed"] is False
    assert 0.0 < scored.quality < 1.0
    assert scored.quality == scored.details["mean_recall"]

    # Overshooting the document limit costs the documents past it and nothing else, so a ranker that returns one
    # too many measures as having made a small mistake rather than as having found nothing.
    over = RetrievalEvaluationCase(question=QUESTION, expected_document_ids=both, max_retrieved=1)
    busted = RetrievalHarnessEvaluator(cases=[over]).evaluate(
        target=retrieval_pipeline(store, top_k=2), reference_runs=runs()
    )
    assert any(f.startswith("retrieved_over_budget") for f in busted.details["cases"][0]["failures"])
    assert busted.quality == busted.details["cases"][0]["recall"] > 0.0

    # Issuing too many queries has no equivalent of ignoring the excess, so it still scores nothing.
    talkative = RetrievalEvaluationCase(question=QUESTION, expected_document_ids=both, max_queries=1)
    overspent = RetrievalHarnessEvaluator(cases=[talkative]).evaluate(
        target=retrieval_pipeline(store, top_k=2), reference_runs=runs()
    )
    assert any(f.startswith("queries_over_budget") for f in overspent.details["cases"][0]["failures"])
    assert overspent.details["cases"][0]["recall"] > 0.0
    assert overspent.quality == 0.0
