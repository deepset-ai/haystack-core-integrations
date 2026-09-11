import asyncio

import pytest
from haystack import Document, Pipeline, component
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.evaluation import RetrievalEvalCase, RetrievalHarnessEvaluator

QUESTION = "What is CRISPR used for?"


@component
class Expander:
    """A stage that rewrites the question, so a pipeline has something before its retriever."""

    def __init__(self, expansions: int = 1) -> None:
        self.expansions = expansions

    @component.output_types(queries=list[str])
    def run(self, query: str):
        return {"queries": [query, *(f"{query} v{index}" for index in range(self.expansions))]}


@component
class MultiRetriever:
    """Pools one retrieval per query and deduplicates, the way a multi-query retriever does."""

    def __init__(self, store, top_k: int = 5) -> None:
        self.retriever = InMemoryBM25Retriever(document_store=store, top_k=top_k)

    @component.output_types(documents=list[Document])
    def run(self, queries: list[str]):
        found: dict[str, Document] = {}
        for query in queries:
            for document in self.retriever.run(query=query)["documents"]:
                found[document.id] = document
        return {"documents": list(found.values())}


@pytest.fixture
def store():
    documents = [Document(content=f"chunk {index}: CRISPR gene editing corrects blindness") for index in range(10)]
    store = InMemoryDocumentStore()
    store.write_documents(documents)
    return store


@pytest.fixture
def wanted(store):
    """The document every eval case below expects to be retrieved."""
    return store.filter_documents()[0]


def retrieval_pipeline(store, top_k=5, expansions=1):
    pipeline = Pipeline()
    pipeline.add_component("expander", Expander(expansions=expansions))
    pipeline.add_component("retriever", MultiRetriever(store, top_k=top_k))
    pipeline.connect("expander.queries", "retriever.queries")
    return pipeline


def eval_case(wanted, question=QUESTION, **overrides):
    return RetrievalEvalCase(question=question, evidence={wanted.id: "gene editing"}, **overrides)


def test_a_pipeline_is_driven_by_socket_rather_than_by_component_name(store, wanted):
    """An optimizer may rename or replace any component, so nothing here may depend on what they are called."""
    renamed = Pipeline()
    renamed.add_component("rewrite_step", Expander())
    renamed.add_component("search_step", MultiRetriever(store))
    renamed.connect("rewrite_step.queries", "search_step.queries")

    metrics = RetrievalHarnessEvaluator().evaluate(target=renamed, eval_cases=[eval_case(wanted)])

    assert metrics.quality == 1.0
    assert set(metrics.details["stage_output_sizes"]) == {"rewrite_step", "search_step"}


def test_a_pipeline_the_harness_cannot_read_is_rejected_before_it_is_measured(store):
    """Validation is what turns an undrivable rewiring into an error the optimizer can repair."""
    two_exits = retrieval_pipeline(store)
    two_exits.add_component("second", MultiRetriever(store))
    two_exits.connect("expander.queries", "second.queries")

    with pytest.raises(ValueError, match="exactly one unconnected 'documents' output"):
        RetrievalHarnessEvaluator().validate(target=two_exits)

    no_query = Pipeline()
    no_query.add_component("retriever", MultiRetriever(store))
    with pytest.raises(ValueError, match="at least one unconnected 'query' input"):
        RetrievalHarnessEvaluator().validate(target=no_query)


def test_only_documents_above_the_cutoff_are_scored(store, wanted):
    """`k` is what makes the pipeline answerable for what it ranked highest, not for how much it returned."""
    pipeline = retrieval_pipeline(store, top_k=10)

    deep = RetrievalHarnessEvaluator(k=10).evaluate(target=pipeline, eval_cases=[eval_case(wanted)])
    shallow = RetrievalHarnessEvaluator(k=1).evaluate(target=pipeline, eval_cases=[eval_case(wanted)])

    assert deep.details["eval_cases"][0]["recall_at_k"] == 1.0
    assert shallow.details["eval_cases"][0]["retrieved"] == deep.details["eval_cases"][0]["retrieved"]
    # Scored one deep, precision is either 1 or 0, and recall follows whichever document ranked first.
    assert shallow.details["eval_cases"][0]["precision_at_k"] in (0.0, 1.0)


def test_a_recall_floor_is_what_makes_an_eval_case_fail(store):
    missing = RetrievalEvalCase(question=QUESTION, evidence={"never retrieved": "x"})

    metrics = RetrievalHarnessEvaluator().evaluate(target=retrieval_pipeline(store), eval_cases=[missing])

    assert metrics.quality == 0.0
    assert metrics.details["eval_cases"][0]["failures"] == ["recall_below_1"]
    assert metrics.details["eval_cases"][0]["missed_document_ids"] == ["never retrieved"]


def test_every_stage_reports_how_much_it_emitted(store, wanted):
    """A pooled candidate set has a size no configuration value states, so only measurement reports it."""
    pipeline = retrieval_pipeline(store, top_k=3, expansions=2)

    metrics = RetrievalHarnessEvaluator().evaluate(target=pipeline, eval_cases=[eval_case(wanted)])

    stages = metrics.details["stage_output_sizes"]
    assert stages["expander"]["queries"] == {"min": 3, "median": 3, "max": 3}
    # Three queries at top_k 3 could reach nine documents; deduplication is why it does not.
    assert stages["retriever"]["documents"]["max"] <= 9


def test_the_bounds_report_a_stage_that_only_sometimes_collapses(store, wanted):
    """A median alone would hide it, and a mean would report a size no eval case produced."""

    @component
    class Flaky:
        @component.output_types(queries=list[str])
        def run(self, query: str):
            return {"queries": [query] if "rare" in query else [query, f"{query} v1", f"{query} v2"]}

    pipeline = Pipeline()
    pipeline.add_component("expander", Flaky())
    pipeline.add_component("retriever", MultiRetriever(store))
    pipeline.connect("expander.queries", "retriever.queries")
    cases = [eval_case(wanted, question=f"{QUESTION} {index}") for index in range(3)]
    cases.append(eval_case(wanted, question="rare question"))

    metrics = RetrievalHarnessEvaluator().evaluate(target=pipeline, eval_cases=cases)

    assert metrics.details["stage_output_sizes"]["expander"]["queries"] == {"min": 1, "median": 3, "max": 3}


def test_measuring_nothing_is_rejected_rather_than_dividing_by_zero(store):
    with pytest.raises(ValueError, match="no eval cases to score"):
        RetrievalHarnessEvaluator().evaluate(target=retrieval_pipeline(store), eval_cases=[])


def test_concurrency_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        RetrievalHarnessEvaluator(max_concurrent_eval_cases=0)


def test_eval_cases_measured_concurrently_are_reported_in_order(store, wanted):
    """Concurrency must change how long an evaluation takes, not what it measures."""
    cases = [eval_case(wanted, question=f"{QUESTION} {index}") for index in range(4)]

    sequential = RetrievalHarnessEvaluator(max_concurrent_eval_cases=1).evaluate(
        target=retrieval_pipeline(store), eval_cases=cases
    )
    concurrent = RetrievalHarnessEvaluator(max_concurrent_eval_cases=4).evaluate(
        target=retrieval_pipeline(store), eval_cases=cases
    )

    questions = [entry["question"] for entry in concurrent.details["eval_cases"]]
    assert questions == [entry["question"] for entry in sequential.details["eval_cases"]]
    assert concurrent.quality == sequential.quality


@pytest.mark.asyncio
async def test_evaluating_from_inside_a_running_loop_measures_what_the_sync_call_does(store, wanted):
    """The sync entry point cannot run under a loop, so an async caller has to reach the same work directly."""
    cases = [eval_case(wanted)]

    awaited = await RetrievalHarnessEvaluator().evaluate_async(target=retrieval_pipeline(store), eval_cases=cases)
    blocking = await asyncio.to_thread(
        RetrievalHarnessEvaluator().evaluate, target=retrieval_pipeline(store), eval_cases=cases
    )

    assert awaited.quality == blocking.quality == 1.0
    assert awaited.details["stage_output_sizes"] == blocking.details["stage_output_sizes"]
