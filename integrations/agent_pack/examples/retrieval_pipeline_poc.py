# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Optimize a one-shot retrieval Pipeline, rather than an Agent, against the same labelled evaluation set.

This is the cheap counterpart to `harness_optimization_poc`. The configuration under optimization is a plain
Haystack Pipeline — a `QueryExpander` that turns one question into several, feeding a `MultiQueryTextRetriever`
that runs them all against the store:

    QueryExpander.queries -> MultiQueryTextRetriever.queries -> documents

Nothing generates an answer. The MultiHopRAG cases already name the documents an answer needs, so retrieval is
scored directly as recall and precision over those IDs. That is what makes this affordable: one case costs a
single query-expansion call, against the five to twenty model calls an Agent loop spends. At that point the
optimizer's own turns, not the measurements, are most of what an experiment costs.

It is also a harness where retrieval is the whole configuration. There is no system prompt to tune and no tool
budget to trim, so an optimizer that wants to improve quality has to change the expansion or the retrieval path.

Run from `integrations/agent_pack` with `OPENAI_API_KEY` set. The corpus requires `datasets`:

    hatch run test:python examples/retrieval_pipeline_poc.py
    hatch run test:python examples/retrieval_pipeline_poc.py --max-cases 5 --max-iterations 2
    hatch run test:python examples/retrieval_pipeline_poc.py --docs-mcp
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.query import QueryExpander
from haystack.components.retrievers import MultiQueryTextRetriever
from haystack.document_stores.types import DocumentStore
from multihop_rag import CORPUS_KEY, SPLIT_LENGTH, SPLIT_OVERLAP, build_cases, prepare_corpus
from util import build_retriever

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord
from haystack_integrations.agent_pack.local_run_store import LocalRunStore
from haystack_integrations.agent_pack.optimization import (
    ExperimentJournal,
    ExperimentResult,
    HarnessOptimizationExperiment,
    ModelPrice,
    ModelPriceCatalog,
    OptimizationObjectives,
    create_harness_optimizer_agent,
    create_haystack_documentation_mcp_toolset,
)
from haystack_integrations.agent_pack.retrieval import RetrievalEvaluationCase, RetrievalHarnessEvaluator

WORKSPACE = Path(".agent-pack-retrieval-poc")
EXPANDER_MODEL = "gpt-5.6-luna"

# USD prices per million tokens, used only to rank candidates against each other.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
}

# The reference is under-configured in the two places this pipeline has. One expansion of a multi-hop question
# still asks one thing, and a question whose evidence is spread over several articles needs the retriever to
# surface more than a couple of chunks. Neither limit is where the fix has to be: which of them matters, and
# whether the retrieval path should be reranked rather than widened, is what the experiment is for.
POOR_EXPANSIONS = 1
POOR_TOP_K = 2

RETRIEVAL_OPTIMIZER_GUIDANCE = """
The configuration is a one-shot retrieval pipeline, and retrieval is all of it. A question enters at the `query`
input, whatever the pipeline does with it must end at exactly one unconnected `documents` output, and that output
is scored as recall and precision against the documents the answer needed. Nothing writes an answer, so nothing is
gained by adding a generator.

Cases require evidence spread across several documents, and one query phrased for the whole question tends to
surface only the documents that share its wording. Expansion buys recall with model calls, and a wider candidate
set buys it with precision; the case reports recall and precision separately, and names the documents that were
missed, so the two are distinguishable.

Each case also limits how many documents the pipeline may return. The limit is on what comes out, not on what the
pipeline looks at, so past a certain point recall cannot be bought by widening: what is returned has to be the
right subset of whatever was considered.

Once that limit binds, the way past it is to stop treating those two things as the same. Retrieve a wide candidate
set, then rank it and return only the best of it: the limit applies to the ranked output, while the candidate set
behind it can be as wide as recall needs. Haystack ships ranker components for exactly this, placed between
retrieval and the pipeline's `documents` output. Find which ones this environment can import, confirm what one
serializes to and what its inputs are called, and remember that a ranker driven by a model needs a query as well
as the documents.

Merging the results of several queries is where this goes wrong quietly. A keyword retriever's score is a property
of the query that produced it, not a scale shared between queries, so pooling per-query results and sorting them
by score produces an order that means nothing, and keeping the best few of that order is close to keeping an
arbitrary few. Reciprocal rank fusion is the standard answer: it merges on each document's rank within its own
result list, which is comparable across queries, and it is what a joiner should be asked for when several queries
feed one output. Anything that trims a pooled result set has to settle this question one way or the other.

The retrieval path is part of the configuration and can be restructured, not only retuned. A keyword retriever
ranks by wording alone; a wider candidate set that is then reranked by something else is a different mechanism,
not a bigger version of the same one. Confirm what any component you introduce serializes to, and that this
environment can import it, before spending a measurement on it.
""".strip()


def build_reference_pipeline(store: DocumentStore, model: str) -> Pipeline:
    """
    Build the under-configured retrieval pipeline whose complete configuration will be optimized.

    :param store: The corpus to retrieve from.
    :param model: The model the query expander reasons with.
    :returns: The reference pipeline.
    """
    pipeline = Pipeline()
    pipeline.add_component(
        "expander",
        QueryExpander(chat_generator=OpenAIChatGenerator(model=model), n_expansions=POOR_EXPANSIONS),
    )
    pipeline.add_component(
        "retriever", MultiQueryTextRetriever(retriever=build_retriever(store=store, top_k=POOR_TOP_K))
    )
    pipeline.connect("expander.queries", "retriever.queries")
    return pipeline


def build_pricing() -> ModelPriceCatalog:
    """Build informational prices for the models this example knows about."""
    return ModelPriceCatalog(
        prices=[
            ModelPrice(model_id=model, input_cost_per_million=prices[0], output_cost_per_million=prices[1])
            for model, prices in MODEL_PRICES.items()
        ]
    )


def format_cost(cost: float | None) -> str:
    """Format a measured cost that may be unavailable for an optimizer-selected model."""
    return "unpriced" if cost is None else f"${cost:.6f}"


def report(result: ExperimentResult) -> None:
    """Print baseline, candidate, gate, and recommendation details."""
    baseline = result.baseline
    print("\n--- baseline (reference pipeline) ---")
    print(
        f"  quality={baseline.quality:.2f} cost={format_cost(cost=baseline.cost)} "
        f"latency={baseline.latency_ms:.0f}ms recall={baseline.details['mean_recall']:.2f} "
        f"queries={baseline.details['mean_queries']:.1f} retrieved={baseline.details['mean_retrieved']:.1f}"
    )

    print("\n--- candidates ---")
    for candidate in result.candidates:
        gates = result.gate_failures.get(candidate.candidate_id, ())
        if candidate.metrics is None:
            print(f"  {candidate.candidate_id} -> failed: {candidate.failure}")
            continue
        details = candidate.metrics.details
        print(
            f"  {candidate.candidate_id} -> quality={candidate.metrics.quality:.2f} "
            f"cost={format_cost(cost=candidate.metrics.cost)} recall={details['mean_recall']:.2f} "
            f"queries={details['mean_queries']:.1f} retrieved={details['mean_retrieved']:.1f}"
        )
        print(f"    gates: {'passed' if not gates else ', '.join(gates)}")

    print("\n--- what the search itself cost ---")
    usage = ", ".join(
        f"{model}: {tokens.input_tokens} in / {tokens.output_tokens} out"
        for model, tokens in result.optimizer_usage.items()
    )
    print(f"  optimizer usage: {usage or 'none recorded'}")
    print(f"  optimizer cost:  {format_cost(cost=result.optimizer_cost)} (input tokens charged at full price)")

    print("\n--- recommendation ---")
    if result.recommendation is None:
        print("  none: no candidate cleared every gate and improved on the reference")
        return
    recommendation = result.recommendation
    print(f"  configuration: {result.artifact_directory / 'recommended.yaml'}")
    print(f"  rationale: {recommendation.configuration.rationale}")
    print(f"  reasons: {', '.join(recommendation.reasons)}")
    print("  Nothing was deployed. Approving this recommendation is a separate, human decision.")


def parse_args() -> argparse.Namespace:
    """Parse the PoC command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", choices=("in_memory", "opensearch"), default="in_memory")
    parser.add_argument("--expander-model", default=EXPANDER_MODEL)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE, help="Directory for runs and YAML artifacts.")
    parser.add_argument("--config", type=Path, help="Optional editable YAML file; created from the reference.")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=20,
        help="Cases to evaluate. Quality is a fraction of these, so the smallest difference it can express is "
        "1/max-cases; keep it well above the effect worth detecting.",
    )
    parser.add_argument("--case-seed", type=int, default=0, help="Selects which cases are drawn from the dataset.")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=6,
        help="Per-case cap on issued queries. Without one, expanding without limit is the cheapest way to pass.",
    )
    parser.add_argument(
        "--max-retrieved",
        type=int,
        default=10,
        help="Per-case cap on documents the pipeline may return. Recall on its own is maximized by returning most "
        "of the corpus, and this harness generates no answer, so nothing downstream makes that expensive. The cap "
        "applies to the pipeline's output rather than to its candidate set, so widening and then ranking down "
        "still satisfies it.",
    )
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument(
        "--max-quality-loss",
        type=float,
        default=0.05,
        help="Maximum absolute pass-rate loss from the reference. Keep it at no less than one case, or single-case "
        "measurement noise gates out real improvements.",
    )
    parser.add_argument("--primary", choices=("cost", "latency", "quality"), default="quality")
    parser.add_argument("--max-concurrent-cases", type=int, default=6)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--optimizer-steps", type=int, default=24, help="Editing steps per optimizer turn.")
    parser.add_argument("--docs-mcp", action="store_true", help="Give the optimizer the Haystack documentation MCP.")
    parser.add_argument("--fresh", action="store_true", help="Remove saved runs and journals before starting.")
    return parser.parse_args()


def enable_progress_reporting() -> None:
    """Route library progress to stdout, line buffered, so a redirected run reports as it goes."""
    sys.stdout.reconfigure(line_buffering=True)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("  %(message)s"))
    progress = logging.getLogger("haystack_integrations.agent_pack")
    progress.handlers.clear()
    progress.addHandler(handler)
    progress.setLevel(logging.INFO)
    progress.propagate = False


def main() -> None:
    """Build the experiment and run it end to end."""
    arguments = parse_args()
    enable_progress_reporting()
    if not os.environ.get("OPENAI_API_KEY"):
        message = "OPENAI_API_KEY must be set to run this walkthrough."
        raise SystemExit(message)
    if arguments.max_cases < 1:
        message = "--max-cases must be at least 1."
        raise SystemExit(message)
    if arguments.primary == "latency" and arguments.max_concurrent_cases > 1:
        message = "Ranking by latency requires --max-concurrent-cases 1."
        raise SystemExit(message)
    if arguments.fresh and arguments.workspace.exists():
        shutil.rmtree(path=arguments.workspace)

    print("=== 1. set up corpus and evaluation set ===")
    store, chunks = prepare_corpus(backend=arguments.store)
    document_count = store.count_documents()
    print(f"  {CORPUS_KEY} on {arguments.store}: {document_count} chunks")

    labelled = build_cases(chunks=chunks, limit=arguments.max_cases, seed=arguments.case_seed)
    cases = [
        RetrievalEvaluationCase(
            question=case.question,
            expected_document_ids=case.expected_document_ids,
            max_queries=arguments.max_queries,
            max_retrieved=arguments.max_retrieved,
        )
        for case in labelled
    ]
    expected = sum(len(case.expected_document_ids) for case in cases)
    print(f"  cases: {len(cases)} labelled from evidence, expecting {expected} documents in total")

    reference = build_reference_pipeline(store=store, model=arguments.expander_model)
    print(
        f"  reference: n_expansions={POOR_EXPANSIONS} top_k={POOR_TOP_K} model={arguments.expander_model}; "
        f"budgets: {arguments.max_queries} queries and {arguments.max_retrieved} documents per case"
    )

    # A retrieval run replays only its question, so the store records that rather than a captured pipeline run:
    # unlike an Agent harness, there is no tool trace worth keeping and nothing about the reference's behaviour
    # that the measured baseline does not already report.
    print("\n=== 2. record the questions to replay ===")
    run_store = LocalRunStore(directory=arguments.workspace / "runs")
    run_store.clear()
    for index, case in enumerate(cases):
        run_store.add(record=AgentRunRecord(run_id=f"case-{index}", inputs={"query": case.question}, outputs={}))
    print(f"  recorded {len(cases)} questions")

    print("\n=== 3. optimization experiment ===")
    docs_toolset = create_haystack_documentation_mcp_toolset() if arguments.docs_mcp else None
    evaluator = RetrievalHarnessEvaluator(cases=cases, max_concurrent_cases=arguments.max_concurrent_cases)
    experiment = HarnessOptimizationExperiment(
        reference=reference,
        run_store=run_store,
        evaluator=evaluator,
        pricing=build_pricing(),
        objectives=OptimizationObjectives(
            min_quality=arguments.min_quality,
            max_quality_loss=arguments.max_quality_loss,
            primary=arguments.primary,
        ),
        journal=ExperimentJournal(directory=arguments.workspace / "journals"),
        optimizer_agent=create_harness_optimizer_agent(
            docs_toolset=docs_toolset,
            additional_instructions=RETRIEVAL_OPTIMIZER_GUIDANCE,
            max_agent_steps=arguments.optimizer_steps,
        ),
        max_iterations=arguments.max_iterations,
        config_path=arguments.config,
        configuration_key=f"{CORPUS_KEY}:{SPLIT_LENGTH}:{SPLIT_OVERLAP}:{document_count}",
    )
    result = experiment.run()
    print(f"  measurement context: {result.measurement_context}; run: {result.run_id}")

    print("\n=== 4. outcome ===")
    report(result=result)
    print(f"\nJournal: {experiment.journal.path_for(result.run_id)}")


if __name__ == "__main__":
    main()
