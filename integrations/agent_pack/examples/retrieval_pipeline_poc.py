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
scored directly as recall and precision over those IDs. Quality is the mean of the per-case recall, with a case
that broke one of its budgets scoring nothing, so a configuration that finds more of the evidence measures as
better even while no case yet finds all of it. That is what makes this affordable: one case costs a
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
from haystack.components.generators.chat import OpenAIChatGenerator, OpenAIResponsesChatGenerator
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
from haystack_integrations.agent_pack.optimization.agent import OPTIMIZER_PROMPT_CACHE_KEY
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
is scored against the documents the answer needed. The per-case query budget counts every query issued, including
the original question when the expansion keeps it, so an expansion count set to the budget will exceed it.
Quality is the mean recall over the cases, so retrieving more
of what a question needs registers even when no single case is yet complete; a case that breaks one of its
budgets contributes nothing at all, however much of the evidence it found. Nothing writes an answer, so nothing is
gained by adding a generator.

Cases require evidence spread across several documents, and one query phrased for the whole question tends to
surface only the documents that share its wording. Expansion buys recall with model calls, and a wider candidate
set buys it with precision; the case reports recall and precision separately, and names the documents that were
missed, so the two are distinguishable.

One property of the expander is worth knowing before a measurement is spent discovering it. Its expansion count is
applied by truncating whatever the model returned and keeping that many from the front, and the worked examples
built into its default prompt demonstrate three or four expansions whatever the count is set to. A low count
therefore does not ask the model for fewer queries: it pays for the three or four the examples elicit and then
discards all but an arbitrary prefix of them, which the run reports as a truncation warning. That prompt is part
of this configuration and can be edited, so the count and the examples it shows can be made to agree. The original
question is also appended unless the model already produced it, so the queries actually issued are usually one
more than the count.

Each case also limits how many documents the pipeline may return. The limit is on what comes out, not on what the
pipeline looks at, so past a certain point recall cannot be bought by widening: what is returned has to be the
right subset of whatever was considered.

Once that limit binds, the way past it is to stop treating those two things as the same. Retrieve a wide candidate
set, then rank it and return only the best of it: the limit applies to the ranked output, while the candidate set
behind it can be as wide as recall needs. Use `LLMRanker` for that, placed between retrieval and the pipeline's
`documents` output. It takes a `query` input of its own alongside the documents, and its `top_k` decides how many
survive. Inspect it before writing it in, so its parameters and serialized shape come from the component rather
than from memory.

Why that kind of ranker, on this evaluation set. Its questions are answered by several documents together, and no
one document answers such a question on its own; what the output needs is coverage of the separate facets, not the
several best matches for the question as a whole. `LLMRanker` puts the whole question and every candidate into one
prompt and chooses from them together, so it can see that a candidate repeats evidence already selected and pick
one that adds a facet instead. A ranker that scores each candidate on its own cannot: it has no way to know what
else it is returning, so it fills the output with the nearest matches to whichever facet the question words most
strongly, and the remaining facets go unretrieved. Measured here, judging the candidates jointly retrieved close
to twice the evidence that scoring them one at a time did.

The two stages are answerable to different things. Retrieval before the ranker is judged only on whether the
evidence is somewhere in the candidate set, so widening it costs nothing that is measured and a candidate document
that is never retrieved cannot be recovered later. The ranker is what decides the answer, so it is where being
selective matters. What the case scores in the end is recall, so a ranker that leaves a needed document out has
lost something a narrower candidate set could never have given back.

Its `top_k` is a ceiling on the answer, and the ceiling and the prompt do different jobs. Cases here need between
two and four documents, so a ceiling of four leaves a four-document case no room to be wrong once, while a
two-document case has nothing useful to do with the spare slots. Set the ceiling above the most any case needs and
let the prompt decide how many actually come back — asking for the documents that together cover the question and
no others, so the extra room is available when a question needs it and unused when it does not. A run reporting
that fewer documents came back than the ceiling allows is not necessarily wasting it; a run reporting exactly the
ceiling on every case is being truncated by it.

Two things about running it. It calls a chat model once per case with every candidate's text in the prompt, so
what it costs grows with the candidate set. And it must not be given a `temperature`; the models available here
reject the parameter outright, and the failure is swallowed into a warning that leaves the documents unranked
rather than raising.

Merging the results of several queries is where this goes wrong quietly, and it decides which kind of ranking is
worth adding. A keyword retriever's score is a property of the query that produced it, not a scale shared between
queries, so an order produced by pooling per-query results and sorting them by score means little, and taking the
best few of that order is close to taking an arbitrary few. Reciprocal rank fusion is the standard answer to that
problem, but it merges on each document's rank within its own result list and therefore needs those lists kept
apart. Check whether the retrieval in this configuration keeps them apart or pools them before anything downstream
sees them: if they are already pooled, a joiner placed after it can only deduplicate and trim, and the way to
improve the returned subset is to rank it on something other than those scores.

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


def build_optimizer_generator(model: str | None) -> OpenAIResponsesChatGenerator | None:
    """
    Build the optimizer's generator on a named model, leaving everything else as the default.

    Only the model differs from what the library would build, so a run that changes it measures the model rather
    than the settings around it.

    :param model: Model to reason with, or None to accept the library default.
    :returns: The generator, or None to let `create_harness_optimizer_agent` choose.
    """
    if model is None:
        return None
    return OpenAIResponsesChatGenerator(
        model=model,
        timeout=180.0,
        max_retries=5,
        generation_kwargs={"prompt_cache_key": OPTIMIZER_PROMPT_CACHE_KEY, "reasoning": {"effort": "low"}},
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
        help="Cases to evaluate. Quality averages their recall, so more cases make the measurement finer as well "
        "as less noisy; each one costs a model call per candidate.",
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
    parser.add_argument(
        "--optimizer-model",
        help="Model the optimizer itself reasons with. Its turns are most of what an experiment costs on a harness "
        "whose cases are cheap, so what it is worth paying for them is itself a measurable question. Defaults to "
        "whatever `create_harness_optimizer_agent` chooses.",
    )
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
            chat_generator=build_optimizer_generator(model=arguments.optimizer_model),
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
