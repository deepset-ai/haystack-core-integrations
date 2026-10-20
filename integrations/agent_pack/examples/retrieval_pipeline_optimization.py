# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Optimize a one-shot retrieval Pipeline against a labelled evaluation set.
#
# The configuration under optimization is a plain Haystack Pipeline — a `QueryExpander` that turns one question
# into several, feeding a `MultiQueryTextRetriever` that runs them all against the store:
#
#     QueryExpander.queries -> MultiQueryTextRetriever.queries -> documents
#
# Each MultiHopRAG eval case names the documents its answer needs, and quality is the mean recall over those
# documents.
#
# Run from `integrations/agent_pack` with `OPENAI_API_KEY` set. The corpus requires `datasets`:
#
#     hatch run test:python examples/retrieval_pipeline_optimization.py
#     hatch run test:python examples/retrieval_pipeline_optimization.py --max-eval-cases 5 --max-iterations 2
#     hatch run test:python examples/retrieval_pipeline_optimization.py --docs-mcp

import argparse
import os
import shutil
from pathlib import Path

from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.query import QueryExpander
from haystack.components.retrievers import MultiQueryTextRetriever
from haystack.document_stores.types import DocumentStore
from multihop_rag import CORPUS_KEY, SPLIT_LENGTH, SPLIT_OVERLAP, build_eval_cases, prepare_corpus
from util import ExperimentPrinter, build_bm25_retriever, build_optimizer_generator, enable_progress_reporting, paint

from haystack_integrations.agent_pack.evaluation import (
    EvalMetrics,
    ModelPrice,
    RetrievalEvalCase,
    RetrievalHarnessEvaluator,
)
from haystack_integrations.agent_pack.optimization import (
    ExperimentJournal,
    HarnessOptimizationExperiment,
    OptimizationObjectives,
)

WORKSPACE = Path(".agent-pack-retrieval-optimization")
EXPANDER_MODEL = "gpt-5.6-luna"
# The measurement the experiment treats as quality.
QUALITY_METRIC = "mean_recall_at_k"

# USD prices per million tokens, used only to rank candidates against each other.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
}

_RETRIEVAL_OPTIMIZER_GUIDANCE = """
The configuration is a one-shot retrieval pipeline. A question enters at the `query` input and the pipeline must end
in exactly one unconnected `documents` output, which is scored against the documents the answer needs. Nothing
writes an answer, so adding a generator gains nothing.

Quality is the mean recall over the eval cases, so finding more of what a question needs registers even before any
eval case is complete. Precision is reported but not scored. Each eval case needs evidence spread across several
documents, and one query phrased for the whole question tends to surface only the documents that share its wording.

Every document carries `title`, `category`, `source`, `author`, `published_at` and `url` alongside its content. The
questions name outlets and dates constantly, but no model sees those fields unless a prompt renders them.

`QueryExpander` applies its expansion count by keeping that many queries from the front of whatever the model
returned, and the examples in its default prompt show three or four expansions whatever the count is. A low count
therefore pays for three or four queries and discards the rest, which the run reports as a truncation warning. The
prompt is part of this configuration and can be edited. The original question is also appended, so the queries
issued are usually one more than the count.

A keyword retriever's score belongs to the query that produced it, so pooling several queries' results and sorting
them by score gives a nearly arbitrary order. Check whether this configuration pools them: if it does, the way to
improve what is returned is to rank it on something other than those scores.

Retrieving a wide candidate set and then ranking it is usually what moves recall past the cutoff. Use `LLMRanker`
between retrieval and the `documents` output: it takes its own `query` input and judges every candidate in one prompt,
so it can pick documents that cover different parts of the question rather than near-duplicates of one part. Set its
`top_k` at the cutoff and ask its prompt for the most useful documents up to that many, best first, rather than for
the smallest set that looks sufficient. It costs one model call per eval case, growing with the candidate set, and it
must not be given a `temperature`: the models here reject it, and the failure only shows as a warning with the
documents left unranked.

Inspect any component before writing it into the configuration, so its parameters and serialized shape come from the
installed code rather than from memory.
""".strip()


POOR_EXPANSIONS = 1
POOR_TOP_K = 2


def build_reference_pipeline(store: DocumentStore) -> Pipeline:
    """
    Build the under-configured retrieval pipeline whose complete configuration will be optimized.

    :param store: The corpus to retrieve from.
    :returns: The reference pipeline.
    """
    pipeline = Pipeline()
    # We purposely poorly configure the query expander, to see whether the optimizer can find a better configuration.
    # In this case we use OpenAIChatGenerator instead of OpenAIResponsesChatGenerator, and we set  n_expansions to 1.
    pipeline.add_component(
        "expander",
        QueryExpander(chat_generator=OpenAIChatGenerator(model=EXPANDER_MODEL), n_expansions=POOR_EXPANSIONS),
    )
    # The retriever is also poorly configured, with a top_k of 2.
    pipeline.add_component(
        "retriever", MultiQueryTextRetriever(retriever=build_bm25_retriever(store=store, top_k=POOR_TOP_K))
    )
    pipeline.connect("expander.queries", "retriever.queries")
    return pipeline


def build_prices() -> dict[str, ModelPrice]:
    """Build informational prices for the models this example knows about."""
    return {
        model: ModelPrice(input_cost_per_million=prices[0], output_cost_per_million=prices[1])
        for model, prices in MODEL_PRICES.items()
    }


def retrieval_guidance(k: int) -> str:
    """Return the optimizer guidance for this harness, with the rank cutoff `k` it scores at filled in."""
    cutoff = (
        f"Each eval case is scored at recall@{k}: only the first {k} documents the pipeline returns count towards "
        f"its score, in the order it returned them. Returning more than {k} is not penalized, it simply earns "
        f"nothing for the documents past the cutoff. Nothing caps how many queries the pipeline may issue, but "
        f"each one adds a retrieval, model calls add cost, and both cost and duration are measured."
    )
    return f"{_RETRIEVAL_OPTIMIZER_GUIDANCE}\n\n{cutoff}"


def _components(details: dict) -> str:
    """
    Render how much each component emitted, in the order the components finished.

    :param details: One measurement's details, carrying `component_output_sizes`.
    :returns: One `component.socket median (min-max)` entry per component, or a note when nothing was recorded.
    """
    sizes_by_component = details.get("component_output_sizes") or {}
    entries = [
        f"{component}.{socket} {sizes['median']} ({sizes['min']}-{sizes['max']})"
        for component, sockets in sizes_by_component.items()
        for socket, sizes in sockets.items()
    ]
    return " → ".join(entries) if entries else "not recorded"


def retrieval_details(metrics: EvalMetrics, baseline: EvalMetrics | None) -> list[str]:  # noqa: ARG001
    """
    Render what the retrieval harness evaluator reports beyond recall.

    :param metrics: The measurement to render.
    :param baseline: The reference measurement, which these lines are not compared against.
    :returns: The precision and returned-document line, and the component output sizes.
    """
    details = metrics.details
    return [
        (
            f"    {paint('scores  ', 'bold')} precision@k {details['mean_precision_at_k']:.2f}   "
            f"returned {details['mean_retrieved']:.1f}"
        ),
        f"    {paint('shape   ', 'bold')} {_components(details=details)}",
    ]


def parse_args() -> argparse.Namespace:
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", choices=("in_memory", "opensearch"), default="in_memory")
    parser.add_argument("--workspace", type=Path, default=WORKSPACE, help="Directory for runs and YAML artifacts.")
    parser.add_argument(
        "--max-eval-cases",
        type=int,
        default=20,
        help="Eval cases to evaluate. Quality averages their recall, so more of them make the measurement finer "
        "as well as less noisy; each one costs a model call per candidate.",
    )
    parser.add_argument(
        "--eval-case-seed", type=int, default=0, help="Selects which eval cases are drawn from the dataset."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Rank cutoff eval cases are scored at, giving recall@k. Only the first k documents returned count, so a "
        "pipeline is measured on what it put at the top rather than on how much it returned. Recall with no "
        "cutoff is maximized by returning most of the corpus.",
    )
    parser.add_argument(
        "--max-quality-loss",
        type=float,
        default=0.05,
        help="How far below the reference's mean recall@k a candidate may fall. Keep it at no less than one eval "
        "case's worth, or the measurement noise of a single eval case gates out real improvements.",
    )
    parser.add_argument("--primary", choices=("cost", "duration", "quality"), default="quality")
    parser.add_argument("--max-concurrent-eval-cases", type=int, default=6)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--optimizer-steps", type=int, default=24, help="Editing steps per optimizer turn.")
    parser.add_argument(
        "--optimizer-model",
        help="Model the optimizer itself reasons with. Its turns are most of what an experiment costs on a harness "
        "whose eval cases are cheap, so what it is worth paying for them is itself a measurable question. Defaults to "
        "the experiment's default optimizer model.",
    )
    parser.add_argument("--docs-mcp", action="store_true", help="Give the optimizer the Haystack documentation MCP.")
    parser.add_argument("--fresh", action="store_true", help="Remove saved runs and journals before starting.")
    return parser.parse_args()


def main() -> None:
    """Build the experiment and run it end to end."""
    arguments = parse_args()
    enable_progress_reporting()
    if not os.environ.get("OPENAI_API_KEY"):
        message = "OPENAI_API_KEY must be set to run this example."
        raise SystemExit(message)
    if arguments.max_eval_cases < 1:
        message = "--max-eval-cases must be at least 1."
        raise SystemExit(message)
    if arguments.primary == "duration" and arguments.max_concurrent_eval_cases > 1:
        message = "Ranking by duration requires --max-concurrent-eval-cases 1."
        raise SystemExit(message)
    if arguments.fresh and arguments.workspace.exists():
        shutil.rmtree(path=arguments.workspace)

    print("=== 1. set up corpus and evaluation set ===")
    store, articles = prepare_corpus(backend=arguments.store)
    document_count = store.count_documents()
    print(f"  {CORPUS_KEY} on {arguments.store}: {document_count} chunks")

    labelled = build_eval_cases(articles=articles, limit=arguments.max_eval_cases, seed=arguments.eval_case_seed)
    eval_cases = [RetrievalEvalCase(question=eval_case.question, evidence=eval_case.evidence) for eval_case in labelled]
    expected = sum(len(eval_case.expected_document_ids) for eval_case in eval_cases)
    print(f"  cases: {len(eval_cases)} labelled from evidence, expecting {expected} documents in total")

    reference = build_reference_pipeline(store=store)
    print(
        f"  reference: n_expansions={POOR_EXPANSIONS} top_k={POOR_TOP_K} model={EXPANDER_MODEL}; "
        f"scored at recall@{arguments.k}"
    )

    print("\n=== 2. optimization experiment ===")
    evaluator = RetrievalHarnessEvaluator(k=arguments.k, max_concurrent_eval_cases=arguments.max_concurrent_eval_cases)
    printer = ExperimentPrinter(quality_metric=QUALITY_METRIC, quality_label="recall@k", detail_lines=retrieval_details)
    experiment = HarnessOptimizationExperiment(
        reference=reference,
        eval_cases=eval_cases,
        evaluator=evaluator,
        prices=build_prices(),
        objectives=OptimizationObjectives(
            quality_metric=QUALITY_METRIC,
            max_quality_loss=arguments.max_quality_loss,
            primary=arguments.primary,
        ),
        journal=ExperimentJournal(directory=arguments.workspace / "journals"),
        optimizer_llm=build_optimizer_generator(model=arguments.optimizer_model),
        optimizer_additional_instructions=retrieval_guidance(k=arguments.k),
        optimizer_max_agent_steps=arguments.optimizer_steps,
        optimizer_documentation_tools=arguments.docs_mcp,
        max_iterations=arguments.max_iterations,
        on_baseline=printer.on_baseline,
        on_candidate=printer.on_candidate,
        configuration_key=f"{CORPUS_KEY}:{SPLIT_LENGTH}:{SPLIT_OVERLAP}:{document_count}",
    )
    result = experiment.run()

    print("\n=== 3. outcome ===")
    printer.report(result=result)
    print()
    print(paint(f"    journal  {experiment.journal.path_for(result.run_id)}", "dim"))
    print(paint(f"    context  {result.measurement_context} · {result.run_id}", "dim"))


if __name__ == "__main__":
    main()
