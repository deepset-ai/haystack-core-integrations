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

WORKSPACE = Path(".agent-pack-retrieval-poc")
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
The configuration is a one-shot retrieval pipeline, and retrieval is all of it. A question enters at the `query`
input, whatever the pipeline does with it must end at exactly one unconnected `documents` output, and that output
is scored against the documents the answer needed.
Quality is the mean recall over the eval cases, so retrieving more
of what a question needs registers even when no single eval case is yet complete; an eval case that breaks one of its
budgets contributes nothing at all, however much of the evidence it found. Nothing writes an answer, so nothing is
gained by adding a generator.

Every document carries `title`, `category`, `source`, `author`, `published_at` and `url` alongside its content,
and nothing in this pipeline shows them to a model unless a prompt renders them. The questions name outlets and
dates constantly, so a component that judges documents can be given those fields to judge on.

Eval cases require evidence spread across several documents, and one query phrased for the whole question tends to
surface only the documents that share its wording. Expansion buys recall with model calls, and a wider candidate
set buys it with precision; the eval case reports recall and precision separately, and names the documents that were
missed, so the two are distinguishable.

One property of the expander is worth knowing before a measurement is spent discovering it. Its expansion count is
applied by truncating whatever the model returned and keeping that many from the front, and the worked examples
built into its default prompt demonstrate three or four expansions whatever the count is set to. A low count
therefore does not ask the model for fewer queries: it pays for the three or four the examples elicit and then
discards all but an arbitrary prefix of them, which the run reports as a truncation warning. That prompt is part
of this configuration and can be edited, so the count and the examples it shows can be made to agree. The original
question is also appended unless the model already produced it, so the queries actually issued are usually one
more than the count.

Each eval case also limits how many documents are scored, and how many queries the pipeline may issue; the exact
numbers are stated below. The document limit is on what comes out, not on what the pipeline looks at, so past a
certain point recall cannot be bought by widening: what is returned has to be the right subset of whatever was
considered. Only the first that-many documents count towards recall, in the order the pipeline returned them, so
returning more than the limit wastes the places past it rather than voiding the eval case: set the final ranker's
ceiling at the limit rather than above it, and a run that overshoots by one has lost one document's worth of
credit. The query limit is not forgiving in the same way, since a query already cost what it cost: exceed it and
the eval case scores nothing.

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
selective matters. What the eval case scores in the end is recall, so a ranker that leaves a needed document out has
lost something a narrower candidate set could never have given back.

Fill the allowance, and start there rather than working up to it. Only the documents inside the limit are scored,
precision is not scored at all, and one more document inside the limit can either match a needed one or be
ignored — it cannot cost anything. Returning fewer than the limit is giving those places away. Set the ranker's
`top_k` at the limit and write its prompt to use it: ask for the most useful documents up to that many, ordered
best first, rather than for the smallest set that looks sufficient. A question needing two documents loses nothing
by coming back with the limit's worth, and a question needing four is what the other places were for. Asking for a
minimal or "two to four" set has measured worse here more than once, and never better.

Trimming what comes back is worth measuring only once recall has stopped moving, and then as a latency and cost
question rather than a scoring one.

Two things about running it. It calls a chat model once per eval case with every candidate's text in the prompt, so
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
        f"every query costs a model call and wall-clock time, and both are measured."
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
    """Parse the PoC command line."""
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
        message = "OPENAI_API_KEY must be set to run this walkthrough."
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
