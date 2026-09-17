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
# Nothing generates an answer. The MultiHopRAG eval cases already name the documents an answer needs, so retrieval is
# scored directly against those IDs. Quality is the mean of the per-eval-case recall, so a configuration that finds
# more of the evidence measures as better even while no eval case yet finds all of it.
#
# Run from `integrations/agent_pack` with `OPENAI_API_KEY` set. The corpus requires `datasets`:
#
#     hatch run test:python examples/retrieval_pipeline_optimization.py
#     hatch run test:python examples/retrieval_pipeline_optimization.py --max-eval-cases 5 --max-iterations 2
#     hatch run test:python examples/retrieval_pipeline_optimization.py --docs-mcp

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator, OpenAIResponsesChatGenerator
from haystack.components.query import QueryExpander
from haystack.components.retrievers import MultiQueryTextRetriever
from haystack.document_stores.types import DocumentStore
from multihop_rag import CORPUS_KEY, SPLIT_LENGTH, SPLIT_OVERLAP, build_eval_cases, prepare_corpus
from util import build_bm25_retriever, quiet_hub_warnings

from haystack_integrations.agent_pack.optimization import (
    CandidateProgress,
    ExperimentJournal,
    ExperimentResult,
    HarnessOptimizationExperiment,
    ModelPrice,
    ModelPriceCatalog,
    OptimizationObjectives,
    create_harness_optimizer_agent,
)
from haystack_integrations.agent_pack.optimization.prompts import OPTIMIZER_PROMPT_CACHE_KEY
from haystack_integrations.evaluation import EvalMetrics, RetrievalEvalCase, RetrievalHarnessEvaluator

WORKSPACE = Path(".agent-pack-retrieval-poc")
EXPANDER_MODEL = "gpt-5.6-luna"

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


def build_reference_pipeline(store: DocumentStore, model: str) -> Pipeline:
    """
    Build the under-configured retrieval pipeline whose complete configuration will be optimized.

    :param store: The corpus to retrieve from.
    :param model: The model the query expander reasons with.
    :returns: The reference pipeline.
    """
    pipeline = Pipeline()
    # We purposely poorly configure the query expander, to see whether the optimizer can find a better configuration.
    # In this case we use OpenAIChatGenerator instead of OpenAIResponsesChatGenerator, and we set  n_expansions to 1.
    pipeline.add_component(
        "expander",
        QueryExpander(chat_generator=OpenAIChatGenerator(model=model), n_expansions=POOR_EXPANSIONS),
    )
    # The retriever is also poorly configured, with a top_k of 2.
    pipeline.add_component(
        "retriever", MultiQueryTextRetriever(retriever=build_bm25_retriever(store=store, top_k=POOR_TOP_K))
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


def retrieval_guidance(k: int) -> str:
    """
    State the rank cutoff the harness scores at, alongside the rest of what it knows about itself.

    The cutoff is the harness's own setting, and an optimizer that is not told it has to find it by measurement,
    spending an evaluation to learn a number the harness could simply have stated.

    :param k: The rank cutoff cases are scored at.
    :returns: The harness guidance with its cutoff filled in.
    """
    cutoff = (
        f"Each eval case is scored at recall@{k}: only the first {k} documents the pipeline returns count towards "
        f"its score, in the order it returned them. Returning more than {k} is not penalized, it simply earns "
        f"nothing for the documents past the cutoff. Nothing caps how many queries the pipeline may issue, but "
        f"every query costs a model call and wall-clock time, and both are measured."
    )
    return f"{_RETRIEVAL_OPTIMIZER_GUIDANCE}\n\n{cutoff}"


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


def _stages(details: dict) -> str:
    """
    Render how much each component emitted, in execution order.

    :param details: One measurement's details, carrying `stage_output_sizes`.
    :returns: One `component.socket median (min-max)` entry per stage, or a note when nothing was recorded.
    """
    stages = details.get("stage_output_sizes") or {}
    entries = [
        f"{component}.{socket} {sizes['median']} ({sizes['min']}-{sizes['max']})"
        for component, sockets in stages.items()
        for socket, sizes in sockets.items()
    ]
    return " → ".join(entries) if entries else "not recorded"


# Colour is worth having on a terminal and only noise in a redirected log.
_STYLE = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "off": "\033[0m",
}


def _paint(text: str, *names: str) -> str:
    """
    Apply terminal styles, or none when stdout is redirected.

    :param text: The text to style.
    :param names: Style names to apply.
    :returns: The text, wrapped in escape codes only when a terminal is reading them.
    """
    if not sys.stdout.isatty():
        return text
    return "".join(_STYLE[name] for name in names) + text + _STYLE["off"]


def _rule(label: str) -> str:
    """
    Draw a labelled separator the width of the terminal.

    :param label: What the section below the rule is.
    :returns: The rule, padded to the terminal width.
    """
    width = min(shutil.get_terminal_size(fallback=(100, 24)).columns, 110)
    return _paint(f"── {label} " + "─" * max(width - len(label) - 6, 4), "dim")


def _delta(current: float, baseline: float, *, digits: int = 2, prefix: str = "", more_is_better: bool = True) -> str:
    """
    Render a change against the baseline, coloured by whether it is an improvement.

    :param current: The candidate's value.
    :param baseline: The reference's value.
    :param digits: Decimal places to show.
    :param prefix: Unit to put in front of the number, such as a currency symbol.
    :param more_is_better: Whether a rise is the good direction, which it is for quality and is not for spend.
    :returns: A signed change, or an empty string when nothing moved.
    """
    change = current - baseline
    if abs(change) < 10**-digits / 2:
        return ""
    rendered = f"{prefix}{abs(change):.{digits}f}"
    improved = change > 0 if more_is_better else change < 0
    return _paint(f"  ({'+' if change > 0 else '-'}{rendered})", "green" if improved else "red")


# Serialized Haystack keys are lower snake case. Prose headings inside a rewritten prompt are not, which is
# what tells the two apart in a hunk that does not show the block scalar those headings sit under.
_SETTING = re.compile(r"^[+-]\s*([a-z_][a-z0-9_.]*):\s*(.*)$")


def _summarize(value: str, limit: int = 28) -> str:
    """
    Reduce one YAML value to something that fits on a shared line.

    :param value: The value as it appears in the diff.
    :param limit: How many characters to keep.
    :returns: The value, or a word standing in for one too long to show.
    """
    collapsed = " ".join(value.split())
    # A block scalar marker names no value; what follows it is the rewritten text itself.
    if not collapsed or collapsed.startswith("|") or collapsed.startswith(">"):
        return ""
    # A dotted class path is only worth its last segment on a shared line.
    if "." in collapsed and " " not in collapsed:
        collapsed = collapsed.rsplit(".", 1)[-1]
    return collapsed if len(collapsed) <= limit else "rewritten"


def _changes(diff: str, limit: int = 5) -> str:
    """
    Reduce a unified diff to the settings it actually changed.

    A candidate's diff is mostly re-indented YAML and rewritten prompts. What a reader wants from it is which
    knobs moved, so keys are paired across the removed and added sides, prose inside a rewritten prompt is
    skipped, and anything past the limit is counted.

    :param diff: The unified diff between this candidate and its parent.
    :param limit: How many changed settings to name before counting the rest.
    :returns: One line naming the changes, or a note when the diff holds none of this shape.
    """
    removed: dict[str, str] = {}
    added: dict[str, str] = {}
    order: list[str] = []
    skipped_prose = False
    # A key with no value of its own opens a subtree, and a block scalar opens prose. Both read as a wall of
    # settings when what changed is one component or one prompt, so everything indented under them is skipped.
    skip_below: dict[str, int | None] = {"+": None, "-": None}
    for line in diff.splitlines():
        if line.startswith(("+++", "---", "@@")) or line[:1] not in {"+", "-"}:
            # A prompt is usually edited in place, so its own key stays as context and only the prose moves.
            # That context line is the only thing saying the lines under it are prose rather than settings.
            opener = _SETTING.match(f"+{line[1:]}") if line[:1] == " " else None
            depth = len(line[1:]) - len(line[1:].lstrip())
            carries_text = opener is not None and not _summarize(value=opener.group(2))
            skip_below = {"+": depth, "-": depth} if carries_text else {"+": None, "-": None}
            continue
        side, body = line[0], line[1:]
        indent = len(body) - len(body.lstrip())
        opened = skip_below[side]
        if opened is not None and (not body.strip() or indent > opened):
            skipped_prose = skipped_prose or bool(body.strip())
            continue
        skip_below[side] = None
        match = _SETTING.match(line)
        if match is None:
            continue
        key, raw = match.group(1), match.group(2)
        value = _summarize(value=raw)
        if not value or raw.strip().startswith(("|", ">")):
            skip_below[side] = indent
        if key not in order:
            order.append(key)
        (added if side == "+" else removed)[key] = value

    entries = []
    for key in order:
        was, now = removed.get(key), added.get(key)
        if was is not None and now is not None:
            entries.append(f"{key} {was} → {now}" if was and now else f"{key} rewritten")
        elif now is not None:
            entries.append(_paint(f"+ {key}", "green") + (f" {now}" if now else ""))
        else:
            entries.append(_paint(f"- {key}", "red"))
    if skipped_prose:
        entries.append(_paint("prompt text", "dim"))
    if not entries:
        return _paint("whitespace and formatting only", "dim")
    shown = "  ·  ".join(entries[:limit])
    return shown if len(entries) <= limit else f"{shown}  ·  {_paint(f'+{len(entries) - limit} more', 'dim')}"


def _measurements(metrics: EvalMetrics, baseline: EvalMetrics | None = None) -> list[str]:
    """
    Render one measurement as aligned lines, against the baseline when there is one.

    :param metrics: The measurement to render.
    :param baseline: The reference measurement, or None for the reference itself.
    :returns: The lines to print, already labelled.
    """
    details = metrics.details
    reference = baseline.details if baseline is not None else None
    quality = f"{metrics.quality:.2f}" + (_delta(metrics.quality, baseline.quality) if baseline else "")
    recall = f"{details['mean_recall_at_k']:.2f}" + (
        _delta(details["mean_recall_at_k"], reference["mean_recall_at_k"]) if reference else ""
    )
    cost = format_cost(cost=metrics.cost)
    if baseline is not None and metrics.cost is not None and baseline.cost is not None:
        cost += _delta(metrics.cost, baseline.cost, digits=4, prefix="$", more_is_better=False)
    latency = f"{metrics.latency_ms:.0f}ms" + (
        _delta(metrics.latency_ms, baseline.latency_ms, digits=0, more_is_better=False) if baseline else ""
    )
    scores = (
        f"    {_paint('quality ', 'bold')} {quality}   recall@k {recall}   "
        f"precision@k {details['mean_precision_at_k']:.2f}   returned {details['mean_retrieved']:.1f}"
    )
    return [
        scores,
        f"    {_paint('spend   ', 'bold')} {cost}   latency {latency}",
        f"    {_paint('stages  ', 'bold')} {_stages(details=details)}",
    ]


def print_baseline(metrics: EvalMetrics) -> None:
    """
    Print the reference measurement every candidate is ranked against.

    :param metrics: The priced reference measurement.
    """
    print()
    print(_rule("baseline · the reference pipeline"))
    for line in _measurements(metrics=metrics):
        print(line)


def print_candidate(progress: CandidateProgress) -> None:
    """
    Print one candidate as the search measures it, so a long run can be watched.

    :param progress: The measured candidate, its gates, and whether it now leads.
    """
    evaluation = progress.evaluation
    print()
    print(_rule(f"candidate {progress.position}/{progress.total} · {evaluation.candidate_id}"))
    if evaluation.configuration is not None:
        print(f"    {_paint('why     ', 'bold')} {evaluation.configuration.rationale}")
        print(f"    {_paint('changed ', 'bold')} {_changes(diff=evaluation.configuration.diff)}")
    if evaluation.metrics is None:
        print(f"    {_paint('verdict ', 'bold')} {_paint('did not run: ' + str(evaluation.failure), 'red')}")
        return
    for line in _measurements(metrics=evaluation.metrics, baseline=progress.baseline):
        print(line)
    verdict = (
        _paint("cleared every gate", "green")
        if not progress.gate_failures
        else _paint("gated: " + ", ".join(progress.gate_failures), "red")
    )
    if progress.is_best:
        verdict += _paint("  ← best so far", "cyan", "bold")
    print(f"    {_paint('verdict ', 'bold')} {verdict}")


def report(result: ExperimentResult) -> None:
    """Print what the search cost and what it recommends, the candidates having printed as they were measured."""
    print()
    print(_rule("what the search itself cost"))
    usage = ", ".join(
        f"{model}: {tokens.input_tokens} in / {tokens.output_tokens} out"
        for model, tokens in result.optimizer_usage.items()
    )
    print(f"    {_paint('optimizer', 'bold')} {usage or 'none recorded'}")
    print(f"    {_paint('priced at', 'bold')} {format_cost(cost=result.optimizer_cost)} (input charged at full price)")

    print()
    print(_rule("recommendation"))
    if result.recommendation is None:
        print(f"    {_paint('none', 'red')}: no candidate cleared every gate and improved on the reference")
        return
    recommendation = result.recommendation
    measured = recommendation.evaluation.metrics
    if measured is not None:
        print(
            f"    {_paint('candidate', 'bold')} {recommendation.evaluation.candidate_id}   "
            f"quality {measured.quality:.2f}{_delta(measured.quality, result.baseline.quality)}   "
            f"cost {format_cost(cost=measured.cost)}"
        )
    print(f"    {_paint('why      ', 'bold')} {recommendation.configuration.rationale}")
    print(f"    {_paint('reasons  ', 'bold')} {', '.join(recommendation.reasons)}")
    print(f"    {_paint('config   ', 'bold')} {result.artifact_directory / 'recommended.yaml'}")
    print(
        f"\n    {_paint('Nothing was deployed. Approving this recommendation is a separate, human decision.', 'dim')}"
    )


def parse_args() -> argparse.Namespace:
    """Parse the PoC command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", choices=("in_memory", "opensearch"), default="in_memory")
    parser.add_argument("--expander-model", default=EXPANDER_MODEL)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE, help="Directory for runs and YAML artifacts.")
    parser.add_argument("--config", type=Path, help="Optional editable YAML file; created from the reference.")
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
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument(
        "--max-quality-loss",
        type=float,
        default=0.05,
        help="Maximum absolute pass-rate loss from the reference. Keep it at no less than one eval case, or the "
        "measurement noise of a single eval case gates out real improvements.",
    )
    parser.add_argument("--primary", choices=("cost", "latency", "quality"), default="quality")
    parser.add_argument("--max-concurrent-eval-cases", type=int, default=6)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--optimizer-steps", type=int, default=24, help="Editing steps per optimizer turn.")
    parser.add_argument(
        "--optimizer-model",
        help="Model the optimizer itself reasons with. Its turns are most of what an experiment costs on a harness "
        "whose eval cases are cheap, so what it is worth paying for them is itself a measurable question. Defaults to "
        "whatever `create_harness_optimizer_agent` chooses.",
    )
    parser.add_argument("--docs-mcp", action="store_true", help="Give the optimizer the Haystack documentation MCP.")
    parser.add_argument("--fresh", action="store_true", help="Remove saved runs and journals before starting.")
    return parser.parse_args()


class _Narrate(logging.Filter):
    """Drop what this script now prints itself, and shorten what it keeps to what a reader is watching for."""

    def filter(self, record: logging.LogRecord) -> bool:
        # The candidate block prints all of this, together with the change and rationale that produced it.
        if hasattr(record, "candidate_id") and hasattr(record, "gates"):
            return False
        # The search cost is reported once at the end.
        if hasattr(record, "turns") and hasattr(record, "cost"):
            return False
        if hasattr(record, "steps") and hasattr(record, "calls"):
            record.msg = _paint(f"optimizer · {record.steps} steps · {' → '.join(record.calls)}", "dim")
        return True


def enable_progress_reporting() -> None:
    """Route library progress to stdout, line buffered, so a redirected run reports as it goes."""
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    quiet_hub_warnings()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("    %(message)s"))
    handler.addFilter(_Narrate())
    progress = logging.getLogger("haystack_integrations.agent_pack")
    progress.handlers.clear()
    progress.addHandler(handler)
    progress.setLevel(logging.INFO)
    progress.propagate = False
    # An edit the optimizer repairs within its turn is normal, and Haystack logs the whole rejected call with it.
    # The repair is visible in the turn's step count, so the dump only buries the measurements.
    logging.getLogger("haystack.components.agents.tool_calling").setLevel(logging.CRITICAL)


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
    if arguments.primary == "latency" and arguments.max_concurrent_eval_cases > 1:
        message = "Ranking by latency requires --max-concurrent-eval-cases 1."
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

    reference = build_reference_pipeline(store=store, model=arguments.expander_model)
    print(
        f"  reference: n_expansions={POOR_EXPANSIONS} top_k={POOR_TOP_K} model={arguments.expander_model}; "
        f"scored at recall@{arguments.k}"
    )

    print("\n=== 2. optimization experiment ===")
    evaluator = RetrievalHarnessEvaluator(k=arguments.k, max_concurrent_eval_cases=arguments.max_concurrent_eval_cases)
    experiment = HarnessOptimizationExperiment(
        reference=reference,
        eval_cases=eval_cases,
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
            documentation_tools=arguments.docs_mcp,
            additional_instructions=retrieval_guidance(k=arguments.k),
            max_agent_steps=arguments.optimizer_steps,
        ),
        max_iterations=arguments.max_iterations,
        config_path=arguments.config,
        on_baseline=print_baseline,
        on_candidate=print_candidate,
        configuration_key=f"{CORPUS_KEY}:{SPLIT_LENGTH}:{SPLIT_OVERLAP}:{document_count}",
    )
    result = experiment.run()

    print("\n=== 3. outcome ===")
    report(result=result)
    print()
    print(_paint(f"    journal  {experiment.journal.path_for(result.run_id)}", "dim"))
    print(_paint(f"    context  {result.measurement_context} · {result.run_id}", "dim"))


if __name__ == "__main__":
    main()
