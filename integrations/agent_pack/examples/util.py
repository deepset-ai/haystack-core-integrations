# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import logging
import os
import re
import shutil
import sys
import threading
from collections.abc import Callable
from statistics import median
from typing import Any, cast

from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore
from haystack.lazy_imports import LazyImport

with LazyImport(message='Run "pip install opensearch-haystack" to use an OpenSearch store.') as opensearch_import:
    from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

from haystack_integrations.agent_pack.evaluation import EvalMetrics
from haystack_integrations.agent_pack.optimization import CandidateProgress, ExperimentResult
from haystack_integrations.agent_pack.optimization.prompts import OPTIMIZER_PROMPT_CACHE_KEY


def build_bm25_retriever(store: DocumentStore, top_k: int = 5) -> Any:
    """
    Build the matching BM25 retriever for a document store.

    :param store: The store to retrieve from.
    :param top_k: How many documents one retrieval returns.
    :returns: The retriever.
    """
    if isinstance(store, InMemoryDocumentStore):
        return InMemoryBM25Retriever(document_store=store, top_k=top_k)
    opensearch_import.check()
    # Narrowed for the type checker: anything that is not the in-memory store is the OpenSearch one here.
    return OpenSearchBM25Retriever(document_store=cast("OpenSearchDocumentStore", store), top_k=top_k)


def preview(text: str, limit: int) -> str:
    """
    Collapse text to one line and cut it, marking the cut so a reader knows there is more.

    :param text: The text to preview.
    :param limit: How many characters to keep.
    :returns: The preview, ending in an ellipsis when anything was cut.
    """
    collapsed = " ".join((text or "").split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[:limit]}..."


def quiet_hub_warnings() -> None:
    """
    Stop the Hugging Face Hub's unauthenticated-request notice from interleaving with a walkthrough's output.

    The corpus is public and cached after the first run, so the notice says nothing a reader can act on. It is
    written straight to the standard error descriptor rather than logged or warned, so neither a logging filter
    nor a `sys.stderr` replacement reaches it; the descriptor itself is routed through a pipe and everything but
    that one line is passed on unchanged.
    """
    noise = b"unauthenticated requests to the HF Hub"
    original = os.dup(2)
    reading, writing = os.pipe()
    os.dup2(writing, 2)
    os.close(writing)

    def forward() -> None:
        with os.fdopen(reading, "rb") as incoming:
            for line in incoming:
                if noise not in line:
                    os.write(original, line)

    pump = threading.Thread(target=forward, daemon=True)
    pump.start()

    def restore() -> None:
        """Put the descriptor back, so a final traceback is not lost to a daemon thread on the way out."""
        sys.stderr.flush()
        os.dup2(original, 2)
        pump.join(timeout=2.0)

    atexit.register(restore)


def build_optimizer_generator(model: str | None) -> OpenAIResponsesChatGenerator | None:
    """Build the optimizer's generator with the default settings on another model, or return None for the default."""
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


# Colour is worth having on a terminal and only noise in a redirected log.
_STYLE = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "off": "\033[0m",
}


def paint(text: str, *names: str) -> str:
    """
    Apply terminal styles, or none when stdout is redirected.

    :param text: The text to style.
    :param names: Style names to apply.
    :returns: The text, wrapped in escape codes only when a terminal is reading them.
    """
    if not sys.stdout.isatty():
        return text
    return "".join(_STYLE[name] for name in names) + text + _STYLE["off"]


def rule(label: str) -> str:
    """
    Draw a labelled separator the width of the terminal.

    :param label: What the section below the rule is.
    :returns: The rule, padded to the terminal width.
    """
    width = min(shutil.get_terminal_size(fallback=(100, 24)).columns, 110)
    return paint(f"── {label} " + "─" * max(width - len(label) - 6, 4), "dim")


def delta(current: float, baseline: float, *, digits: int = 2, prefix: str = "", more_is_better: bool = True) -> str:
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
    return paint(f"  ({'+' if change > 0 else '-'}{rendered})", "green" if improved else "red")


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
            entries.append(paint(f"+ {key}", "green") + (f" {now}" if now else ""))
        else:
            entries.append(paint(f"- {key}", "red"))
    if skipped_prose:
        entries.append(paint("prompt text", "dim"))
    if not entries:
        return paint("whitespace and formatting only", "dim")
    shown = "  ·  ".join(entries[:limit])
    return shown if len(entries) <= limit else f"{shown}  ·  {paint(f'+{len(entries) - limit} more', 'dim')}"


class ExperimentPrinter:
    """Print an optimization experiment as it runs: the baseline, each candidate as it is measured, and the outcome."""

    def __init__(
        self,
        *,
        quality_metric: str,
        quality_label: str,
        detail_lines: Callable[[EvalMetrics, EvalMetrics | None], list[str]] | None = None,
    ) -> None:
        """
        Create a printer for one experiment.

        :param quality_metric: The key in `EvalMetrics.details` the experiment treats as quality.
        :param quality_label: What to call that quality when printing it, such as `recall@k`.
        :param detail_lines: Extra lines for one measurement, given the measurement and the baseline (`None` for the
            baseline itself), for whatever the harness evaluator reports beyond quality.
        """
        self.quality_metric = quality_metric
        self.quality_label = quality_label
        self.detail_lines = detail_lines

    def _measurement(
        self,
        metrics: EvalMetrics,
        cost: float | None,
        baseline: EvalMetrics | None = None,
        baseline_cost: float | None = None,
    ) -> list[str]:
        """
        Render one measurement as aligned lines, against the baseline when there is one.

        :param metrics: The measurement to render.
        :param cost: Its cost, or None when it could not be priced.
        :param baseline: The reference measurement, or None for the reference itself.
        :param baseline_cost: The reference's cost, or None when it could not be priced.
        :returns: The lines to print.
        """
        quality = metrics.details[self.quality_metric]
        shown = f"{quality:.2f}" + (delta(quality, baseline.details[self.quality_metric]) if baseline else "")
        spend = format_cost(cost=cost)
        if cost is not None and baseline_cost is not None:
            spend += delta(cost, baseline_cost, digits=4, prefix="$", more_is_better=False)
        duration = f"{median(metrics.durations):.1f}s" + (
            delta(median(metrics.durations), median(baseline.durations), digits=1, more_is_better=False)
            if baseline
            else ""
        )
        return [
            f"    {paint('quality ', 'bold')} {self.quality_label} {shown}",
            f"    {paint('spend   ', 'bold')} {spend}   median duration {duration}",
            *(self.detail_lines(metrics, baseline) if self.detail_lines else []),
        ]

    def on_baseline(self, metrics: EvalMetrics, cost: float | None) -> None:
        """
        Print the reference measurement every candidate is ranked against.

        :param metrics: The reference measurement.
        :param cost: Its cost, or None when it could not be priced.
        """
        print()
        print(rule("baseline · the reference"))
        for line in self._measurement(metrics=metrics, cost=cost):
            print(line)

    def on_candidate(self, progress: CandidateProgress) -> None:
        """
        Print one candidate as the search measures it, so a long run can be watched.

        :param progress: The measured candidate, its gates, and whether it now leads.
        """
        evaluation = progress.evaluation
        print()
        print(rule(f"candidate {progress.position}/{progress.total} · {evaluation.candidate_id}"))
        if evaluation.configuration is not None:
            print(f"    {paint('why     ', 'bold')} {evaluation.configuration.rationale}")
            print(f"    {paint('changed ', 'bold')} {_changes(diff=evaluation.configuration.diff)}")
        if evaluation.metrics is None:
            print(f"    {paint('verdict ', 'bold')} {paint('did not run: ' + str(evaluation.failure), 'red')}")
            return
        for line in self._measurement(
            metrics=evaluation.metrics,
            cost=evaluation.cost,
            baseline=progress.baseline,
            baseline_cost=progress.baseline_cost,
        ):
            print(line)
        verdict = (
            paint("cleared every gate", "green")
            if not progress.gate_failures
            else paint("gated: " + ", ".join(progress.gate_failures), "red")
        )
        if progress.is_best:
            verdict += paint("  ← best so far", "cyan", "bold")
        print(f"    {paint('verdict ', 'bold')} {verdict}")

    def report(self, result: ExperimentResult) -> None:
        """
        Print what the search cost and what it recommends, the candidates having printed as they were measured.

        :param result: What the experiment returned.
        """
        print()
        print(rule("what the search itself cost"))
        usage = ", ".join(
            f"{model}: {tokens.input_tokens} in / {tokens.output_tokens} out"
            for model, tokens in result.optimizer_usage.items()
        )
        print(f"    {paint('optimizer', 'bold')} {usage or 'none recorded'}")
        print(
            f"    {paint('priced at', 'bold')} {format_cost(cost=result.optimizer_cost)} (input charged at full price)"
        )

        print()
        print(rule("recommendation"))
        if result.recommendation is None:
            print(f"    {paint('none', 'red')}: no candidate cleared every gate and improved on the reference")
            return
        recommendation = result.recommendation
        measured = recommendation.evaluation.metrics
        if measured is not None:
            quality = measured.details[self.quality_metric]
            print(
                f"    {paint('candidate', 'bold')} {recommendation.evaluation.candidate_id}   "
                f"{self.quality_label} {quality:.2f}{delta(quality, result.baseline.details[self.quality_metric])}   "
                f"cost {format_cost(cost=recommendation.evaluation.cost)}"
            )
        print(f"    {paint('why      ', 'bold')} {recommendation.configuration.rationale}")
        print(f"    {paint('reasons  ', 'bold')} {', '.join(recommendation.reasons)}")
        print(f"    {paint('config   ', 'bold')} {result.artifact_directory / 'recommended.yaml'}")
        print()
        print(
            f"    {paint('Nothing was deployed. Approving this recommendation is a separate, human decision.', 'dim')}"
        )


class _Narrate(logging.Filter):
    """Drop what this script now prints itself, and shorten what it keeps to what a reader is watching for."""

    def filter(self, record: logging.LogRecord) -> bool:
        # The candidate block prints all of this, together with the change and rationale that produced it.
        if hasattr(record, "candidate_id") and hasattr(record, "gates"):
            return False
        # The search cost is reported once at the end.
        if hasattr(record, "turns") and hasattr(record, "cost"):
            return False
        # Each measurement is summarized in its own block, so the eval-case-by-eval-case progress is noise here.
        if hasattr(record, "verdict") and hasattr(record, "question"):
            return False
        if hasattr(record, "steps") and hasattr(record, "calls"):
            record.msg = paint(f"optimizer · {record.steps} steps · {' → '.join(record.calls)}", "dim")
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
