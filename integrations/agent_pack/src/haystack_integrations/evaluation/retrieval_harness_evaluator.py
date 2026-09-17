# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from dataclasses import asdict, dataclass
from statistics import median_low
from typing import Any

from haystack import Document, Pipeline, logging, tracing

from .dataclasses import EvalMetrics, ModelTokenUsage, RetrievalEvalCase
from .harness_log_collector import HarnessLogCollector
from .tracer import EVAL_CASE_SPAN, EvalCaseSummary, HarnessSpan, HarnessTracer

logger = logging.getLogger(__name__)

QUERY_SOCKET = "query"
DOCUMENTS_SOCKET = "documents"


@dataclass(kw_only=True)
class RetrievalEvalCaseMetrics:
    """
    Score for one retrieval eval case.

    :param question: The question that was posed.
    :param passed: Whether the run met every expectation, which is true exactly when `failures` is empty.
    :param score: What quality aggregates over eval cases, which is recall@k rather than whether the eval case
        passed. Recall over a handful of expected documents moves in steps of a half or a third, so a threshold
        on it reports a configuration that went from finding none of the evidence to two thirds of it as no
        change at all.
    :param stage_outputs: How many items each component emitted, by component name and output socket.
    :param stage_texts: A capped sample of whatever each component emitted as text, by component name and
        output socket. For a pipeline that rewrites the question, this is what it actually asked the store,
        which is usually what explains a recall failure rather than how many queries there were.
    :param failures: Every expectation the run missed, named.
    :param recall_at_k: The share of the expected documents found within the first `k` returned.
    :param precision_at_k: The share of the first `k` returned that were expected.
    :param retrieved: Everything the run returned, which is separate from how deep it was scored: returning more
        than `k` is not a fault, it simply earns nothing for the documents past the cutoff.
    :param missed_document_ids: The expected documents the run did not return, sorted.
    :param latency_ms: Measured wall-clock duration of the run.
    """

    question: str
    passed: bool
    score: float
    stage_outputs: dict[str, dict[str, int]]
    stage_texts: dict[str, dict[str, list[str]]]
    failures: tuple[str, ...]
    recall_at_k: float
    precision_at_k: float
    retrieved: int
    missed_document_ids: tuple[str, ...]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        data = asdict(self)
        for key in ("failures", "missed_document_ids"):
            data[key] = list(getattr(self, key))
        return data


def _query_entry_points(pipeline: Pipeline) -> set[str]:
    """
    Find every component with an unconnected `query` input the question should be posed to.

    :param pipeline: The pipeline to inspect.
    :returns: The names of components awaiting the question.
    """
    return {name for name, sockets in pipeline.inputs().items() if QUERY_SOCKET in sockets}


def _documents_exit_point(pipeline: Pipeline) -> str:
    """
    Find the component whose `documents` output is what the pipeline retrieved.

    :param pipeline: The pipeline to inspect.
    :returns: The name of the component producing the final documents.
    :raises ValueError: If no component, or more than one, exposes a `documents` output.
    """
    producers = [name for name, sockets in pipeline.outputs().items() if DOCUMENTS_SOCKET in sockets]
    if len(producers) != 1:
        msg = (
            f"The pipeline must end in exactly one unconnected {DOCUMENTS_SOCKET!r} output, so the harness knows "
            f"what was retrieved; found {sorted(producers)}."
        )
        raise ValueError(msg)
    return producers[0]


def _stage_output_sizes(eval_metrics: list[RetrievalEvalCaseMetrics]) -> dict[str, dict[str, dict[str, int]]]:
    """
    Summarize how many items each component emitted.

    We report the range of output sizes from a component's output socket and the "low median" of the sizes it produced.
    The low median is used since it's a better summarization measure on non-normal distributions.

    :param eval_metrics: List of `RetrievalEvalCaseMetrics` objects to summarize.
    :returns: The smallest, typical and largest output size, by component name and output socket.
    """
    sizes: dict[str, dict[str, list[int]]] = {}
    for metric in eval_metrics:
        for component, sockets in metric.stage_outputs.items():
            for socket, size in sockets.items():
                sizes.setdefault(component, {}).setdefault(socket, []).append(size)
    return {
        component: {
            socket: {"min": min(seen), "median": median_low(seen), "max": max(seen)} for socket, seen in sockets.items()
        }
        for component, sockets in sizes.items()
    }


def _score_retrieval_result(
    result: dict[str, Any],
    eval_case: RetrievalEvalCase,
    *,
    exit_point: str,
    k: int | None = None,
    latency_ms: float,
    stage_outputs: dict[str, dict[str, int]] | None = None,
    stage_texts: dict[str, dict[str, list[str]]] | None = None,
) -> RetrievalEvalCaseMetrics:
    """
    Score one retrieval run against its labelled evidence.

    :param result: What `Pipeline.run_async` returned.
    :param eval_case: The expectations to score against.
    :param exit_point: Component whose documents are what the pipeline retrieved.
    :param k: Rank cutoff the run is scored at, or `None` to score everything it returned.
    :param latency_ms: Measured wall-clock duration of the run.
    :param stage_outputs: How many items each component emitted, by component name and output socket.
    :param stage_texts: A capped sample of whatever each component emitted as text.
    :returns: The score, naming every expectation the run missed.
    """
    retrieved = (result.get(exit_point) or {}).get(DOCUMENTS_SOCKET) or []
    returned_ids = [document.id for document in retrieved if isinstance(document, Document)]
    found = eval_case.found_at(document_ids=returned_ids, k=k)
    recall_at_k = eval_case.recall_at(document_ids=returned_ids, k=k)
    precision_at_k = eval_case.precision_at(document_ids=returned_ids, k=k)

    failures: list[str] = []
    if recall_at_k < eval_case.min_recall:
        failures.append(f"recall_below_{eval_case.min_recall:g}")
    if precision_at_k < eval_case.min_precision:
        failures.append(f"precision_below_{eval_case.min_precision:g}")

    return RetrievalEvalCaseMetrics(
        question=eval_case.question,
        passed=not failures,
        stage_outputs=stage_outputs or {},
        stage_texts=stage_texts or {},
        score=recall_at_k,
        failures=tuple(failures),
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        retrieved=len(set(returned_ids)),
        missed_document_ids=tuple(sorted(eval_case.expected_document_ids - found)),
        latency_ms=latency_ms,
    )


class RetrievalHarnessEvaluator:
    """Pose every eval case's question to a retrieval pipeline and score what came back."""

    def __init__(self, *, k: int | None = None, max_concurrent_eval_cases: int = 1) -> None:
        """
        Create an evaluator.

        :param k: Rank cutoff every eval case is scored at, giving recall@k and precision@k. Only the first
            `k` documents a run returns count, in the order it ranked them, so a pipeline is measured on what
            it put at the top rather than on how much it returned. `None` scores everything returned.
        :param max_concurrent_eval_cases: How many eval cases to measure at once. Eval cases are independent and each
        spends its
            time waiting on a model, so this decides wall-clock time rather than cost. Leave it at 1 when ranking
            by latency, or the objective measures contention rather than the configuration.
        :raises ValueError: If `max_concurrent_eval_cases` is below one.
        """
        if max_concurrent_eval_cases < 1:
            msg = "max_concurrent_eval_cases must be at least 1."
            raise ValueError(msg)
        self.k = k
        self.max_concurrent_eval_cases = max_concurrent_eval_cases

    def validate(self, target: Pipeline) -> None:
        """
        Validate the candidate pipeline to expose at least one `query` input and exactly one `documents` output.

        :param target: Candidate pipeline deserialized from YAML.
        :raises ValueError: If the pipeline exposes no `query` input, or not exactly one `documents` output.
        """
        if not _query_entry_points(pipeline=target):
            msg = f"The pipeline must expose at least one unconnected {QUERY_SOCKET!r} input to receive the question."
            raise ValueError(msg)
        _documents_exit_point(pipeline=target)

    async def _measure(
        self, target: Pipeline, eval_cases: list[RetrievalEvalCase]
    ) -> list[tuple[RetrievalEvalCaseMetrics, EvalCaseSummary]]:
        """
        Measure every eval case, running up to `max_concurrent_eval_cases` of them at once.

        :param target: The candidate pipeline to measure.
        :param eval_cases: The labelled expectations to pose.
        :returns: One result per eval case, in the order the eval cases were given.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_eval_cases)
        exit_point = _documents_exit_point(pipeline=target)
        entry_points = _query_entry_points(pipeline=target)

        async def measure(
            position: int, eval_case: RetrievalEvalCase
        ) -> tuple[RetrievalEvalCaseMetrics, EvalCaseSummary]:
            """Pose one question once a slot is free."""
            data = {name: {QUERY_SOCKET: eval_case.question} for name in entry_points}
            async with semaphore:
                started = time.perf_counter()
                with tracing.tracer.trace(
                    EVAL_CASE_SPAN, tags={"haystack.harness.eval_case.question": eval_case.question}
                ) as span:
                    result = await target.run_async(data=data)
                # An empty summary when a HarnessTracer was not the active tracer.
                eval_case_summary = (
                    span.collected.summarize()
                    if isinstance(span, HarnessSpan) and span.collected is not None
                    else EvalCaseSummary()
                )
            latency_ms = (time.perf_counter() - started) * 1000
            eval_case_metrics = _score_retrieval_result(
                result=result,
                eval_case=eval_case,
                exit_point=exit_point,
                k=self.k,
                latency_ms=latency_ms,
                stage_outputs=dict(eval_case_summary.outputs),
                stage_texts=dict(eval_case_summary.texts),
            )
            logger.info(
                "eval case {position}/{total} {verdict} in {latency:.0f}ms: {question}",
                position=position,
                total=len(eval_cases),
                verdict="passed" if eval_case_metrics.passed else f"FAILED ({', '.join(eval_case_metrics.failures)})",
                latency=latency_ms,
                question=eval_case.question[:80],
            )
            return eval_case_metrics, eval_case_summary

        return list(
            await asyncio.gather(*(measure(index, eval_case) for index, eval_case in enumerate(eval_cases, start=1)))
        )

    def evaluate(self, target: Pipeline, eval_cases: list[RetrievalEvalCase]) -> EvalMetrics:
        """
        Pose every eval case to the pipeline and return raw experiment metrics, from synchronous code.

        :param target: The materialized candidate pipeline to score.
        :param eval_cases: The labelled expectations to score it against.
        :returns: What `evaluate_async` measured.
        :raises RuntimeError: If an event loop is already running; await `evaluate_async` from inside one.
        """
        return asyncio.run(self.evaluate_async(target=target, eval_cases=eval_cases))

    async def evaluate_async(self, target: Pipeline, eval_cases: list[RetrievalEvalCase]) -> EvalMetrics:
        """
        Pose every eval case to the pipeline and return raw experiment metrics.

        :param target: The materialized candidate pipeline to score.
        :param eval_cases: The labelled expectations to score it against.
        :returns: Fraction of eval cases passed, raw model usage, and mean latency, with per-eval-case detail.
        :raises ValueError: If no eval cases were supplied, leaving nothing to score.
        """
        if not eval_cases:
            msg = "The retrieval evaluator was given no eval cases to score."
            raise ValueError(msg)

        # Pre-warm up the pipeline to avoid cold-start latency
        await target.warm_up_async()

        # Run the evaluation with a HarnessTracer and log collector to capture diagnostics and run-time information
        tracer = HarnessTracer()
        with HarnessLogCollector().collect() as diagnostics, tracer.activate():
            measured = await self._measure(target=target, eval_cases=eval_cases)

        # Extract the evaluation metrics
        eval_metrics = [metric for metric, _ in measured]

        # Aggregate model usage across all measured eval cases
        model_usage: dict[str, ModelTokenUsage] = {}
        for _, summary in measured:
            for model, tokens in summary.models.items():
                current = model_usage.get(model, ModelTokenUsage())
                model_usage[model] = ModelTokenUsage(
                    input_tokens=current.input_tokens + tokens.input_tokens,
                    output_tokens=current.output_tokens + tokens.output_tokens,
                )

        return EvalMetrics(
            quality=sum(metric.score for metric in eval_metrics) / len(eval_metrics),
            latency_ms=sum(metric.latency_ms for metric in eval_metrics) / len(eval_metrics),
            model_usage=model_usage,
            eval_cases=[metric.to_dict() for metric in eval_metrics],
            details={
                "all_tokens_reported": all(summary.all_tokens_reported for _, summary in measured),
                "mean_recall_at_k": sum(metric.recall_at_k for metric in eval_metrics) / len(eval_metrics),
                "mean_precision_at_k": sum(metric.precision_at_k for metric in eval_metrics) / len(eval_metrics),
                "mean_retrieved": sum(metric.retrieved for metric in eval_metrics) / len(eval_metrics),
                # Report the range and low median of the output sizes each component produced. Useful for understanding
                # intermediate stages of a pipeline.
                "stage_output_sizes": _stage_output_sizes(eval_metrics=eval_metrics),
                # Report any warnings from the logger that were emitted during the evaluation
                "warnings": diagnostics.to_list(),
            },
        )
