# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from typing import Any

from haystack import Document, Pipeline, logging

from haystack_integrations.agent_pack.dataclasses import EvaluationMetrics, ModelTokenUsage
from haystack_integrations.agent_pack.evaluation import RetrievalEvalCase
from haystack_integrations.agent_pack.evaluation.component_logs import ComponentLogCollector
from haystack_integrations.agent_pack.evaluation.harness_evaluator import HarnessEvaluator
from haystack_integrations.tracing.agent_pack.tracer import EvalCaseUsage, HarnessTracer
from retrieval.dataclasses import RetrievalEvalCaseMetrics

logger = logging.getLogger(__name__)

QUERY_SOCKET = "query"
DOCUMENTS_SOCKET = "documents"


def query_entry_points(pipeline: Pipeline) -> set[str]:
    """
    Find every component with an unconnected `query` input the question should be posed to.

    :param pipeline: The pipeline to inspect.
    :returns: The names of components awaiting the question.
    """
    return {name for name, sockets in pipeline.inputs().items() if QUERY_SOCKET in sockets}


def documents_exit_point(pipeline: Pipeline) -> str:
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


def _mean_stage_outputs(scored: list[RetrievalEvalCaseMetrics]) -> dict[str, dict[str, float]]:
    """
    Average how many items each component emitted, over the eval cases that reached it.

    :param scored: The measured eval cases.
    :returns: Mean output size by component name and output socket.
    """
    totals: dict[str, dict[str, list[int]]] = {}
    for metric in scored:
        for component, sockets in metric.stage_outputs.items():
            for socket, size in sockets.items():
                totals.setdefault(component, {}).setdefault(socket, []).append(size)
    return {
        component: {socket: sum(sizes) / len(sizes) for socket, sizes in sockets.items()}
        for component, sockets in totals.items()
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

    How the documents were fetched does not matter here: one query or twenty, the run is scored on what reached
    the end. What each stage in between emitted is counted by the tracer instead.

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


class RetrievalHarnessEvaluator(HarnessEvaluator):
    """
    Pose every eval case's question to a retrieval pipeline and score what came back.

    Nothing here names a component. The pipeline under measurement is the thing being optimized, so an optimizer is
    free to rename `retriever`, insert a ranker, or replace the retrieval path entirely; this finds where to put the
    question and where to read documents by socket name, and says so at validation time when it cannot.

    Quality is the mean of the per-eval-case scores, normalized to `[0.0, 1.0]`. An eval case scores its recall@k, so a
    configuration that finds more of the evidence is measured as better even while no eval case yet finds all of it.
    `passed` is still reported per eval case, and remains the stricter reading.
    No answer is generated: the labelled evidence names the documents an answer needs, which is what makes one
    eval case cost a single model call rather than an agent loop.
    """

    def __init__(
        self,
        *,
        eval_cases: list[RetrievalEvalCase],
        k: int | None = None,
        max_concurrent_eval_cases: int = 1,
    ) -> None:
        """
        Create an evaluator.

        :param eval_cases: Labelled expectations, keyed internally by question.
        :param k: Rank cutoff every eval case is scored at, giving recall@k and precision@k. Only the first
            `k` documents a run returns count, in the order it ranked them, so a pipeline is measured on what
            it put at the top rather than on how much it returned. `None` scores everything returned.
        :param max_concurrent_eval_cases: How many eval cases to measure at once. Eval cases are independent and each
        spends its
            time waiting on a model, so this decides wall-clock time rather than cost. Leave it at 1 when ranking
            by latency, or the objective measures contention rather than the configuration.
        :raises ValueError: If `eval cases` is empty or `max_concurrent_eval_cases` is below one.
        """
        if not eval_cases:
            msg = "The retrieval evaluator needs at least one labelled eval case."
            raise ValueError(msg)
        if max_concurrent_eval_cases < 1:
            msg = "max_concurrent_eval_cases must be at least 1."
            raise ValueError(msg)
        self.eval_cases = {eval_case.question: eval_case for eval_case in eval_cases}
        self.k = k
        self.max_concurrent_eval_cases = max_concurrent_eval_cases

    def validate(self, target: Pipeline) -> None:
        """
        Require the sockets the harness poses questions to and reads documents from.

        Checked before a candidate is measured, so a rewiring that the harness cannot drive costs a validation
        error the optimizer can repair rather than a whole measurement.

        :param target: Candidate pipeline deserialized from YAML.
        :raises ValueError: If the pipeline exposes no `query` input, or not exactly one `documents` output.
        """
        if not query_entry_points(pipeline=target):
            msg = f"The pipeline must expose at least one unconnected {QUERY_SOCKET!r} input to receive the question."
            raise ValueError(msg)
        documents_exit_point(pipeline=target)

    async def _measure(
        self, target: Pipeline, tracer: HarnessTracer
    ) -> list[tuple[RetrievalEvalCaseMetrics, EvalCaseUsage]]:
        """
        Measure every eval case, running up to `max_concurrent_eval_cases` of them at once.

        :param target: The candidate pipeline to measure.
        :param tracer: Collector for per-eval-case generator usage.
        :returns: One result per eval case, in the order the eval cases were given.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_eval_cases)
        exit_point = documents_exit_point(pipeline=target)
        entry_points = query_entry_points(pipeline=target)
        eval_cases = list(self.eval_cases.values())

        async def measure(
            position: int, eval_case: RetrievalEvalCase
        ) -> tuple[RetrievalEvalCaseMetrics, EvalCaseUsage]:
            """Pose one question once a slot is free."""
            data = {name: {QUERY_SOCKET: eval_case.question} for name in entry_points}
            async with semaphore:
                started = time.perf_counter()
                with tracer.eval_case() as usage:
                    result = await target.run_async(data=data)
            latency_ms = (time.perf_counter() - started) * 1000
            scored = _score_retrieval_result(
                result=result,
                eval_case=eval_case,
                exit_point=exit_point,
                k=self.k,
                latency_ms=latency_ms,
                stage_outputs=dict(usage.outputs),
                stage_texts=dict(usage.texts),
            )
            logger.info(
                "eval case {position}/{total} {verdict} in {latency:.0f}ms: {question}",
                position=position,
                total=len(eval_cases),
                verdict="passed" if scored.passed else f"FAILED ({', '.join(scored.failures)})",
                latency=latency_ms,
                question=eval_case.question[:80],
            )
            return scored, usage

        return list(
            await asyncio.gather(*(measure(index, eval_case) for index, eval_case in enumerate(eval_cases, start=1)))
        )

    def evaluate(self, target: Pipeline) -> EvaluationMetrics:
        """
        Pose every eval case to the pipeline and return raw experiment metrics, from synchronous code.

        :param target: The materialized candidate pipeline to score.
        :returns: What `evaluate_async` measured.
        :raises RuntimeError: If an event loop is already running; await `evaluate_async` from inside one.
        """
        return asyncio.run(self.evaluate_async(target=target))

    async def evaluate_async(self, target: Pipeline) -> EvaluationMetrics:
        """
        Pose every eval case to the pipeline and return raw experiment metrics.

        :param target: The materialized candidate pipeline to score.
        :returns: Fraction of eval cases passed, raw model usage, and mean latency, with per-eval-case detail.
        """
        tracer = HarnessTracer()
        await target.warm_up_async()
        with ComponentLogCollector().collect() as diagnostics, tracer.activate():
            measured = await self._measure(target=target, tracer=tracer)

        scored = [metric for metric, _ in measured]
        model_usage: dict[str, ModelTokenUsage] = {}
        for _, usage in measured:
            for model, tokens in usage.models.items():
                current = model_usage.get(model, ModelTokenUsage())
                model_usage[model] = ModelTokenUsage(
                    input_tokens=current.input_tokens + tokens.input_tokens,
                    output_tokens=current.output_tokens + tokens.output_tokens,
                )
        eval_cases = [metric.to_dict() for metric in scored]
        return EvaluationMetrics(
            quality=sum(metric.score for metric in scored) / len(scored),
            latency_ms=sum(metric.latency_ms for metric in scored) / len(scored),
            model_usage=model_usage,
            details={
                # A pipeline whose only model call is an expansion reports no usage at all when nothing expands,
                # which is a legitimate configuration rather than a broken measurement.
                "usage_complete": all(usage.complete for _, usage in measured),
                "mean_recall_at_k": sum(metric.recall_at_k for metric in scored) / len(scored),
                "mean_precision_at_k": sum(metric.precision_at_k for metric in scored) / len(scored),
                "mean_retrieved": sum(metric.retrieved for metric in scored) / len(scored),
                # How much reached each stage. A candidate set is pooled from several searches and deduplicated,
                # so its size follows from no configuration value and only measurement reports where the path
                # actually narrows.
                "mean_stage_outputs": _mean_stage_outputs(scored=scored),
                # What the components said about themselves. A component that degrades rather than failing keeps
                # the run alive and reports it only here, so a score with no explanation gets one.
                "warnings": diagnostics.to_list(),
                "eval_cases": eval_cases,
            },
        )
