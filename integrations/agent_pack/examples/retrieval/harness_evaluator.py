# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from typing import Any

from haystack import Document, Pipeline, logging

from haystack_integrations.agent_pack.dataclasses import EvaluationMetrics, ModelTokenUsage, RunRecord
from haystack_integrations.agent_pack.evaluation.component_logs import ComponentLogCollector
from haystack_integrations.tracing.agent_pack.tracer import EvalCaseUsage, HarnessTracer
from retrieval.evaluation import (
    RetrievalCaseMetrics,
    RetrievalEvaluationCase,
    RetrievalOutcome,
    score_retrieval_result,
)

logger = logging.getLogger(__name__)

QUERY_SOCKET = "query"
DOCUMENTS_SOCKET = "documents"
QUERIES_SOCKET = "queries"


def question_from_run(record: RunRecord) -> str:
    """
    Return the question a reference run replays.

    :param record: The recorded reference run.
    :returns: The recorded question.
    :raises ValueError: If the run carries no question.
    """
    question = record.inputs.get(QUERY_SOCKET)
    if not isinstance(question, str) or not question:
        msg = f"Run {record.run_id} does not contain a replayable {QUERY_SOCKET!r} string."
        raise ValueError(msg)
    return question


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


def query_reporters(pipeline: Pipeline) -> set[str]:
    """
    Find components whose `queries` output records what was actually asked.

    These are read for evidence only. A pipeline that expands nothing simply reports the original question.

    :param pipeline: The pipeline to inspect.
    :returns: The names of components producing a `queries` output.
    """
    reporters = set()
    for name in pipeline.graph.nodes:
        sockets = getattr(pipeline.get_component(name), "__haystack_output__", None)
        if sockets is not None and QUERIES_SOCKET in sockets._sockets_dict:
            reporters.add(name)
    return reporters


def _mean_stage_outputs(scored: list[RetrievalCaseMetrics]) -> dict[str, dict[str, float]]:
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


class RetrievalHarnessEvaluator:
    """
    Replay the recorded question of every eval case through a retrieval pipeline and score what came back.

    Nothing here names a component. The pipeline under measurement is the thing being optimized, so an optimizer is
    free to rename `retriever`, insert a ranker, or replace the retrieval path entirely; this finds where to put the
    question and where to read documents by socket name, and says so at validation time when it cannot.

    Quality is the mean of the per-eval-case scores, normalized to `[0.0, 1.0]`. A case scores its recall@k, so a
    configuration that finds more of the evidence is measured as better even while no case yet finds all of it.
    `passed` is still reported per eval case, and remains the stricter reading.
    No answer is generated: the labelled evidence names the documents an answer needs, which is what makes one
    case cost a single model call rather than an agent loop.
    """

    def __init__(
        self,
        *,
        cases: list[RetrievalEvaluationCase],
        max_concurrent_cases: int = 1,
        max_reported_queries: int = 8,
    ) -> None:
        """
        Create an evaluator.

        :param cases: Labelled expectations, keyed internally by question.
        :param max_concurrent_cases: How many cases to measure at once. Cases are independent and each spends its
            time waiting on a model, so this decides wall-clock time rather than cost. Leave it at 1 when ranking
            by latency, or the objective measures contention rather than the configuration.
        :param max_reported_queries: How many issued queries to report per eval case. A configuration that expands
            without limit would otherwise put its whole expansion into the optimizer's context.
        :raises ValueError: If `cases` is empty or `max_concurrent_cases` is below one.
        """
        if not cases:
            msg = "The retrieval evaluator needs at least one labelled case."
            raise ValueError(msg)
        if max_concurrent_cases < 1:
            msg = "max_concurrent_cases must be at least 1."
            raise ValueError(msg)
        self.cases = {case.question: case for case in cases}
        self.max_concurrent_cases = max_concurrent_cases
        self.max_reported_queries = max_reported_queries

    def validate_pipeline(self, target: Pipeline) -> None:
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

    def fingerprint(self) -> dict[str, Any]:
        """
        Describe the evaluation set so an experiment journal is invalidated when it changes.

        :returns: Every configured case, ordered by question.
        """
        return {"cases": sorted((case.to_dict() for case in self.cases.values()), key=lambda entry: entry["question"])}

    def _outcome(self, result: dict[str, Any], exit_point: str, reporters: set[str], question: str) -> RetrievalOutcome:
        """
        Read the documents and the issued queries out of one pipeline result.

        :param result: What `Pipeline.run_async` returned.
        :param exit_point: Component whose documents were retrieved.
        :param reporters: Components that report the queries they issued.
        :param question: The original question, reported when nothing expanded it.
        :returns: The run outcome.
        """
        documents = [
            document
            for document in (result.get(exit_point) or {}).get(DOCUMENTS_SOCKET) or []
            if isinstance(document, Document)
        ]
        issued: list[str] = []
        for name in sorted(reporters):
            issued.extend(str(query) for query in (result.get(name) or {}).get(QUERIES_SOCKET) or [])
        return RetrievalOutcome(documents=documents, queries=tuple(issued or [question]))

    async def _measure(
        self, target: Pipeline, resolved: list[RetrievalEvaluationCase], tracer: HarnessTracer
    ) -> list[tuple[RetrievalCaseMetrics, EvalCaseUsage]]:
        """
        Measure every case, running up to `max_concurrent_cases` of them at once.

        :param target: The candidate pipeline to measure.
        :param resolved: The eval cases to pose.
        :param tracer: Collector for per-eval-case generator usage.
        :returns: One result per eval case, in the order the eval cases were given.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_cases)
        exit_point = documents_exit_point(pipeline=target)
        reporters = query_reporters(pipeline=target)
        entry_points = query_entry_points(pipeline=target)

        async def measure(position: int, case: RetrievalEvaluationCase) -> tuple[RetrievalCaseMetrics, EvalCaseUsage]:
            """Pose one question once a slot is free."""
            data = {name: {QUERY_SOCKET: case.question} for name in entry_points}
            async with semaphore:
                started = time.perf_counter()
                with tracer.case() as usage:
                    result = await target.run_async(data=data, include_outputs_from=reporters | {exit_point})
            latency_ms = (time.perf_counter() - started) * 1000
            outcome = self._outcome(result=result, exit_point=exit_point, reporters=reporters, question=case.question)
            scored = score_retrieval_result(
                outcome=outcome, case=case, latency_ms=latency_ms, stage_outputs=dict(usage.outputs)
            )
            logger.info(
                "case {position}/{total} {verdict} in {latency:.0f}ms with {queries} queries: {question}",
                position=position,
                total=len(resolved),
                verdict="passed" if scored.passed else f"FAILED ({', '.join(scored.failures)})",
                latency=latency_ms,
                queries=len(outcome.queries),
                question=case.question[:80],
            )
            return scored, usage

        return list(await asyncio.gather(*(measure(index, case) for index, case in enumerate(resolved, start=1))))

    def evaluate(self, target: Pipeline, reference_runs: list[RunRecord]) -> EvaluationMetrics:
        """
        Replay every selected run through the pipeline and return raw experiment metrics.

        :param target: The materialized candidate pipeline to score.
        :param reference_runs: The successful runs supplying the questions to replay.
        :returns: Fraction of cases passed, raw model usage, and mean latency, with per-eval-case detail.
        :raises ValueError: If a recorded question has no labelled case.
        """
        resolved = []
        for record in reference_runs:
            question = question_from_run(record=record)
            case = self.cases.get(question)
            if case is None:
                msg = f"No labelled retrieval case for question {question!r}."
                raise ValueError(msg)
            resolved.append(case)

        tracer = HarnessTracer()
        target.warm_up()
        with ComponentLogCollector().collect() as diagnostics, tracer.activate():
            measured = asyncio.run(self._measure(target=target, resolved=resolved, tracer=tracer))

        scored = [metric for metric, _ in measured]
        model_usage: dict[str, ModelTokenUsage] = {}
        for _, usage in measured:
            for model, tokens in usage.models.items():
                current = model_usage.get(model, ModelTokenUsage())
                model_usage[model] = ModelTokenUsage(
                    input_tokens=current.input_tokens + tokens.input_tokens,
                    output_tokens=current.output_tokens + tokens.output_tokens,
                )
        cases = [metric.to_dict() for metric in scored]
        for case_detail in cases:
            case_detail["queries"] = case_detail["queries"][: self.max_reported_queries]
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
                "mean_queries": sum(len(metric.queries) for metric in scored) / len(scored),
                # How much reached each stage. A candidate set is pooled from several searches and deduplicated,
                # so its size follows from no configuration value and only measurement reports where the path
                # actually narrows.
                "mean_stage_outputs": _mean_stage_outputs(scored=scored),
                # What the components said about themselves. A component that degrades rather than failing keeps
                # the run alive and reports it only here, so a score with no explanation gets one.
                "warnings": diagnostics.to_list(),
                "cases": cases,
            },
        )
