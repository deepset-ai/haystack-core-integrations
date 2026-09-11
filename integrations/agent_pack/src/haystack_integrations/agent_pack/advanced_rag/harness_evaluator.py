# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from typing import Any

from haystack import logging, tracing
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import flatten_tools_or_toolsets

from haystack_integrations.agent_pack.advanced_rag.evaluation import (
    METADATA_TOOLS,
    RETRIEVAL_TOOLS,
    AdvancedRAGEvalCaseMetrics,
    score_advanced_rag_result,
)
from haystack_integrations.agent_pack.advanced_rag.tools import (
    GetMetadataFieldRangeTool,
    GetMetadataFieldValuesTool,
    ListMetadataFieldsTool,
)
from haystack_integrations.evaluation.agent_run_digest import AGENT_RUN_DIGEST_KEY, AgentRunDigestPolicy
from haystack_integrations.evaluation.dataclasses import (
    EvalMetrics,
    ModelTokenUsage,
    RAGEvalCase,
    ToolNames,
)
from haystack_integrations.evaluation.harness_log_collector import HarnessLogCollector
from haystack_integrations.evaluation.tool_budgets import resolve_tool_budgets
from haystack_integrations.evaluation.tracer import EVAL_CASE_SPAN, EvalCaseSummary, HarnessSpan, HarnessTracer

logger = logging.getLogger(__name__)


def widen_tool_budgets(
    budgets: dict[ToolNames, int], retrieval_tools: frozenset[str], metadata_tools: frozenset[str]
) -> dict[ToolNames, int]:
    """
    Extend an eval case's retrieval and metadata groups with the names this candidate uses for them.

    An eval case names the tools the reference had. A candidate can rename one or add another, and a budget
    written as `search_documents` would stop covering the retrieval the candidate actually does. Growing the
    group keeps one allowance over the whole job rather than handing the new tool an allowance of its own.

    :param budgets: What the eval case declared, as `{tool name or names: limit}`.
    :param retrieval_tools: Every name the candidate retrieves documents under.
    :param metadata_tools: Every name the candidate inspects metadata under.
    :returns: The same allowances, over groups grown to match the candidate.
    """
    widened: dict[ToolNames, int] = {}
    for tools, limit in budgets.items():
        names = {tools} if isinstance(tools, str) else set(tools)
        if names & retrieval_tools:
            names |= retrieval_tools
        if names & metadata_tools:
            names |= metadata_tools
        widened[tuple(sorted(names))] = limit
    return widened


class AdvancedRAGHarnessEvaluator:
    """
    Pose every labelled question to an Advanced RAG candidate and score what came back.

    Quality is the fraction of eval cases that pass, and is therefore normalized to `[0.0, 1.0]`. An eval case passes
    only when its retrieval, answer, citation, metadata-inspection, and tool-budget expectations pass. Each is measured
    once, so breadth across eval cases, rather than repeated measurement of a few, is what makes that fraction
    discriminating.
    """

    def __init__(
        self,
        *,
        digest_policy: AgentRunDigestPolicy | None = None,
        max_traced_eval_cases: int | None = 6,
        max_concurrent_eval_cases: int = 1,
    ) -> None:
        """
        Create an evaluator.

        :param digest_policy: Caps applied to the tool trace recorded for each eval case.
        :param max_traced_eval_cases: How many eval case traces to keep, or `None` to keep every one. A trace explains a
            result but a reader's history of them is cumulative, so failing eval cases keep theirs first: a passing eval
            case has nothing to diagnose.
        :param max_concurrent_eval_cases: How many eval cases to measure at once. Eval cases are independent and each
        one spends
            its time waiting on a model, so measuring several together is most of what makes an experiment
            affordable in wall-clock terms. Token usage, and therefore cost, is unaffected. Measured latency is
            not: concurrent runs contend for the same rate limits, so leave this at 1 when ranking by latency, or
            the objective measures this setting rather than the configuration. Eval cases are driven through
            `Agent.run_async` whatever this is set to, so one at a time is simply a concurrency of one.
        :raises ValueError: If `max_concurrent_eval_cases` is below one.
        """
        if max_concurrent_eval_cases < 1:
            msg = "max_concurrent_eval_cases must be at least 1."
            raise ValueError(msg)
        self.digest_policy = digest_policy
        self.max_concurrent_eval_cases = max_concurrent_eval_cases
        self.max_traced_eval_cases = max_traced_eval_cases

    def validate(self, target: Agent) -> None:
        """
        Require document state used for retrieval and citation scoring.

        :param target: Candidate Agent deserialized from YAML.
        :raises ValueError: If the Agent reports no documents for the harness to score.
        """
        if "documents" not in target.resolved_state_schema:
            msg = "The RAG evaluator requires a documents state output."
            raise ValueError(msg)

    def _traced_eval_cases(self, metrics: list[AdvancedRAGEvalCaseMetrics]) -> list[dict[str, Any]]:
        """
        Report every eval case, keeping tool traces for the ones worth diagnosing.

        :param metrics: Every scored eval case.
        :returns: JSON-compatible eval case records, with the trace dropped from eval cases beyond the cap.
        """
        eval_cases = [metric.to_dict() for metric in metrics]
        if self.max_traced_eval_cases is None:
            return eval_cases
        ranked = sorted(range(len(metrics)), key=lambda index: (metrics[index].passed, index))
        for index in ranked[self.max_traced_eval_cases :]:
            eval_cases[index].pop(AGENT_RUN_DIGEST_KEY, None)
        return eval_cases

    def _score(
        self,
        result: dict[str, Any],
        eval_case: RAGEvalCase,
        started: float,
        position: int,
        total: int,
        retrieval_tools: frozenset[str] = RETRIEVAL_TOOLS,
        metadata_tools: frozenset[str] = METADATA_TOOLS,
        tool_budgets: dict[tuple[str, ...], int] | None = None,
    ) -> AdvancedRAGEvalCaseMetrics:
        """
        Score one completed Agent run and report it.

        :param result: What `Agent.run` returned.
        :param eval_case: The expectations to score against.
        :param started: The `perf_counter` reading from before the run.
        :param position: Which eval case this is, for reporting.
        :param total: How many eval cases there are, for reporting.
        :param retrieval_tools: Names resolved from candidate document outputs.
        :param metadata_tools: Names resolved from metadata tool classes.
        :param tool_budgets: The eval case's allowances, resolved against the tools the candidate has.
        :returns: The eval case score.
        """
        latency_ms = (time.perf_counter() - started) * 1000
        scored = score_advanced_rag_result(
            result=result,
            eval_case=eval_case,
            latency_ms=latency_ms,
            digest_policy=self.digest_policy,
            retrieval_tools=retrieval_tools,
            metadata_tools=metadata_tools,
            tool_budgets=tool_budgets,
        )
        logger.info(
            "eval case {position}/{total} {verdict} in {latency:.0f}ms: {question}",
            position=position,
            total=total,
            verdict="passed" if scored.passed else f"FAILED ({', '.join(scored.failures)})",
            latency=latency_ms,
            question=eval_case.question[:80],
        )
        return scored

    async def _measure(
        self, agent: Agent, eval_cases: list[RAGEvalCase]
    ) -> list[tuple[AdvancedRAGEvalCaseMetrics, EvalCaseSummary]]:
        """
        Measure every eval case, running up to `max_concurrent_eval_cases` of them at once.

        :param agent: The candidate to measure.
        :param eval_cases: The labelled expectations to pose.
        :returns: One result per eval case, in the order the eval cases were given.
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_eval_cases)
        tools = flatten_tools_or_toolsets(getattr(agent, "tools", []))
        retrieval_tools = RETRIEVAL_TOOLS | frozenset(
            tool.name for tool in tools if "documents" in (tool.outputs_to_state or {})
        )
        metadata_tools = METADATA_TOOLS | frozenset(
            tool.name
            for tool in tools
            if isinstance(tool, (ListMetadataFieldsTool, GetMetadataFieldValuesTool, GetMetadataFieldRangeTool))
        )
        tool_names = [tool.name for tool in tools]

        async def measure(position: int, eval_case: RAGEvalCase) -> tuple[AdvancedRAGEvalCaseMetrics, EvalCaseSummary]:
            """Run one eval case, waiting for a slot first."""
            async with semaphore:
                started = time.perf_counter()
                with tracing.tracer.trace(
                    EVAL_CASE_SPAN, tags={"haystack.harness.eval_case.question": eval_case.question}
                ) as span:
                    result = await agent.run_async(messages=[ChatMessage.from_user(text=eval_case.question)])
                # An empty summary when a HarnessTracer was not the active tracer.
                usage = (
                    span.collected.summarize()
                    if isinstance(span, HarnessSpan) and span.collected is not None
                    else EvalCaseSummary()
                )
            scored = self._score(
                result=result,
                eval_case=eval_case,
                started=started,
                position=position,
                total=len(eval_cases),
                retrieval_tools=retrieval_tools,
                metadata_tools=metadata_tools,
                tool_budgets=resolve_tool_budgets(
                    budgets=widen_tool_budgets(
                        budgets=eval_case.tool_budgets, retrieval_tools=retrieval_tools, metadata_tools=metadata_tools
                    ),
                    tool_names=tool_names,
                ),
            )
            return scored, usage

        return list(
            await asyncio.gather(*(measure(position, eval_case) for position, eval_case in enumerate(eval_cases, 1)))
        )

    def evaluate(self, target: Agent, eval_cases: list[RAGEvalCase]) -> EvalMetrics:
        """
        Pose every eval case and return raw experiment metrics, from synchronous code.

        :param target: The materialized candidate Agent to score.
        :param eval_cases: The labelled expectations to score it against.
        :returns: What `evaluate_async` measured.
        :raises RuntimeError: If an event loop is already running; await `evaluate_async` from inside one.
        """
        return asyncio.run(self.evaluate_async(target=target, eval_cases=eval_cases))

    async def evaluate_async(self, target: Agent, eval_cases: list[RAGEvalCase]) -> EvalMetrics:
        """
        Pose every eval case and return raw experiment metrics.

        :param target: The materialized candidate Agent to score.
        :param eval_cases: The labelled expectations to score it against.
        :returns: Fraction of eval cases passed, raw model usage, and mean latency for the candidate, with
            per-eval-case detail.
        :raises ValueError: If no eval cases were supplied, leaving nothing to score.
        """
        if not eval_cases:
            msg = "The Advanced RAG evaluator was given no eval cases to score."
            raise ValueError(msg)

        tracer = HarnessTracer()
        await target.warm_up_async()
        with HarnessLogCollector().collect() as diagnostics, tracer.activate():
            measured = await self._measure(agent=target, eval_cases=eval_cases)

        flattened = [scored for scored, _ in measured]
        model_usage: dict[str, ModelTokenUsage] = {}
        for _, usage in measured:
            for model, tokens in usage.models.items():
                current = model_usage.get(model, ModelTokenUsage())
                model_usage[model] = ModelTokenUsage(
                    input_tokens=current.input_tokens + tokens.input_tokens,
                    output_tokens=current.output_tokens + tokens.output_tokens,
                )
        quality = sum(metric.passed for metric in flattened) / len(flattened)
        input_tokens = sum(usage.input_tokens for usage in model_usage.values())
        output_tokens = sum(usage.output_tokens for usage in model_usage.values())
        model_id = getattr(target.chat_generator, "model", None)
        return EvalMetrics(
            quality=quality,
            latency_ms=sum(metric.latency_ms for metric in flattened) / len(flattened),
            model_usage=model_usage,
            eval_cases=self._traced_eval_cases(metrics=flattened),
            details={
                "model": model_id,
                # A run that attributed nothing to a model is as unpriceable as one that under-reported.
                "all_tokens_reported": all(usage.all_tokens_reported and usage.models for _, usage in measured),
                "mean_recall": sum(metric.recall for metric in flattened) / len(flattened),
                "mean_precision": sum(metric.precision for metric in flattened) / len(flattened),
                # What the components said about themselves while they ran; a tool or hook that degrades rather
                # than failing reports it only here.
                "warnings": diagnostics.to_list(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        )
