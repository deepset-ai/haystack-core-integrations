# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from haystack import default_from_dict, default_to_dict, logging, tracing
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import flatten_tools_or_toolsets

from haystack_integrations.agent_pack.advanced_rag.tools import (
    GetMetadataFieldRangeTool,
    GetMetadataFieldValuesTool,
    ListMetadataFieldsTool,
)
from haystack_integrations.agent_pack.evaluation.agent_run_digest import (
    AGENT_RUN_DIGEST_KEY,
    AgentRunDigestPolicy,
    digest_agent_run,
)
from haystack_integrations.agent_pack.evaluation.dataclasses import (
    EvalMetrics,
    ModelTokenUsage,
    RetrievalEvalCase,
    ToolNames,
    ToolRunStats,
)
from haystack_integrations.agent_pack.evaluation.harness_log_collector import HarnessLogCollector
from haystack_integrations.agent_pack.evaluation.tool_budgets import budgets_exceeded, resolve_tool_budgets
from haystack_integrations.agent_pack.evaluation.tracer import (
    EVAL_CASE_SPAN,
    EvalCaseSummary,
    HarnessSpan,
    HarnessTracer,
)

logger = logging.getLogger(__name__)


RETRIEVAL_TOOLS = frozenset({"search_documents", "fetch_documents_by_filter"})
METADATA_TOOLS = frozenset({"list_metadata_fields", "get_metadata_field_values", "get_metadata_field_range"})

# Citation format produced by the Advanced RAG toolset: the first eight characters of a document ID.
CITATION_PATTERN = re.compile(r"\[doc ([0-9a-fA-F]{8})\]")


@dataclass(kw_only=True)
class AdvancedRAGEvalCaseMetrics:
    """
    Detailed score for one Advanced RAG evaluation case.

    `failures` names every expectation the run missed, so a regression report says what broke rather than only that
    something did. `passed` is true exactly when `failures` is empty. `agent_run_digest` records what the Agent actually
    did — every tool call with its arguments and result — so a failure can be diagnosed rather than only counted.
    `exit_reason` is what the counts hide: a run cut off by its step budget is answered by the backup-answer
    hook, which does not cite, so it fails a citation expectation for a reason that has nothing to do with
    retrieval.

    :param budgeted_recall: `cited_recall` when citations are required and `recall` when they are not, and 0.0
        for a run that spent more tool calls than its budget allowed, however much evidence it found. Recall over a
        handful of expected documents moves in steps of a half or a third, so this changes when a configuration
        finds more of the evidence even before it passes.
    :param recall: Share of the needed documents the run retrieved, whether or not the answer used them.
    :param cited_recall: Share of the needed documents the answer actually cited. Retrieving evidence and
        answering from it are different things, and only this one says the answer was grounded in it.
    """

    question: str
    passed: bool
    budgeted_recall: float
    failures: tuple[str, ...]
    recall: float
    precision: float
    cited_recall: float
    citations_resolved: bool
    cited_document_ids: tuple[str, ...]
    inspected_first: bool
    metadata_calls: int
    retrieval_calls: int
    filtered_retrieval_calls: int
    tool_errors: int
    steps: int
    duration: float
    exit_reason: str | None = None
    agent_run_digest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the AdvancedRAGEvalCaseMetrics into a dictionary.

        :returns: A dictionary with one key per field, with the failure and citation tuples as lists.
        """
        data = asdict(self)
        data["failures"] = list(self.failures)
        data["cited_document_ids"] = list(self.cited_document_ids)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdvancedRAGEvalCaseMetrics":
        """
        Create a new AdvancedRAGEvalCaseMetrics object from a dictionary.

        :param data: The dictionary to build the metrics from.
        :returns: The created object.
        """
        arguments = dict(data)
        arguments["failures"] = tuple(arguments.get("failures") or ())
        arguments["cited_document_ids"] = tuple(arguments.get("cited_document_ids") or ())
        return cls(**arguments)


def _score_advanced_rag_result(
    result: dict[str, Any],
    eval_case: RetrievalEvalCase,
    *,
    duration: float,
    min_recall: float = 1.0,
    min_precision: float = 0.0,
    require_citations: bool = True,
    max_tool_errors: int = 0,
    tool_budgets: dict[tuple[str, ...], int] | None = None,
    digest_policy: AgentRunDigestPolicy | None = None,
    retrieval_tools: frozenset[str] = RETRIEVAL_TOOLS,
    metadata_tools: frozenset[str] = METADATA_TOOLS,
) -> AdvancedRAGEvalCaseMetrics:
    """
    Score retrieval grounding, answer behaviour, and process budgets for one Agent result.

    :param result: The dictionary returned by `Agent.run`.
    :param eval_case: The expectations to score the result against.
    :param duration: How long the run took, in seconds.
    :param min_recall: Share of the needed documents the answer must be grounded in to pass.
    :param min_precision: Share of what the run retrieved that must be needed for it to pass.
    :param require_citations: Whether the answer is scored on what it cited rather than on what was retrieved,
        and must cite at least one retrieved document.
    :param max_tool_errors: How many failing tool calls the run may make and still pass.
    :param tool_budgets: Allowances already resolved against the tools the candidate has, or `None` for no budget.
    :param digest_policy: Caps applied to the recorded tool trace.
    :returns: The score, naming every expectation the run missed, and the trace explaining why.
    """
    messages = result.get("messages") or []
    stats = ToolRunStats.from_messages(messages=messages)
    retrieval_names, metadata_names = tuple(retrieval_tools), tuple(metadata_tools)
    # Reported rather than required: inspecting metadata first is good practice, not a correct answer.
    inspected_first = stats.called_before(tools=metadata_names, other=retrieval_names)
    metadata_calls = stats.calls_to(tools=metadata_names)
    retrieval_calls = stats.calls_to(tools=retrieval_names)
    filtered_retrieval_calls = stats.calls_with_argument(tools=retrieval_names, argument="filters")
    last_message = result.get("last_message")
    answer = (getattr(last_message, "text", None) or "") if last_message is not None else ""
    retrieved_documents = result.get("documents") or []
    # An Agent accumulates documents over several searches rather than ranking one list, so nothing is cut off.
    retrieved_ids = [document.id for document in retrieved_documents]
    recall = eval_case.recall_at(document_ids=retrieved_ids)
    precision = eval_case.precision_at(document_ids=retrieved_ids)

    cited_refs = tuple(CITATION_PATTERN.findall(answer))
    cited_ids = [
        document.id for document in retrieved_documents if any(document.id.startswith(ref) for ref in cited_refs)
    ]
    citations_resolved = all(
        any(document.id.startswith(reference) for document in retrieved_documents) for reference in cited_refs
    )
    cited_recall = eval_case.recall_at(document_ids=cited_ids)
    # Retrieving the evidence is not the same as answering from it. An eval case that asks for citations is scored
    # on what the answer actually cited; one that does not is scored on what the run retrieved.
    grounded_recall = cited_recall if require_citations else recall

    failures: list[str] = []
    # A tool-call budget is what an eval case will not pay past, so breaking one voids it rather than scoring it.
    # Everything else is reported as a failure and still scores what it found.
    over_budget: list[str] = []

    if grounded_recall < min_recall:
        failures.append(f"recall_below_{min_recall:g}")
    if precision < min_precision:
        failures.append(f"precision_below_{min_precision:g}")
    if not citations_resolved:
        failures.append("unresolvable_citation")
    if require_citations and retrieved_documents and not cited_refs:
        failures.append("answer_cites_nothing")

    # The label names the group and nothing else, so repeats of one problem count as one recurring failure
    # rather than as several distinct ones. How much was spent is on `retrieval_calls` and `tool_errors`.
    for group in budgets_exceeded(stats=stats, budgets=tool_budgets or {}):
        over_budget.append(f"tool_calls_over_budget:{'+'.join(group)}")
    if len(stats.errors) > max_tool_errors:
        failures.append("tool_errors")
    failures.extend(over_budget)

    steps = int(result.get("step_count") or 0)

    return AdvancedRAGEvalCaseMetrics(
        question=eval_case.question,
        passed=not failures,
        budgeted_recall=0.0 if over_budget else grounded_recall,
        failures=tuple(failures),
        recall=recall,
        precision=precision,
        cited_recall=cited_recall,
        citations_resolved=citations_resolved,
        cited_document_ids=cited_refs,
        inspected_first=inspected_first,
        metadata_calls=metadata_calls,
        retrieval_calls=retrieval_calls,
        filtered_retrieval_calls=filtered_retrieval_calls,
        tool_errors=len(stats.errors),
        steps=steps,
        duration=duration,
        exit_reason=result.get("exit_reason"),
        agent_run_digest=digest_agent_run(result=result, policy=digest_policy),
    )


def widen_tool_budgets(
    budgets: dict[ToolNames, int], retrieval_tools: frozenset[str], metadata_tools: frozenset[str]
) -> dict[ToolNames, int]:
    """
    Extend the budgeted retrieval and metadata groups with the names this candidate uses for them.

    The budgets name the tools the reference had. A candidate can rename one or add another, and a budget
    written as `search_documents` would stop covering the retrieval the candidate actually does. Growing the
    group keeps one allowance over the whole job rather than handing the new tool an allowance of its own.

    :param budgets: The allowances the evaluator was configured with, as `{tool name or names: limit}`.
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

    An eval case passes only when its retrieval, citation and tool-budget expectations all hold. The same
    expectations apply to every eval case, so they are set once here.
    """

    def __init__(
        self,
        *,
        min_recall: float = 1.0,
        min_precision: float = 0.0,
        require_citations: bool = True,
        max_tool_errors: int = 0,
        tool_budgets: dict[ToolNames, int] | None = None,
        digest_policy: AgentRunDigestPolicy | None = None,
        max_traced_eval_cases: int | None = 6,
        max_concurrent_eval_cases: int = 1,
    ) -> None:
        """
        Create an evaluator.

        :param min_recall: Share of an eval case's needed documents the answer must be grounded in for that eval case
            to pass. `budgeted_recall` is reported either way; this only decides which eval cases are failures.
        :param min_precision: Share of what a run retrieved that must be needed for its eval case to pass.
        :param require_citations: Whether answers are scored on what they cited rather than on what was retrieved.
            An answer with no citations at all would otherwise pass a citation check trivially.
        :param max_tool_errors: How many failing tool calls a run may make and still pass. A run over this fails
            without voiding what it retrieved, since a tool that errors is the configuration's problem.
        :param tool_budgets: How many times a run may call a tool, or a group of tools sharing one allowance, as
            `{tool name or names: limit}`. Anything left unnamed falls to the `*` entry, or to a default. A run over
            any of them scores zero. See `resolve_tool_budgets`.
        :param digest_policy: Caps applied to the tool trace recorded for each eval case.
        :param max_traced_eval_cases: How many eval case traces to keep, or `None` to keep every one. A trace explains a
            result but a reader's history of them is cumulative, so failing eval cases keep theirs first: a passing eval
            case has nothing to diagnose.
        :param max_concurrent_eval_cases: How many eval cases to measure at once. Eval cases are independent and
            each one spends its time waiting on a model, so measuring several together is most of what makes an
            experiment affordable in wall-clock terms. Token usage, and therefore cost, is unaffected. Measured
            durations are not: concurrent runs contend for the same rate limits, so leave this at 1 when ranking
            by duration, or the objective measures this setting rather than the configuration. Eval cases are driven
            through `Agent.run_async` whatever this is set to, so one at a time is simply a concurrency of one.
        :raises ValueError: If `max_concurrent_eval_cases` is below one.
        """
        if max_concurrent_eval_cases < 1:
            msg = "max_concurrent_eval_cases must be at least 1."
            raise ValueError(msg)
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.require_citations = require_citations
        self.max_tool_errors = max_tool_errors
        self.tool_budgets = dict(tool_budgets or {})
        self.digest_policy = digest_policy
        self.max_concurrent_eval_cases = max_concurrent_eval_cases
        self.max_traced_eval_cases = max_traced_eval_cases

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the evaluator.

        :returns: The evaluator's `type` and its settings, with `tool_budgets` as `[tool name or names, limit]` pairs.
        """
        policy = self.digest_policy
        return default_to_dict(
            self,
            min_recall=self.min_recall,
            min_precision=self.min_precision,
            require_citations=self.require_citations,
            max_tool_errors=self.max_tool_errors,
            # A group of tools is a tuple, which JSON cannot hold as a key
            tool_budgets=[
                [tools if isinstance(tools, str) else list(tools), limit] for tools, limit in self.tool_budgets.items()
            ],
            digest_policy=None
            if policy is None
            else {**asdict(policy), "keep_full_results_for": sorted(policy.keep_full_results_for)},
            max_traced_eval_cases=self.max_traced_eval_cases,
            max_concurrent_eval_cases=self.max_concurrent_eval_cases,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdvancedRAGHarnessEvaluator":
        """
        Restore an evaluator from what `to_dict` returned.

        :param data: The serialized evaluator.
        :returns: The restored evaluator.
        """
        parameters = dict(data.get("init_parameters") or {})
        parameters["tool_budgets"] = {
            tools if isinstance(tools, str) else tuple(tools): limit
            for tools, limit in parameters.get("tool_budgets") or []
        }
        if (policy := parameters.get("digest_policy")) is not None:
            parameters["digest_policy"] = AgentRunDigestPolicy(
                **{**policy, "keep_full_results_for": frozenset(policy["keep_full_results_for"])}
            )
        return default_from_dict(cls, {**data, "init_parameters": parameters})

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
        eval_case: RetrievalEvalCase,
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
        :param tool_budgets: The allowances, resolved against the tools the candidate has.
        :returns: The eval case score.
        """
        duration = time.perf_counter() - started
        scored = _score_advanced_rag_result(
            result=result,
            eval_case=eval_case,
            duration=duration,
            min_recall=self.min_recall,
            min_precision=self.min_precision,
            require_citations=self.require_citations,
            max_tool_errors=self.max_tool_errors,
            tool_budgets=tool_budgets,
            digest_policy=self.digest_policy,
            retrieval_tools=retrieval_tools,
            metadata_tools=metadata_tools,
        )
        logger.info(
            "eval case {position}/{total} {verdict} in {duration:.1f}s: {question}",
            position=position,
            total=total,
            verdict="passed" if scored.passed else f"FAILED ({', '.join(scored.failures)})",
            duration=duration,
            question=eval_case.question[:80],
        )
        return scored

    async def _measure(
        self, agent: Agent, eval_cases: list[RetrievalEvalCase]
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
        tool_budgets = resolve_tool_budgets(
            budgets=widen_tool_budgets(
                budgets=self.tool_budgets, retrieval_tools=retrieval_tools, metadata_tools=metadata_tools
            ),
            tool_names=[tool.name for tool in tools],
        )

        async def measure(
            position: int, eval_case: RetrievalEvalCase
        ) -> tuple[AdvancedRAGEvalCaseMetrics, EvalCaseSummary]:
            """Run one eval case, waiting for a slot first."""
            async with semaphore:
                started = time.perf_counter()
                with tracing.tracer.trace(
                    EVAL_CASE_SPAN, tags={"haystack.harness.eval_case.question": eval_case.question}
                ) as span:
                    result = await agent.run_async(messages=[ChatMessage.from_user(text=eval_case.question)])
                # Without a HarnessTracer active nothing was recorded, so nothing can be vouched for.
                usage = (
                    span.collected.summarize()
                    if isinstance(span, HarnessSpan) and span.collected is not None
                    else EvalCaseSummary(all_tokens_reported=False)
                )
            scored = self._score(
                result=result,
                eval_case=eval_case,
                started=started,
                position=position,
                total=len(eval_cases),
                retrieval_tools=retrieval_tools,
                metadata_tools=metadata_tools,
                tool_budgets=tool_budgets,
            )
            return scored, usage

        return list(
            await asyncio.gather(*(measure(position, eval_case) for position, eval_case in enumerate(eval_cases, 1)))
        )

    def evaluate(self, target: Agent, eval_cases: list[RetrievalEvalCase]) -> EvalMetrics:
        """
        Pose every eval case and return raw experiment metrics, from synchronous code.

        :param target: The materialized candidate Agent to score.
        :param eval_cases: The labelled expectations to score it against.
        :returns: What `evaluate_async` measured.
        :raises RuntimeError: If an event loop is already running; await `evaluate_async` from inside one.
        """
        return asyncio.run(self.evaluate_async(target=target, eval_cases=eval_cases))

    async def evaluate_async(self, target: Agent, eval_cases: list[RetrievalEvalCase]) -> EvalMetrics:
        """
        Pose every eval case and return raw experiment metrics.

        :param target: The materialized candidate Agent to score.
        :param eval_cases: The labelled expectations to score it against.
        :returns: Per-eval-case durations and raw model usage, with `details` holding `mean_budgeted_recall`,
            `mean_cited_recall`, `mean_recall`, `mean_precision`, `pass_rate`, `model` and `warnings`, and one record
            per eval case in `eval_cases`.
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
            for model, tokens in usage.model_usage.items():
                model_usage[model] = model_usage.get(model, ModelTokenUsage()) + tokens
        return EvalMetrics(
            durations=[metric.duration for metric in flattened],
            # A run that attributed nothing to a model is as unpriceable as one that under-reported.
            all_tokens_reported=all(usage.all_tokens_reported and bool(usage.model_usage) for _, usage in measured),
            model_usage=model_usage,
            eval_cases=self._traced_eval_cases(metrics=flattened),
            details={
                "mean_budgeted_recall": sum(metric.budgeted_recall for metric in flattened) / len(flattened),
                "pass_rate": sum(metric.passed for metric in flattened) / len(flattened),
                "mean_recall": sum(metric.recall for metric in flattened) / len(flattened),
                "mean_cited_recall": sum(metric.cited_recall for metric in flattened) / len(flattened),
                "mean_precision": sum(metric.precision for metric in flattened) / len(flattened),
                # What the components said about themselves while they ran; a tool or hook that degrades rather
                # than failing reports it only here.
                "warnings": diagnostics.to_list(),
                "model": getattr(target.chat_generator, "model", None),
            },
        )
