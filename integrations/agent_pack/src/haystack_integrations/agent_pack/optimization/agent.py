# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import distributions
from typing import Any

from haystack import Pipeline, logging
from haystack import __version__ as haystack_version
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool, Toolset, flatten_tools_or_toolsets, warm_up_tools

from haystack_integrations.agent_pack.evaluation.dataclasses import EvalMetrics, ModelPrice, cost_of_model_usage
from haystack_integrations.agent_pack.evaluation.harness_evaluator import HarnessEvaluator
from haystack_integrations.agent_pack.optimization import prompts
from haystack_integrations.agent_pack.optimization.dataclasses import (
    CandidateConfiguration,
    CandidateOutcome,
    ConfigurationDraft,
    KnownConfigurations,
    OptimizationObjectives,
    ProposalResult,
)
from haystack_integrations.agent_pack.optimization.tools import (
    ValidateConfig,
    _make_haystack_documentation_toolset,
    edit_config,
    finish,
    inspect_component,
    read_config,
    restore_candidate,
    submit_candidate,
)
from haystack_integrations.agent_pack.optimization.utils import _configuration_id, load_agent

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 180.0
_DEFAULT_MAX_RETRIES = 5

EVAL_CASE_SUMMARY_KEY = "eval_case_summary"


def _default_llm(model: str) -> OpenAIResponsesChatGenerator:
    """
    A default OpenAI Responses-API generator with the pack's timeout/retry settings and low reasoning effort.

    :param model: The OpenAI model name.
    :returns: The generator, with the optimizer's prompt cache key set.
    """
    return OpenAIResponsesChatGenerator(
        model=model,
        timeout=_DEFAULT_TIMEOUT,
        max_retries=_DEFAULT_MAX_RETRIES,
        generation_kwargs={"prompt_cache_key": prompts.OPTIMIZER_PROMPT_CACHE_KEY, "reasoning": {"effort": "low"}},
    )


def _describe_environment() -> str:
    """
    Describe what an experiment can actually import, as a line for the optimizer's instructions.

    :returns: A sentence naming the Haystack version and every installed Haystack integration.
    """
    integrations = sorted(
        f"{distribution.metadata['Name']} {distribution.version}"
        for distribution in distributions()
        if "haystack" in (distribution.metadata["Name"] or "").lower()
        and distribution.metadata["Name"] != "haystack-ai"
    )
    installed = ", ".join(integrations) if integrations else "none"
    return (
        f"This environment runs Haystack {haystack_version} and has these Haystack integrations installed: "
        f"{installed}. Anything a configuration names has to be importable from those; nothing else is available, "
        f"however well it would suit the experiment."
    )


def create_harness_optimizer_agent(
    *,
    evaluator: HarnessEvaluator,
    loader: Callable[[str], Agent | Pipeline] = load_agent,
    llm: ChatGenerator | None = None,
    system_prompt: str | None = None,
    max_agent_steps: int = 24,
    additional_instructions: str | None = None,
    documentation_tools: bool = False,
) -> Agent:
    """
    Create the harness optimizer agent.

    The agent reads a configuration's YAML together with its evaluation results and edits it into the next
    candidate to measure.

    :param evaluator: The evaluator the candidates are measured with. `validate_config` runs its `validate` on every
        draft, and it has to implement `to_dict` and `from_dict` so the agent can be serialized.
    :param loader: Builds a configuration from YAML, and decides what shape a candidate must keep. Defaults to the
        Agent contract; pass `load_pipeline` to optimize a Pipeline that is not a single Agent.
    :param llm: LLM that reasons about the evaluation results and edits the configuration. Defaults to
        `OpenAIResponsesChatGenerator("gpt-5.6-terra")` with low reasoning effort.
    :param system_prompt: Overrides the pre-made system prompt. A description of the installed Haystack version
        and integrations is appended either way.
    :param max_agent_steps: Maximum steps for the agent loop while producing one candidate.
    :param additional_instructions: Guidance appended to the system prompt about what a good configuration looks
        like for one specific harness.
    :param documentation_tools: Give the agent read-only search over the public Haystack documentation, so it can
        look up a component's parameters and serialized shape. Requires `mcp-haystack`, and reaches the
        documentation server over the network.
    :returns: The harness optimizer `Agent`. Its editing tools work on the `draft` and `known` state passed to `run`,
        and a run ends when the optimizer calls `submit_candidate` or `finish`. `propose_candidate` runs one turn
        with that state and the prompt built from the experiment so far.
    """
    instructions = system_prompt or prompts.HARNESS_OPTIMIZER_SYSTEM_PROMPT
    instructions = f"{instructions}\n\n## This environment\n\n{_describe_environment()}"
    if additional_instructions is not None:
        instructions = f"{instructions}\n\n## This harness\n\n{additional_instructions.strip()}"
    llm = llm or _default_llm("gpt-5.6-terra")
    tools: list[Tool | Toolset] = [
        read_config,
        edit_config,
        ValidateConfig(evaluator=evaluator, loader=loader),
        submit_candidate,
        restore_candidate,
        finish,
        inspect_component,
    ]
    if documentation_tools:
        tools.append(_make_haystack_documentation_toolset())
    return Agent(
        chat_generator=llm,
        tools=tools,
        system_prompt=instructions,
        exit_conditions=["submit_candidate", "finish"],
        # The state one proposal turn's editing tools share. `validation_failures` is a list, so its writes are
        # appended; every other write replaces the value.
        state_schema={
            "draft": {"type": ConfigurationDraft},
            "known": {"type": KnownConfigurations},
            "submitted": {"type": CandidateConfiguration | None},
            "finish_reason": {"type": str | None},
            "validation_failures": {"type": list[dict[str, str]]},
        },
        max_agent_steps=max_agent_steps,
    )


def _tool_specifications(reference: Agent | Pipeline) -> list[dict[str, Any]]:
    """
    Describe the tools the reference Agent can call.

    :param reference: The configuration whose tools to describe. A Pipeline that is not an Agent has none.
    :returns: One `{name, description, parameters}` entry per tool, or an empty list when they cannot be read.
    """
    tools = getattr(reference, "tools", None)
    if not tools:
        return []
    try:
        warm_up_tools(tools=tools)
        return [tool.tool_spec for tool in flatten_tools_or_toolsets(tools=tools)]
    except Exception as error:
        logger.warning("Could not describe the reference Agent's tools: {error}", error=error)
        return []


def _fenced(text: str, language: str = "") -> str:
    """
    Wrap text in a fence long enough that nothing inside can close it early.

    :param text: Content to fence, reproduced exactly.
    :param language: Optional language hint for the opening fence.
    :returns: The fenced block.
    """
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _section(title: str, body: str) -> str:
    """Render one titled section, or nothing when there is no body."""
    return f"## {title}\n\n{body}\n\n" if body else ""


def _component_size(sizes: dict[str, int]) -> str:
    """
    Render one component's output size, naming the bounds only when it varied across eval cases.

    :param sizes: The component's smallest, typical and largest output size.
    :returns: The typical size, followed by the bounds when they differ from it.
    """
    low, high = sizes["min"], sizes["max"]
    return f"{sizes['median']}" if low == high else f"{sizes['median']} ({low}-{high})"


def _headline(metrics: EvalMetrics | None, cost: float | None) -> str:
    """
    Reduce one outcome's measurement to what a reader compares across candidates.

    :param metrics: The candidate's measurement, or None when it could not be measured.
    :param cost: The candidate's cost, or None when it could not be priced.
    :returns: A single line of headline numbers.
    """
    if metrics is None:
        return "not measured"
    details = metrics.details
    parts = ["cost unpriced" if cost is None else f"cost ${cost:.4f}"]
    # Every numeric value the harness reported, in its order
    for key, value in details.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            parts.append(f"{key} {value:.3f}")
    # Output sizes per component, in the order the components finished
    if sizes_by_component := details.get("component_output_sizes"):
        parts.append(
            "components "
            + " -> ".join(
                f"{component}.{socket} {_component_size(sizes=sizes)}"
                for component, sockets in sizes_by_component.items()
                for socket, sizes in sockets.items()
            )
        )
    if metrics.eval_cases:
        summary = _summarize_eval_cases(eval_cases=metrics.eval_cases)
        parts.append(f"{summary['passed']}/{summary['total']} eval cases clean")
        if summary["failures"]:
            parts.append("failures " + ", ".join(f"{name} x{count}" for name, count in summary["failures"].items()))
    if not metrics.all_tokens_reported:
        parts.append("usage incomplete")
    for warning in details.get("warnings") or []:
        parts.append(f"warning x{warning['count']}: {warning['message']}")
    return " | ".join(parts)


def _summarize_eval_cases(eval_cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce a per-eval-case listing to how many passed and which kinds of failure occurred."""
    failures: Counter[str] = Counter()
    for eval_case in eval_cases:
        failures.update(str(label) for label in eval_case.get("failures") or ())
    return {
        "total": len(eval_cases),
        "passed": sum(1 for eval_case in eval_cases if eval_case.get("passed")),
        "failures": dict(failures.most_common()),
    }


def _summarized_metrics(metrics: EvalMetrics) -> dict[str, Any]:
    """
    Serialize one measurement with its per-eval-case listing replaced by a count of how the eval cases ended.

    :param metrics: One measurement.
    :returns: The serialized measurement, with `eval_case_summary` in place of `eval_cases` when it has any.
    """
    summarized = metrics.to_dict()
    if metrics.eval_cases:
        del summarized["eval_cases"]
        summarized[EVAL_CASE_SUMMARY_KEY] = _summarize_eval_cases(eval_cases=metrics.eval_cases)
    return summarized


def _render_outcomes(history: list[CandidateOutcome]) -> str:
    """
    Render every measured candidate as a readable entry, newest last.

    :param history: Prior candidate outcomes.
    :returns: One section per candidate.
    """
    entries = []
    for position, outcome in enumerate(history, start=1):
        parent = (outcome.parent_id or "reference")[:12]
        gates = ", ".join(outcome.gate_failures) or "passed"
        lines = [f"### {position}. `{outcome.candidate_id[:12]}` from `{parent}` — gates {gates}"]
        lines.append(_headline(metrics=outcome.metrics, cost=outcome.cost))
        if outcome.failure:
            lines.append(f"failed to evaluate: {outcome.failure}")
        if outcome.rationale:
            lines.append(f"hypothesis: {outcome.rationale}")
        if outcome.diff:
            lines.append(_fenced(text=outcome.diff.rstrip(), language="diff"))
        entries.append("\n\n".join(lines))
    return "\n\n".join(entries)


def propose_candidate(
    optimizer_agent: Agent,
    reference: Agent | Pipeline,
    reference_yaml: str,
    prices: dict[str, ModelPrice],
    objectives: OptimizationObjectives,
    baseline: EvalMetrics,
    history: list[CandidateOutcome],
    history_digest_window: int = 1,
    remaining_evaluations: int | None = None,
    candidates: dict[str, str] | None = None,
    base_id: str | None = None,
) -> ProposalResult:
    """
    Let the optimizer edit, validate and submit one YAML candidate.

    :param optimizer_agent: The agent from `create_harness_optimizer_agent`.
    :param reference: Reference configuration supplying tool specifications, when it has any.
    :param reference_yaml: The reference serialized as YAML, which the optimizer can restore as `"reference"`.
    :param prices: Known token prices keyed by model identifier.
    :param objectives: Quality gates and ranking objective.
    :param baseline: Reference measurement.
    :param history: Prior candidate outcomes, oldest first.
    :param history_digest_window: How many of the most recent outcomes are shown in full.
    :param remaining_evaluations: How many candidates, including this one, the experiment can still measure. Shown
        to the optimizer as its budget, or as unknown when None.
    :param candidates: The YAML of every candidate submitted on earlier turns, keyed by candidate ID. The optimizer
        can restore any of them, and submitting one of them again is refused.
    :param base_id: The candidate this turn's edits start from, or `None` to start from the reference.
    :returns: The submitted candidate, or `None` in its place when the optimizer finished or ran out of steps, along
        with why it finished and the drafts that failed validation.
    :raises ValueError: If `base_id` is not one of `candidates`.
    """
    # The turn's starting point, which the editing tools read from the agent's state
    reference_id = _configuration_id(reference_yaml)
    known = KnownConfigurations(
        reference_id=reference_id, yaml_by_id={reference_id: reference_yaml, **(candidates or {})}
    )
    if base_id is not None and base_id not in known.yaml_by_id:
        msg = f"Unknown base candidate ID: {base_id}."
        raise ValueError(msg)
    parent_id = base_id or reference_id
    draft = ConfigurationDraft(yaml=known.yaml_by_id[parent_id], parent_id=parent_id)

    # The three messages go from least to most often changing, so each turn reuses as much of the prompt cache as
    # possible.

    # First message: identical every turn, so it caches in full.
    tools = _tool_specifications(reference=reference)
    context = "".join(
        [
            _section(title="Objectives", body=json.dumps(objectives.to_dict())),
            _section(
                title="Known model prices",
                body=json.dumps({model_id: asdict(obj=price) for model_id, price in prices.items()}),
            ),
            _section(title="Tools available to the reference", body=json.dumps(tools) if tools else ""),
            # The reference is shown eval case by eval case until the first candidate is measured
            _section(
                title="Reference measurement",
                body=json.dumps(
                    {
                        "cost": cost_of_model_usage(model_usage=baseline.model_usage, prices=prices),
                        **(baseline.to_dict() if not history else _summarized_metrics(metrics=baseline)),
                    }
                ),
            ),
        ]
    )

    # Second message: append-only, so every turn re-reads all but the newest entry from cache.
    outcomes = _section(title="Outcomes so far", body=_render_outcomes(history=history))

    # Third message: rewritten every turn, so none of it is cacheable and all of it goes last.
    recent = history[-history_digest_window:] if history_digest_window > 0 else []
    current = "".join(
        [
            _section(
                title="Budget",
                body="unknown"
                if remaining_evaluations is None
                else f"{remaining_evaluations} evaluations remain, including this one.",
            ),
            _section(
                title="Most recent outcome in full",
                body="\n\n".join(json.dumps(outcome.to_dict()) for outcome in recent),
            ),
            _section(
                title="Candidate YAML",
                body=f"revision `{draft.revision}`, edited from `{draft.parent_id[:12]}`\n\n"
                # Shown verbatim, since the editing tools match against this exact text
                + _fenced(text=draft.yaml, language="yaml"),
            ),
        ]
    )

    result = optimizer_agent.run(
        messages=[
            ChatMessage.from_user(text=context),
            ChatMessage.from_user(text=outcomes),
            ChatMessage.from_user(text=current),
        ],
        draft=draft,
        known=known,
    )

    logger.info(
        "optimizer turn: steps={steps}/{budget} exit={exit_reason} calls={calls} usage={usage}",
        steps=result.get("step_count"),
        budget=optimizer_agent.max_agent_steps,
        exit_reason=result.get("exit_reason"),
        calls=[call.tool_name for message in result.get("messages") or [] for call in message.tool_calls],
        usage=result.get("token_usage"),
    )
    return ProposalResult(
        candidate=result.get("submitted"),
        finish_reason=result.get("finish_reason"),
        validation_failures=list(result.get("validation_failures") or []),
    )
