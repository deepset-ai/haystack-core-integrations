# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, field
from typing import Any

from haystack.dataclasses import ChatMessage, ToolCall

AGENT_RUN_DIGEST_KEY = "agent_run_digest"


@dataclass(kw_only=True)
class AgentRunDigestPolicy:
    """
    Caps applied while compressing one Agent run.

    :param max_tool_calls: Tool calls kept, from the start of the run. A run long enough to exceed this has usually
        broken a budget already, and the leading calls are the ones that explain it.
    :param max_argument_chars: Characters kept of one call's serialized arguments. Arguments are the highest-value
        evidence per character in a run, since they are what the Agent chose.
    :param max_result_chars: Characters kept of one tool result.
    :param max_answer_chars: Characters kept of the final answer.
    :param keep_full_results_for: Tools whose results are never truncated. Use it for introspection tools whose
        result is a listing whose completeness matters.
    """

    max_tool_calls: int = 40
    max_argument_chars: int = 800
    max_result_chars: int = 600
    max_answer_chars: int = 600
    keep_full_results_for: frozenset[str] = field(default_factory=frozenset)


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """Cut text to a limit, naming how much was dropped so a prefix cannot be mistaken for the whole."""
    if len(text) <= limit:
        return text, False
    return f"{text[:limit]}… [{len(text) - limit} more characters omitted]", True


def _tool_steps(messages: list[ChatMessage], policy: AgentRunDigestPolicy) -> tuple[list[dict[str, Any]], int]:
    """Pair each tool call with its result, in call order."""
    results: dict[str, Any] = {}
    for message in messages:
        for call_result in message.tool_call_results:
            origin: ToolCall | None = getattr(call_result, "origin", None)
            # A result is matched to its call by id. `origin` also repeats the call's name and arguments, so it is
            # never emitted: doing so would duplicate every argument dict in the digest.
            if origin is not None and origin.id is not None:
                results[origin.id] = call_result

    calls = [call for message in messages for call in message.tool_calls]
    steps: list[dict[str, Any]] = []
    for call in calls[: policy.max_tool_calls]:
        arguments, arguments_truncated = _truncate(
            text=json.dumps(call.arguments or {}, default=str), limit=policy.max_argument_chars
        )
        step: dict[str, Any] = {"tool": call.tool_name, "arguments": arguments}
        if arguments_truncated:
            step["arguments_truncated"] = True
        if (tool_result := results.get(call.id or "")) is not None:
            body = str(tool_result.result)
            limit = len(body) if call.tool_name in policy.keep_full_results_for else policy.max_result_chars
            text, truncated = _truncate(text=body, limit=limit)
            step["result"] = text
            step["error"] = bool(tool_result.error)
            if truncated:
                step["result_truncated"] = True
        steps.append(step)
    return steps, max(len(calls) - policy.max_tool_calls, 0)


def digest_agent_run(result: dict[str, Any], policy: AgentRunDigestPolicy | None = None) -> dict[str, Any]:
    """
    Compress one Agent run to its tool behaviour and outcome.

    :param result: The dictionary returned by `Agent.run`.
    :param policy: Caps to apply. Defaults to `AgentRunDigestPolicy()`.
    :returns: A JSON-compatible digest: how the run ended, the tools it called with their arguments and results,
        and an excerpt of its answer. `tool_call_counts` is kept verbatim because it names every tool available to
        the Agent, including tools it never called.
    """
    policy = policy or AgentRunDigestPolicy()
    messages: list[ChatMessage] = result.get("messages") or []
    steps, omitted = _tool_steps(messages=messages, policy=policy)
    answers = [message.text for message in messages if message.is_from("assistant") and message.text]
    answer, answer_truncated = _truncate(text=answers[-1] if answers else "", limit=policy.max_answer_chars)

    digest: dict[str, Any] = {
        "exit_reason": result.get("exit_reason"),
        "step_count": result.get("step_count"),
        "tool_call_counts": result.get("tool_call_counts"),
        "tool_steps": steps,
        "answer_excerpt": answer,
    }
    if omitted:
        digest["omitted_tool_calls"] = omitted
    if answer_truncated:
        digest["answer_truncated"] = True
    return digest
