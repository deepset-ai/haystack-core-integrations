# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Compress an Agent run to the evidence a reader can act on.

An `Agent.run` result is dominated by material that says nothing about how the Agent behaved: provider response
metadata, model reasoning, and retrieved document bodies that the tool results already quote. What remains after
those are dropped — every tool call with its arguments, every tool result, and how the run ended — is the part that
explains a run, and it is a small fraction of the whole.

This module depends only on the `Agent.run` output contract, so both the optimization experiment and a harness
evaluator can use it without either importing the other.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from haystack.dataclasses import ChatMessage, ToolCall

RUN_DIGEST_KEY = "run_digest"


@dataclass(frozen=True, kw_only=True)
class RunDigestPolicy:
    """
    Caps applied while compressing one Agent run.

    Truncation always announces itself, because a reader that cannot tell a complete tool result from a prefix will
    treat a partial listing as exhaustive.

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


def _tool_steps(messages: list[ChatMessage], policy: RunDigestPolicy) -> tuple[list[dict[str, Any]], int]:
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
            json.dumps(call.arguments or {}, default=str), policy.max_argument_chars
        )
        step: dict[str, Any] = {"tool": call.tool_name, "arguments": arguments}
        if arguments_truncated:
            step["arguments_truncated"] = True
        if (tool_result := results.get(call.id or "")) is not None:
            body = str(tool_result.result)
            limit = len(body) if call.tool_name in policy.keep_full_results_for else policy.max_result_chars
            text, truncated = _truncate(body, limit)
            step["result"] = text
            step["error"] = bool(tool_result.error)
            if truncated:
                step["result_truncated"] = True
        steps.append(step)
    return steps, max(len(calls) - policy.max_tool_calls, 0)


def digest_agent_run(result: dict[str, Any], policy: RunDigestPolicy | None = None) -> dict[str, Any]:
    """
    Compress one Agent run to its tool behaviour and outcome.

    :param result: The dictionary returned by `Agent.run`.
    :param policy: Caps to apply. Defaults to `RunDigestPolicy()`.
    :returns: A JSON-compatible digest: how the run ended, the tools it called with their arguments and results,
        and an excerpt of its answer. `tool_call_counts` is kept verbatim because it names every tool available to
        the Agent, including tools it never called.
    """
    policy = policy or RunDigestPolicy()
    messages: list[ChatMessage] = result.get("messages") or []
    steps, omitted = _tool_steps(messages=messages, policy=policy)
    answers = [message.text for message in messages if message.is_from("assistant") and message.text]
    answer, answer_truncated = _truncate(answers[-1] if answers else "", policy.max_answer_chars)

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


def strip_run_digests(payload: Any) -> Any:
    """
    Remove every run digest from a JSON-compatible structure.

    Digests are the largest part of an experiment history, and a history is cumulative, so older entries have to
    give theirs up to keep a request bounded. This works without knowing where a harness chose to publish them.

    :param payload: Any JSON-compatible structure.
    :returns: The same structure with every `RUN_DIGEST_KEY` entry removed.
    """
    if isinstance(payload, list):
        return [strip_run_digests(payload=item) for item in payload]
    if isinstance(payload, dict):
        return {key: strip_run_digests(payload=value) for key, value in payload.items() if key != RUN_DIGEST_KEY}
    return payload
