# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Deriving replay inputs and reference outputs from captured traces."""

from typing import Any

from haystack_integrations.agent_pack.optimization.tracing.dataclasses import (
    AGENT_INPUT_TAG,
    AGENT_OUTPUT_TAG,
    AGENT_RUN_OPERATION,
    TraceArtifact,
)
from haystack_integrations.agent_pack.optimization.tracing.serialization import span_tag


def _root_agent_span(artifact: TraceArtifact) -> dict[str, Any] | None:
    """
    Return the outermost Agent run span in the artifact.

    The Agent span is not necessarily the root of the trace: running the reference Agent inside a Pipeline, or under
    any surrounding instrumentation, gives it a parent. Depth is measured through captured parent links so the
    outermost Agent run is selected either way.
    """
    by_id = {span.get("span_id"): span for span in artifact.traces if span.get("span_id") is not None}
    candidates = [span for span in artifact.traces if span.get("operation_name") == AGENT_RUN_OPERATION]
    if not candidates:
        return None

    def depth(span: dict[str, Any]) -> int:
        seen: set[Any] = set()
        steps = 0
        parent_id = span.get("parent_span_id")
        while parent_id is not None and parent_id in by_id and parent_id not in seen:
            seen.add(parent_id)
            steps += 1
            parent_id = by_id[parent_id].get("parent_span_id")
        return steps

    return min(candidates, key=lambda span: (depth(span), span.get("start_time") or ""))


def _agent_content_tag(artifact: TraceArtifact, tag: str) -> dict[str, Any]:
    """Read one content tag off the outermost Agent run span, or explain why it is not there."""
    span = _root_agent_span(artifact=artifact)
    if span is None:
        msg = f"Trace {artifact.run_id} contains no {AGENT_RUN_OPERATION} span and cannot be replayed."
        raise ValueError(msg)
    value = span_tag(span=span, key=tag)
    if not isinstance(value, dict):
        msg = (
            f"Trace {artifact.run_id} has no captured {tag}. Capture the reference run with a "
            "LocalTraceCollector configured with capture_content=True."
        )
        raise ValueError(msg)
    return value


def extract_agent_replay_inputs(artifact: TraceArtifact) -> dict[str, Any]:
    """
    Extract the Agent inputs required to replay a reference trace.

    :param artifact: The captured reference trace.
    :returns: The recorded Agent run inputs.
    :raises ValueError: If the trace holds no Agent run span, or was captured without content.
    """
    return _agent_content_tag(artifact=artifact, tag=AGENT_INPUT_TAG)


def extract_agent_reference_output(artifact: TraceArtifact) -> dict[str, Any]:
    """
    Extract the Agent output recorded in a reference trace.

    :param artifact: The captured reference trace.
    :returns: The recorded Agent run output.
    :raises ValueError: If the trace holds no Agent run span, or was captured without content.
    """
    return _agent_content_tag(artifact=artifact, tag=AGENT_OUTPUT_TAG)


def is_replayable(artifact: TraceArtifact) -> bool:
    """
    Return whether replay inputs can be derived from this artifact.

    :param artifact: The captured reference trace.
    :returns: True if the trace carries Agent run inputs.
    """
    try:
        extract_agent_replay_inputs(artifact=artifact)
    except ValueError:
        return False
    return True
