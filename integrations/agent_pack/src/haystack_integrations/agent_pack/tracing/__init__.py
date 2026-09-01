# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Local, machine-readable trace capture for Haystack runs.

Haystack ships no way to get a run's spans back as data: `LoggingTracer` writes for humans, and the capturing
tracers in Haystack's own test suite are not packaged. This module fills that gap for anything that needs runs as
structured data — evaluation, replay, regression checks, harness optimization.

It depends only on Haystack, deliberately: it is generic infrastructure kept here while its shape is proved out,
and a candidate for moving into Haystack once we are happy with it.
"""

from haystack_integrations.agent_pack.tracing.collectors import (
    LocalTraceCollector,
    TraceCapturingAgentRunner,
)
from haystack_integrations.agent_pack.tracing.dataclasses import (
    TRACE_SCHEMA_VERSION,
    CapturedAgentRun,
    TraceArtifact,
    TraceCaptureLimits,
    TraceSelection,
)
from haystack_integrations.agent_pack.tracing.extraction import (
    extract_agent_reference_output,
    extract_agent_replay_inputs,
    is_replayable,
)
from haystack_integrations.agent_pack.tracing.serialization import span_tag
from haystack_integrations.agent_pack.tracing.stores import LocalTraceStore
from haystack_integrations.agent_pack.tracing.types import TraceSource

__all__ = [
    "TRACE_SCHEMA_VERSION",
    "CapturedAgentRun",
    "LocalTraceCollector",
    "LocalTraceStore",
    "TraceArtifact",
    "TraceCaptureLimits",
    "TraceCapturingAgentRunner",
    "TraceSelection",
    "TraceSource",
    "extract_agent_reference_output",
    "extract_agent_replay_inputs",
    "is_replayable",
    "span_tag",
]
