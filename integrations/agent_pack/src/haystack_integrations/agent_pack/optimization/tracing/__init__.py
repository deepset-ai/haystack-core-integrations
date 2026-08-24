# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Local, machine-readable trace capture for Agent Pack optimization campaigns."""

from haystack_integrations.agent_pack.optimization.tracing.collectors import (
    LocalTraceCollector,
    TraceCapturingAgentRunner,
)
from haystack_integrations.agent_pack.optimization.tracing.dataclasses import (
    TRACE_SCHEMA_VERSION,
    CapturedAgentRun,
    TraceArtifact,
    TraceCaptureLimits,
    TraceSelection,
)
from haystack_integrations.agent_pack.optimization.tracing.extraction import (
    extract_agent_reference_output,
    extract_agent_replay_inputs,
    is_replayable,
)
from haystack_integrations.agent_pack.optimization.tracing.serialization import span_tag
from haystack_integrations.agent_pack.optimization.tracing.stores import LocalTraceStore
from haystack_integrations.agent_pack.optimization.tracing.types import TraceSource

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
