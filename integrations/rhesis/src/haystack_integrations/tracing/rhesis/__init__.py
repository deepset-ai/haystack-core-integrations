# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from .conversation import DEFAULT_TURN_SPAN_NAME, ConversationTurn, RhesisTracing
from .tracer import (
    DefaultSpanHandler,
    RhesisSpan,
    RhesisTelemetry,
    RhesisTracer,
    SpanContext,
    SpanHandler,
    rhesis_invocation_context,
    tracing_context_var,
)

__all__ = [
    "DEFAULT_TURN_SPAN_NAME",
    "ConversationTurn",
    "DefaultSpanHandler",
    "RhesisSpan",
    # Exported because it is the declared type of ``SpanHandler.tracer`` and of
    # ``RhesisTracer.telemetry``: a custom handler that reads ``self.tracer.otel_tracer`` or opens its
    # own span needs the type without importing a private module path.
    "RhesisTelemetry",
    "RhesisTracer",
    "RhesisTracing",
    "SpanContext",
    "SpanHandler",
    "rhesis_invocation_context",
    "tracing_context_var",
]
