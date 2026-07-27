# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging, tracing
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.base_serialization import deserialize_class_instance, serialize_class_instance

from haystack_integrations.tracing.rhesis import RhesisTracer, SpanHandler
from haystack_integrations.tracing.rhesis.tracer import RhesisTelemetry, resolve_frontend_url, tracing_context_var
from rhesis.telemetry.provider import get_tracer_provider

logger = logging.getLogger(__name__)


@component
class RhesisConnector:
    """
    Connects Haystack to [Rhesis](https://rhesis.ai) for OpenTelemetry-based tracing of pipelines.

    Add this component to a pipeline without connecting it to other components. It enables tracing
    for all pipeline operations when Haystack tracing is active.

    **Environment Configuration:**
    - ``RHESIS_API_KEY``: Required API key for trace ingestion.
    - ``RHESIS_BASE_URL``: Backend URL (default ``http://localhost:8080`` for local development).
    - ``RHESIS_PROJECT_ID``: Optional project identifier (resolved from the API key when omitted).
    - ``RHESIS_ENVIRONMENT``: Deployment environment label (default ``development``).
    - ``RHESIS_FRONTEND_URL``: Optional frontend URL used to build ``trace_url`` deep links.
    - ``HAYSTACK_CONTENT_TRACING_ENABLED``: Must be ``"true"`` **before importing Haystack** to
      capture input/output on spans.
    - ``HAYSTACK_RHESIS_ENFORCE_FLUSH``: When ``"true"`` (default), flushes after each span.
      Set to ``"false"`` in long-running services and flush on shutdown instead.

    Example shutdown flush for FastAPI:

    ```python
    from haystack.tracing import tracer

    @app.on_event("shutdown")
    async def shutdown_event():
        tracer.actual_tracer.flush()
    ```
    """

    def __init__(
        self,
        name: str,
        api_key: Secret | None = Secret.from_env_var("RHESIS_API_KEY"),
        base_url: str | None = None,
        project_id: str | None = None,
        environment: str | None = None,
        frontend_url: str | None = None,
        span_handler: SpanHandler | None = None,
    ) -> None:
        """
        Initialize the RhesisConnector component.

        :param name: Trace name shown in the Rhesis UI.
        :param api_key: Rhesis API key. Defaults to ``RHESIS_API_KEY``.
        :param base_url: Rhesis backend base URL. Defaults to ``RHESIS_BASE_URL`` or
            ``http://localhost:8080``.
        :param project_id: Rhesis project ID. Defaults to ``RHESIS_PROJECT_ID``.
        :param environment: Environment label. Defaults to ``RHESIS_ENVIRONMENT`` or
            ``development``.
        :param frontend_url: Frontend base URL for ``trace_url``. Defaults to ``RHESIS_FRONTEND_URL``.
        :param span_handler: Optional custom span handler. Uses :class:`DefaultSpanHandler` when omitted.
        """
        self.name = name
        self.api_key = api_key
        resolved_base_url = base_url if base_url is not None else os.getenv("RHESIS_BASE_URL", "http://localhost:8080")
        resolved_environment = (
            environment if environment is not None else os.getenv("RHESIS_ENVIRONMENT", "development")
        )
        self.base_url = resolved_base_url or "http://localhost:8080"
        self.environment = resolved_environment or "development"
        self.project_id = project_id or os.getenv("RHESIS_PROJECT_ID")
        self.frontend_url = frontend_url or os.getenv("RHESIS_FRONTEND_URL")
        self.span_handler = span_handler

        resolved_api_key = api_key.resolve_value() if api_key else None
        if not resolved_api_key:
            msg = "RHESIS_API_KEY is required for RhesisConnector"
            raise ValueError(msg)

        provider = get_tracer_provider(
            service_name="rhesis-haystack",
            api_key=resolved_api_key,
            base_url=self.base_url,
            project_id=self.project_id,
            environment=self.environment,
        )
        otel_tracer = provider.get_tracer("haystack_integrations.tracing.rhesis")
        telemetry = RhesisTelemetry(
            provider=provider,
            otel_tracer=otel_tracer,
            project_id=self.project_id,
            environment=self.environment,
            base_url=self.base_url,
            frontend_url=resolve_frontend_url(self.base_url, self.frontend_url),
        )
        self.tracer = RhesisTracer(telemetry=telemetry, name=name, span_handler=span_handler)
        tracing.enable_tracing(self.tracer)

    @component.output_types(name=str, trace_url=str, trace_id=str)
    def run(self, invocation_context: dict[str, Any] | None = None) -> dict[str, str]:
        """
        Run the connector and return trace metadata.

        :param invocation_context: Optional key-value metadata attached to the root trace
            (session, test run identifiers, tags, etc.).
        :returns: Dictionary with ``name``, ``trace_url``, and ``trace_id``.
        """
        if invocation_context:
            tracing_context_var.set(invocation_context)
            logger.debug(
                "Rhesis tracer invoked with context: '{invocation_context}'",
                invocation_context=invocation_context,
            )
        return {
            "name": self.name,
            "trace_url": self.tracer.get_trace_url(),
            "trace_id": self.tracer.get_trace_id(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        span_handler = serialize_class_instance(self.span_handler) if self.span_handler else None
        return default_to_dict(
            self,
            name=self.name,
            api_key=self.api_key.to_dict() if self.api_key else None,
            base_url=self.base_url,
            project_id=self.project_id,
            environment=self.environment,
            frontend_url=self.frontend_url,
            span_handler=span_handler,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RhesisConnector":
        """Deserialize this component from a dictionary."""
        init_params = data["init_parameters"]
        deserialize_secrets_inplace(init_params, keys=["api_key"])
        if init_params.get("span_handler") is not None:
            init_params["span_handler"] = deserialize_class_instance(init_params["span_handler"])
        return default_from_dict(cls, data)
