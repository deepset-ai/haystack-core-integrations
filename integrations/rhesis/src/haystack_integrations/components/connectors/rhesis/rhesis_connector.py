# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

from haystack import DeserializationError, component, default_from_dict, default_to_dict, logging, tracing
from haystack.core.serialization import generate_qualified_class_name, import_class_by_name
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.tracing.rhesis import RhesisTracer, SpanHandler
from haystack_integrations.tracing.rhesis.tracer import RhesisTelemetry, tracing_context_var
from rhesis.telemetry.provider import get_tracer_provider

logger = logging.getLogger(__name__)


# haystack 3.0 dropped ``serialize_class_instance``/``deserialize_class_instance`` from
# ``haystack.utils.base_serialization``. The span handler is an arbitrary user class rather than a
# component, so it still needs the type-tagged envelope those helpers produced.
def _serialize_span_handler(span_handler: SpanHandler) -> dict[str, Any]:
    return {"type": generate_qualified_class_name(type(span_handler)), "data": span_handler.to_dict()}


def _deserialize_span_handler(serialized: dict[str, Any]) -> SpanHandler:
    handler_class = import_class_by_name(serialized["type"])
    if not issubclass(handler_class, SpanHandler):
        msg = f"Class '{serialized['type']}' is not a subclass of SpanHandler."
        raise DeserializationError(msg)
    return handler_class.from_dict(serialized["data"])


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
    - ``HAYSTACK_RHESIS_ENFORCE_FLUSH``: When ``"true"`` (default), exports once per pipeline run,
      as the root span closes. Set to ``"false"`` to leave exporting to the batch processor and
      flush on shutdown instead.

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
        api_key: Secret | None = Secret.from_env_var("RHESIS_API_KEY"),  # noqa: B008
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
        :raises ValueError: If no API key resolves. A component the user explicitly added to a
            pipeline should say so rather than silently trace nothing — but it does mean
            ``Pipeline.from_dict`` on a YAML containing this component needs credentials present.
            :class:`~haystack_integrations.tracing.rhesis.RhesisTracing` deliberately does the
            opposite and degrades to a no-op, because there the caller did not put tracing in the
            data path.
        """
        self.name = name
        self.api_key = api_key
        # Kept exactly as passed, so to_dict serializes the caller's intent rather than this
        # machine's environment. A pipeline dumped on a laptop would otherwise carry
        # `http://localhost:8080` into production, where the environment can no longer correct it.
        self._base_url_arg = base_url
        self._environment_arg = environment
        self._project_id_arg = project_id
        self._frontend_url_arg = frontend_url

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
            frontend_url=self.frontend_url,
        )
        self.tracer = RhesisTracer(telemetry=telemetry, name=name, span_handler=span_handler)
        tracing.enable_tracing(self.tracer)

    @component.output_types(name=str, trace_url=str, trace_id=str)
    def run(self, invocation_context: dict[str, Any] | None = None) -> dict[str, str]:
        """
        Run the connector and return trace metadata.

        The context applies to the pipeline run that invoked this component and no other: the
        tracer scopes it to the run's root span and discards it when that span closes. To attach
        metadata to work that is not a pipeline run — a standalone ``Agent``, say — wrap the call
        in :func:`~haystack_integrations.tracing.rhesis.rhesis_invocation_context` instead.

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
        """
        Serialize this component to a dictionary.

        Records the arguments as they were passed, not as they were resolved: anything left to the
        environment stays ``None`` so that deserializing on another machine resolves it there. This
        mirrors how ``Secret.from_env_var`` serializes a reference rather than the secret's value.
        """
        span_handler = _serialize_span_handler(self.span_handler) if self.span_handler else None
        return default_to_dict(
            self,
            name=self.name,
            api_key=self.api_key.to_dict() if self.api_key else None,
            base_url=self._base_url_arg,
            project_id=self._project_id_arg,
            environment=self._environment_arg,
            frontend_url=self._frontend_url_arg,
            span_handler=span_handler,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RhesisConnector":
        """Deserialize this component from a dictionary."""
        init_params = data["init_parameters"]
        deserialize_secrets_inplace(init_params, keys=["api_key"])
        if init_params.get("span_handler") is not None:
            init_params["span_handler"] = _deserialize_span_handler(init_params["span_handler"])
        return default_from_dict(cls, data)
