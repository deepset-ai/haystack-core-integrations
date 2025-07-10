import contextlib
import os
from collections.abc import Iterator
from typing import Any, Optional, Union

from haystack import logging
from haystack.tracing import Span, Tracer
from haystack.tracing import tracer as haystack_tracer
from haystack.tracing.utils import coerce_tag_value

import weave
from weave.trace.weave_client import Call, WeaveClient

logger = logging.getLogger(__name__)


class WeaveSpan(Span):
    """
    A bridge between Haystack's Span interface and Weave's Call object.

    Stores metadata about a component execution and its inputs and outputs, and manages the attributes/tags
    that describe the operation.
    """

    def __init__(
        self, call: Optional[Call] = None, parent: Optional[Call] = None, operation: Optional[str] = None
    ) -> None:
        self._call = call
        self._parent = parent
        self._operation = operation
        self._attributes: dict[str, Any] = {}

    def set_tag(self, key: str, value: Any) -> None:
        """
        Set a tag by adding it to the call's inputs.

        :param key: The tag key.
        :param value: The tag value.
        """
        self._attributes[key] = value

    def set_tags(self, tags: dict[str, Any]) -> None:
        for k, v in tags.items():
            self.set_tag(k, v)

    def raw_span(self) -> Any:
        """Access to the underlying Weave Call object."""
        return self._call

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        """Correlation data for logging."""
        if not self._call:
            return {}
        return {"weave.call_id": self._call.id, "weave.run_id": getattr(self._call, "run_id", "")}

    def set_call(self, call: Call) -> None:
        self._call = call

    def get_attributes(self) -> dict[str, Any]:
        return self._attributes


class WeaveTracer(Tracer):
    """
    Implements a Haystack's Tracer to make an interface with Weights and Bias Weave.

    It's responsible for creating and managing Weave calls, and for converting Haystack spans
    to Weave spans. It creates spans for each Haystack component run.
    """

    def __init__(self, project_name: str, **weave_init_kwargs: Any) -> None:
        """
        Initialize the WeaveTracer.

        :param project_name: The name of the project to trace, this is will be the name appearing in Weave project.
        :param weave_init_kwargs: Additional arguments to pass to the Weave client.
        """

        content_tracing_enabled = os.getenv("HAYSTACK_CONTENT_TRACING_ENABLED", "false").lower()
        haystack_tracer.is_content_tracing_enabled = content_tracing_enabled == "true"
        if not haystack_tracer.is_content_tracing_enabled:
            logger.warning(
                "Inputs and Outputs of components traces will not be logged because Haystack tracing is disabled."
                "To enable, set the HAYSTACK_CONTENT_TRACING_ENABLED environment variable to true "
                "before importing Haystack."
            )

        self._client = weave.init(project_name, **weave_init_kwargs)
        self._current_span: Optional[WeaveSpan] = None

    @staticmethod
    def create_call(
        attributes: dict, client: WeaveClient, parent_span: Union[WeaveSpan, None], operation_name: str
    ) -> Call:
        comp_name = attributes.pop("haystack.component.name", "")
        comp_type = attributes.pop("haystack.component.type", "")
        comp_input = attributes.pop("haystack.component.input", {})
        call = client.create_call(
            op=operation_name,
            inputs={key: coerce_tag_value(value) for key, value in comp_input.items()},
            attributes={key: coerce_tag_value(value) for key, value in attributes.items()},
            display_name=f"{comp_name}[{comp_type}].run",
            parent=parent_span.raw_span() if parent_span else None,
        )
        return call

    def current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return self._current_span

    @staticmethod
    def _create_component_span(parent_span: Optional[WeaveSpan], operation_name: str) -> WeaveSpan:
        """Create a span for component runs with deferred call creation."""
        return WeaveSpan(parent=parent_span.raw_span() if parent_span else None, operation=operation_name)

    def _create_regular_span(
        self, operation_name: str, tags: Optional[dict], parent_span: Optional[WeaveSpan]
    ) -> WeaveSpan:
        """Create a span for regular operations with immediate call creation."""
        call = self._client.create_call(
            op=operation_name,
            inputs=tags,
            parent=parent_span.raw_span() if parent_span else None,
        )
        return WeaveSpan(call=call)

    def _finish_component_call(self, span: WeaveSpan, parent_span: Optional[WeaveSpan], operation_name: str) -> None:
        """Create and finish call for component runs."""
        attributes = span.get_attributes()
        call = self.create_call(
            attributes=attributes,
            client=self._client,
            parent_span=parent_span,
            operation_name=operation_name,
        )
        span.set_call(call)
        comp_output = attributes.pop("haystack.component.output", {})
        self._client.finish_call(
            call,
            output={key: coerce_tag_value(value) for key, value in comp_output.items()},
        )

    def _finish_regular_call(self, span: WeaveSpan) -> None:
        """Finish call for regular operations."""
        attributes = span.get_attributes()
        pipeline_output = attributes.pop("haystack.pipeline.output_data", {})
        self._client.finish_call(
            span.raw_span(),
            output={key: coerce_tag_value(value) for key, value in pipeline_output.items()},
        )

    @contextlib.contextmanager
    def trace(
        self,
        operation_name: str,
        tags: Optional[dict[str, Any]] = None,
        # the current implementation violates the Liskov Substitution Principle by using WeaveSpan instead of Span
        # unfortunately, it seems hard to fix without rewriting the Tracer
        parent_span: Optional[WeaveSpan] = None,  # type: ignore[override]
    ) -> Iterator[WeaveSpan]:
        """
        A context manager that creates and manages spans for tracking operations in Weights & Biases Weave.

        It has two main workflows:

        A) For regular operations (operation_name != "haystack.component.run"):
            Creates a Weave Call immediately
            Creates a WeaveSpan with this call
            Sets any provided tags
            Yields the span for use in the with block
            When the block ends, updates the call with pipeline output data

        B) For component runs (operation_name == "haystack.component.run"):
            Creates a WeaveSpan WITHOUT a call initially (deferred creation)
            Sets any provided tags
            Yields the span for use in the with block
            Creates the actual Weave Call only at the end, when all component information is available
            Updates the call with component output data

        This distinction is important because Weave's calls can't be updated once created, but the content
        tags are only set on the Span at a later stage. To get the inputs on call creation, we need to create
        the call after we yield the span.

        """
        # create an appropriate span based on operation type
        is_component_run = operation_name == "haystack.component.run"
        span = (
            self._create_component_span(parent_span, operation_name)
            if is_component_run
            else self._create_regular_span(operation_name, tags, parent_span)
        )

        self._current_span = span
        if tags:
            span.set_tags(tags)

        # this method acts as a context manager so yielding the span yields the context to the caller
        # try-except-else-finally to ensure that a call is created and finished correctly for both types of operations
        try:
            yield span

        except Exception as e:
            # case when an exception is raised, an error occurred note that for a component run ensure call is
            # created even on error
            if is_component_run:
                attributes = span.get_attributes()
                call = self.create_call(
                    attributes=attributes,
                    client=self._client,
                    parent_span=parent_span,
                    operation_name=operation_name,
                )
                span.set_call(call)
                self._client.finish_call(call, exception=e)
            else:
                self._client.finish_call(span.raw_span(), exception=e)
            raise  # re-raise the same exception

        else:
            # case when no exception is raised, operation was successful
            if is_component_run:
                self._finish_component_call(span, parent_span, operation_name)
            else:
                self._finish_regular_call(span)

        finally:
            # reset the current span
            self._current_span = None
