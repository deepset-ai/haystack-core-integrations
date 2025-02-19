import contextlib
from collections.abc import Iterator
from typing import Any, Optional, Union

from haystack.tracing import Span, Tracer
from haystack.tracing.utils import coerce_tag_value

import weave
from weave.trace.weave_client import Call, WeaveClient


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

    def set_call(self, call: "weave.Call") -> None:
        self._call = call

    def get_attributes(self) -> dict[str, Any]:
        return self._attributes


class WeaveTracer(Tracer):
    """
    Implements a Haystack's Tracer to make an interface with Weights and Bias Weave.

    It's responsible for creating and managing Weave calls, and for converting Haystack spans
    to Weave spans. It creates spans for each Haystack component run.


    """

    def __init__(self, project_name: str) -> None:
        """
        Initialize the WeaveTracer.

        :param project_name: The name of the project to trace, this is will be the name appearing in Weave project.
        """
        self._client = weave.init(project_name)
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

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: Optional[dict[str, Any]] = None, parent_span: Optional[WeaveSpan] = None
    ) -> Iterator[WeaveSpan]:
        """
        A context manager that creates and manages spans for tracking operations in Weights & Biases Weave.

        :param operation_name: The name of the operation to trace.
        :param tags: A dictionary of tags to add to the span.
        :param parent_span: The parent span to use for the new span.

        :returns:
            An iterator of WeaveSpan objects.
        """

        # We need to defer call creation for components as a Call in Weave can't be updated
        # but the content tags are only set on the Span at a later stage. To get the inputs on
        # call creation, we need to create the call after we yield the span.

        if operation_name == "haystack.component.run":
            span = WeaveSpan(parent=parent_span.raw_span() if parent_span else None, operation=operation_name)
        else:
            call = self._client.create_call(
                op=operation_name,
                inputs=tags,
                parent=parent_span.raw_span() if parent_span else None,
            )
            span = WeaveSpan(call=call)
        self._current_span = span

        if tags:
            span.set_tags(tags)

        try:
            yield span

        except Exception as e:
            # If the operation is a haystack component run, we haven't created the call yet.
            # That's why we need to create it here.
            if operation_name == "haystack.component.run":
                attributes = span.get_attributes()
                call = self.create_call(
                    attributes=attributes,
                    client=self._client,
                    parent_span=parent_span,
                    operation_name=operation_name,
                )
                span.set_call(call)

            self._client.finish_call(call, exception=e)
            raise

        else:
            attributes = span.get_attributes()
            # If the operation is a haystack component run, we haven't created the call yet.
            # That's why we need to create it here.
            if operation_name == "haystack.component.run":
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
            else:
                pipeline_output = attributes.pop("haystack.pipeline.output_data", {})
                self._client.finish_call(
                    call,
                    output={key: coerce_tag_value(value) for key, value in pipeline_output.items()},
                )
        finally:
            self._current_span = None
