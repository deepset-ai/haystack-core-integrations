# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import re
from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.streaming_chunk import StreamingCallbackT
from haystack.tools import ToolsType
from haystack.utils.deserialization import deserialize_component_inplace

from dbos import DBOS, StepOptions
from haystack_integrations.dbos._runtime import checkpointing, in_step, run_component_async

logger = logging.getLogger(__name__)

_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _default_name(chat_generator: Any) -> str:
    """Derive a step name prefix from the wrapped generator's class name."""
    return _CAMEL_BOUNDARY.sub("_", type(chat_generator).__name__).lower()


def _invoke(chat_generator: Any, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Step body: run the wrapped generator and return its replies in Haystack's dictionary format.

    Only the return value is checkpointed, so the replies are serialized here rather than handed to DBOS as
    `ChatMessage` objects. That keeps the checkpoint readable, portable across serializers, and independent of
    the `ChatMessage` class layout at the time of recovery.
    """
    result = chat_generator.run(**inputs)
    return [message.to_dict() for message in result["replies"]]


async def _invoke_async(chat_generator: Any, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Async twin of `_invoke`."""
    result = await run_component_async(chat_generator, **inputs)
    return [message.to_dict() for message in result["replies"]]


@component
class DBOSChatGenerator:
    """
    Wraps a Chat Generator so that every model call is checkpointed as a DBOS step.

    Inside a `@DBOS.workflow()`, each call is recorded in the DBOS system database. When the workflow is recovered
    after a crash, completed calls return their recorded replies instead of being re-issued, so an Agent resumes at
    the point it reached rather than replaying the conversation against the model.

    Outside a workflow the wrapper is transparent: it simply calls the generator it wraps, so the same object works
    in tests and in applications that never launch DBOS.

    ```python
    from dbos import DBOS, DBOSConfig
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.dbos import DBOSChatGenerator

    agent = Agent(chat_generator=DBOSChatGenerator(OpenAIChatGenerator(), name="support_agent"))


    @DBOS.workflow()
    def answer(question: str) -> str:
        return agent.run(messages=[ChatMessage.from_user(question)])["last_message"].text


    config: DBOSConfig = {"name": "support", "system_database_url": "sqlite:///dbos.sqlite"}
    DBOS(config=config)
    DBOS.launch()
    ```
    """

    def __init__(
        self,
        chat_generator: Any,
        *,
        name: str | None = None,
        step_options: StepOptions | None = None,
    ) -> None:
        """
        Initialize the wrapper.

        :param chat_generator:
            The Chat Generator to wrap. Its `run` is invoked inside the DBOS step.
        :param name:
            Prefix for the recorded step name, which becomes `{name}__chat_generator.run`. Give each Agent in an
            application its own name: DBOS matches recorded steps by name during recovery, so distinct names keep a
            mismatch detectable. Defaults to the wrapped generator's class name in snake case.
        :param step_options:
            Options forwarded to DBOS for the step, such as `retries_allowed`, `max_attempts` and `backoff_rate`.
            A `name` given here wins over the one derived from `name`.
        """
        self.chat_generator = chat_generator
        self.name = name or _default_name(chat_generator)
        self.step_options = step_options
        self._step_options: StepOptions = {
            "name": f"{self.name}__chat_generator.run",
            **(step_options or {}),
        }
        self._supports_tools = "tools" in inspect.signature(chat_generator.run).parameters

    def _inputs(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None,
        generation_kwargs: dict[str, Any] | None,
        tools: ToolsType | None,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the inputs for the wrapped generator, omitting the ones that were not provided."""
        inputs: dict[str, Any] = {"messages": messages, **extra}
        if streaming_callback is not None:
            inputs["streaming_callback"] = streaming_callback
        if generation_kwargs is not None:
            inputs["generation_kwargs"] = generation_kwargs
        if tools is not None:
            if not self._supports_tools:
                msg = (
                    f"{type(self.chat_generator).__name__} does not accept a tools parameter in its run method, "
                    f"so {type(self).__name__} cannot forward the tools it was given."
                )
                raise TypeError(msg)
            inputs["tools"] = tools
        return inputs

    def _warn_if_nested(self) -> None:
        if in_step():
            logger.warning(
                "'{name}' is running inside a DBOS step, so its model calls are not checkpointed separately. This "
                "happens when an Agent is invoked from a tool that is itself a step; the outer step is recovered as "
                "a whole and the inner Agent replays from the beginning.",
                name=self.name,
            )

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        *,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run the wrapped Chat Generator, checkpointing the call when inside a DBOS workflow.

        :param messages: The conversation so far.
        :param streaming_callback: Optional callback for streaming chunks. It is not invoked on recovery, because a
            checkpointed call is replayed from the database rather than re-issued.
        :param generation_kwargs: Optional generation parameters forwarded to the wrapped generator.
        :param tools: Optional tools forwarded to the wrapped generator.
        :param kwargs: Any further inputs the wrapped generator accepts.
        :returns: A dictionary with a `replies` key holding the generated messages.
        """
        inputs = self._inputs(messages, streaming_callback, generation_kwargs, tools, kwargs)
        if not checkpointing():
            self._warn_if_nested()
            return {"replies": self.chat_generator.run(**inputs)["replies"]}
        serialized = DBOS.run_step(self._step_options, _invoke, self.chat_generator, inputs)
        return {"replies": [ChatMessage.from_dict(reply) for reply in serialized]}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        *,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Async version of `run`.

        :param messages: The conversation so far.
        :param streaming_callback: Optional callback for streaming chunks. It is not invoked on recovery.
        :param generation_kwargs: Optional generation parameters forwarded to the wrapped generator.
        :param tools: Optional tools forwarded to the wrapped generator.
        :param kwargs: Any further inputs the wrapped generator accepts.
        :returns: A dictionary with a `replies` key holding the generated messages.
        """
        inputs = self._inputs(messages, streaming_callback, generation_kwargs, tools, kwargs)
        if not checkpointing():
            self._warn_if_nested()
            return {"replies": (await run_component_async(self.chat_generator, **inputs))["replies"]}
        serialized = await DBOS.run_step_async(self._step_options, _invoke_async, self.chat_generator, inputs)
        return {"replies": [ChatMessage.from_dict(reply) for reply in serialized]}

    def warm_up(self) -> None:
        """Warm up the wrapped Chat Generator, if it supports it."""
        warm_up = getattr(self.chat_generator, "warm_up", None)
        if warm_up is not None:
            warm_up()

    async def warm_up_async(self) -> None:
        """Warm up the wrapped Chat Generator asynchronously, if it supports it."""
        warm_up_async = getattr(self.chat_generator, "warm_up_async", None)
        if warm_up_async is not None:
            await warm_up_async()
            return
        self.warm_up()

    def close(self) -> None:
        """Close the wrapped Chat Generator, if it supports it."""
        close = getattr(self.chat_generator, "close", None)
        if close is not None:
            close()

    async def close_async(self) -> None:
        """Close the wrapped Chat Generator asynchronously, if it supports it."""
        close_async = getattr(self.chat_generator, "close_async", None)
        if close_async is not None:
            await close_async()
            return
        self.close()

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self.chat_generator, name="chat_generator"),
            name=self.name,
            step_options=dict(self.step_options) if self.step_options is not None else None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DBOSChatGenerator":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_component_inplace(data["init_parameters"], key="chat_generator")
        return default_from_dict(cls, data)
