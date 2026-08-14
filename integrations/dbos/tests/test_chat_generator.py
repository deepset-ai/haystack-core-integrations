# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json

import pytest
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

from haystack_integrations.dbos import DBOSChatGenerator

from .conftest import FakeChatGenerator, ToollessChatGenerator


class TestTransparency:
    """Without DBOS launched, the wrapper must behave exactly like the generator it wraps."""

    def test_run_calls_through_once(self):
        inner = FakeChatGenerator(replies=[ChatMessage.from_assistant("hello")])
        result = DBOSChatGenerator(inner).run(messages=[ChatMessage.from_user("hi")])

        assert len(inner.calls) == 1
        assert result["replies"] == [ChatMessage.from_assistant("hello")]

    @pytest.mark.asyncio
    async def test_run_async_calls_through_once(self):
        inner = FakeChatGenerator(replies=[ChatMessage.from_assistant("hello")])
        result = await DBOSChatGenerator(inner).run_async(messages=[ChatMessage.from_user("hi")])

        assert len(inner.calls) == 1
        assert result["replies"] == [ChatMessage.from_assistant("hello")]

    def test_returns_the_original_reply_objects(self):
        reply = ChatMessage.from_assistant("hello")
        result = DBOSChatGenerator(FakeChatGenerator(replies=[reply])).run(messages=[])

        assert result["replies"][0] is reply


class TestInputForwarding:
    def test_forwards_optional_inputs_only_when_given(self):
        inner = FakeChatGenerator()
        wrapper = DBOSChatGenerator(inner)

        assert wrapper._inputs([], None, None, None, {}) == {"messages": []}

        def callback(chunk):  # noqa: ARG001
            return None

        tools = [Tool(name="t", description="d", parameters={"type": "object", "properties": {}}, function=len)]
        inputs = wrapper._inputs([], callback, {"temperature": 0.1}, tools, {"extra": 1})

        assert inputs["streaming_callback"] is callback
        assert inputs["generation_kwargs"] == {"temperature": 0.1}
        assert inputs["tools"] == tools
        assert inputs["extra"] == 1

    def test_raises_when_the_wrapped_generator_does_not_support_tools(self):
        wrapper = DBOSChatGenerator(ToollessChatGenerator())
        tools = [Tool(name="t", description="d", parameters={"type": "object", "properties": {}}, function=len)]

        with pytest.raises(TypeError, match="does not accept a tools parameter"):
            wrapper.run(messages=[], tools=tools)

    def test_declares_tools_so_the_agent_accepts_it(self):
        # The Agent introspects the run signature to decide whether a generator supports tools.
        assert "tools" in inspect.signature(DBOSChatGenerator.run).parameters
        assert "tools" in inspect.signature(DBOSChatGenerator.run_async).parameters


class TestNaming:
    def test_default_name_is_the_generator_class_in_snake_case(self):
        wrapper = DBOSChatGenerator(FakeChatGenerator())

        assert wrapper.name == "fake_chat_generator"
        assert wrapper._step_options["name"] == "fake_chat_generator__chat_generator.run"

    def test_explicit_name_is_used_for_the_step(self):
        wrapper = DBOSChatGenerator(FakeChatGenerator(), name="support")

        assert wrapper._step_options["name"] == "support__chat_generator.run"

    def test_step_options_override_the_derived_name(self):
        wrapper = DBOSChatGenerator(FakeChatGenerator(), name="support", step_options={"name": "custom"})

        assert wrapper._step_options["name"] == "custom"

    def test_step_options_are_forwarded(self):
        wrapper = DBOSChatGenerator(FakeChatGenerator(), step_options={"retries_allowed": True, "max_attempts": 5})

        assert wrapper._step_options["retries_allowed"] is True
        assert wrapper._step_options["max_attempts"] == 5


class TestLifecycle:
    def test_forwards_warm_up_and_close(self):
        inner = FakeChatGenerator()
        wrapper = DBOSChatGenerator(inner)

        wrapper.warm_up()
        wrapper.close()

        assert inner.warmed_up == 1
        assert inner.closed == 1

    @pytest.mark.asyncio
    async def test_async_lifecycle_falls_back_to_the_sync_methods(self):
        inner = FakeChatGenerator()
        wrapper = DBOSChatGenerator(inner)

        await wrapper.warm_up_async()
        await wrapper.close_async()

        assert inner.warmed_up == 1
        assert inner.closed == 1

    def test_lifecycle_is_a_no_op_when_the_generator_has_none(self):
        wrapper = DBOSChatGenerator(ToollessChatGenerator())

        wrapper.warm_up()
        wrapper.close()


class TestSerialization:
    def test_to_dict_and_from_dict_round_trip(self):
        wrapper = DBOSChatGenerator(FakeChatGenerator(), name="support", step_options={"max_attempts": 2})
        data = wrapper.to_dict()

        assert data["type"] == "haystack_integrations.dbos.chat_generator.DBOSChatGenerator"
        assert data["init_parameters"]["name"] == "support"
        assert data["init_parameters"]["step_options"] == {"max_attempts": 2}

        restored = DBOSChatGenerator.from_dict(data)

        assert restored.name == "support"
        assert restored.step_options == {"max_attempts": 2}
        assert isinstance(restored.chat_generator, FakeChatGenerator)

    def test_to_dict_is_json_serializable(self):
        data = DBOSChatGenerator(FakeChatGenerator()).to_dict()

        assert json.loads(json.dumps(data)) == data


class TestCheckpointPayload:
    """The step records serialized replies, so the format has to survive a round trip and stay JSON-safe."""

    @pytest.mark.parametrize(
        "message",
        [
            ChatMessage.from_assistant("plain text"),
            ChatMessage.from_assistant(tool_calls=[ToolCall(id="c1", tool_name="record", arguments={"value": "x"})]),
            ChatMessage.from_assistant("text", meta={"model": "gpt", "usage": {"prompt_tokens": 3}}),
            ChatMessage.from_assistant("named", name="assistant-1"),
        ],
    )
    def test_reply_survives_the_step_boundary(self, message):
        serialized = message.to_dict()

        assert json.loads(json.dumps(serialized)) == serialized
        assert ChatMessage.from_dict(serialized) == message
