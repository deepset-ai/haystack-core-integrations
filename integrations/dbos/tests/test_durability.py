# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import SearchableToolset, Toolset

from haystack_integrations.dbos import DBOSChatGenerator, durable_agent

from .conftest import FakeChatGenerator, counting_tool


class TestDurableAgent:
    def test_wraps_the_chat_generator(self):
        inner = FakeChatGenerator()
        agent = durable_agent(Agent(chat_generator=inner), name="support")

        assert isinstance(agent.chat_generator, DBOSChatGenerator)
        assert agent.chat_generator.chat_generator is inner
        assert agent.chat_generator.name == "support"

    def test_does_not_modify_the_original_agent(self):
        inner = FakeChatGenerator()
        original = Agent(chat_generator=inner)

        durable_agent(original)

        assert original.chat_generator is inner

    def test_preserves_the_other_init_parameters(self):
        counter: dict[str, int] = {}
        tool = counting_tool(counter)
        original = Agent(
            chat_generator=FakeChatGenerator(),
            tools=[tool],
            system_prompt="be brief",
            max_agent_steps=7,
            tool_concurrency_limit=1,
            exit_conditions=["text"],
        )

        agent = durable_agent(original)

        assert agent.tools == [tool]
        assert agent.system_prompt == "be brief"
        assert agent.max_agent_steps == 7
        assert agent.tool_concurrency_limit == 1
        assert agent.exit_conditions == ["text"]

    def test_forwards_step_options(self):
        agent = durable_agent(Agent(chat_generator=FakeChatGenerator()), step_options={"max_attempts": 4})

        assert agent.chat_generator.step_options == {"max_attempts": 4}

    def test_rejects_double_wrapping(self):
        agent = durable_agent(Agent(chat_generator=FakeChatGenerator()))

        with pytest.raises(ValueError, match="already a DBOSChatGenerator"):
            durable_agent(agent)

    def test_the_wrapped_agent_still_runs_without_dbos(self):
        agent = durable_agent(Agent(chat_generator=FakeChatGenerator(replies=[ChatMessage.from_assistant("hi")])))

        result = agent.run(messages=[ChatMessage.from_user("hello")])

        assert result["last_message"].text == "hi"


class TestDynamicToolsetWarning:
    def test_warns_on_a_searchable_toolset(self, caplog):
        counter: dict[str, int] = {}
        toolset = SearchableToolset([counting_tool(counter)])

        durable_agent(Agent(chat_generator=FakeChatGenerator(), tools=toolset))

        assert "SearchableToolset" in caplog.text

    def test_warns_on_an_empty_toolset(self, caplog):
        durable_agent(Agent(chat_generator=FakeChatGenerator(), tools=Toolset(tools=[])))

        assert "empty at construction time" in caplog.text

    def test_stays_quiet_for_a_static_tool_list(self, caplog):
        counter: dict[str, int] = {}

        durable_agent(Agent(chat_generator=FakeChatGenerator(), tools=[counting_tool(counter)]))

        assert "recovery" not in caplog.text
