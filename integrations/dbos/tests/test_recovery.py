# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Replay behaviour.

Recovery in DBOS means re-executing the workflow body from the top while completed steps return their recorded
output. These tests exercise that path with DBOS's own public recovery operations rather than killing a process:
`fork_workflow` restarts a failed workflow from a chosen step, and `resume_workflow` picks an interrupted one back
up. Both take the same replay path a crash-restart would.
"""

import contextlib
import threading

import pytest
from dbos import DBOS, SetWorkflowID
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage

from haystack_integrations.dbos import DBOSChatGenerator, durable_agent

from .conftest import FakeChatGenerator, counting_tool, tool_call_reply

# Forking from step 2 keeps the first recorded step and re-runs everything from the second one onwards.
_SECOND_STEP = 2


@pytest.mark.usefixtures("dbos_app")
def test_model_calls_are_replayed_not_reissued():
    inner = FakeChatGenerator(replies=[ChatMessage.from_assistant("the answer")])
    agent = durable_agent(Agent(chat_generator=inner), name="replay")
    crash = {"pending": True}

    @DBOS.workflow()
    def answer(question: str) -> str:
        text = agent.run(messages=[ChatMessage.from_user(question)])["last_message"].text
        if crash["pending"]:
            crash["pending"] = False
            msg = "crash after the model call"
            raise RuntimeError(msg)
        return text

    DBOS.launch()

    with pytest.raises(RuntimeError), SetWorkflowID("wf-replay"):
        answer("hi")
    assert len(inner.calls) == 1
    assert DBOS.get_workflow_status("wf-replay").status == "ERROR"

    recovered = DBOS.fork_workflow("wf-replay", _SECOND_STEP)

    assert recovered.get_result() == "the answer"
    # The model call was taken from its checkpoint. Had it been re-issued, this would be 2.
    assert len(inner.calls) == 1


@pytest.mark.usefixtures("dbos_app")
def test_model_calls_are_replayed_after_an_interrupted_run():
    """A workflow that is cancelled mid-flight - the closest in-process analogue of losing the process."""
    inner = FakeChatGenerator(replies=[ChatMessage.from_assistant("the answer")])
    agent = durable_agent(Agent(chat_generator=inner), name="interrupted")
    reached_tail = threading.Event()
    release_tail = threading.Event()

    def tail() -> str:
        reached_tail.set()
        release_tail.wait(timeout=10)
        return "tail"

    @DBOS.workflow()
    def answer(question: str) -> str:
        text = agent.run(messages=[ChatMessage.from_user(question)])["last_message"].text
        DBOS.run_step({"name": "tail"}, tail)
        return text

    DBOS.launch()

    with SetWorkflowID("wf-interrupted"):
        handle = DBOS.start_workflow(answer, "hi")

    assert reached_tail.wait(timeout=10)
    assert len(inner.calls) == 1

    DBOS.cancel_workflow("wf-interrupted")
    release_tail.set()
    # Draining the handle of a cancelled workflow raises on some dbos versions and returns on others; either way
    # the workflow is left unfinished, which is what this test needs.
    with contextlib.suppress(Exception):
        handle.get_result()

    assert DBOS.resume_workflow("wf-interrupted").get_result() == "the answer"
    assert len(inner.calls) == 1


@pytest.mark.usefixtures("dbos_app")
def test_tool_calls_are_not_checkpointed_and_run_again():
    counter: dict[str, int] = {}
    inner = FakeChatGenerator(replies=[tool_call_reply(), ChatMessage.from_assistant("finished")])
    agent = durable_agent(
        Agent(chat_generator=inner, tools=[counting_tool(counter)], tool_concurrency_limit=1),
        name="tools",
    )
    crash = {"pending": True}

    @DBOS.workflow()
    def answer(question: str) -> str:
        text = agent.run(messages=[ChatMessage.from_user(question)])["last_message"].text
        if crash["pending"]:
            crash["pending"] = False
            msg = "crash after the tool ran"
            raise RuntimeError(msg)
        return text

    DBOS.launch()

    with pytest.raises(RuntimeError), SetWorkflowID("wf-tools"):
        answer("hi")
    assert counter["record"] == 1

    assert DBOS.fork_workflow("wf-tools", _SECOND_STEP).get_result() == "finished"

    # The first model call replayed from its checkpoint, but the tool is not a step, so it executed a second time.
    # This is the at-least-once contract the README documents; it is asserted so a change is caught.
    assert len(inner.calls) == 3
    assert counter["record"] == 2


@pytest.mark.usefixtures("dbos_app")
def test_recorded_step_name_uses_the_agent_name():
    agent = durable_agent(Agent(chat_generator=FakeChatGenerator()), name="support")

    @DBOS.workflow()
    def answer() -> str:
        return agent.run(messages=[ChatMessage.from_user("hi")])["last_message"].text

    DBOS.launch()

    with SetWorkflowID("wf-named"):
        answer()

    names = [step["function_name"] for step in DBOS.list_workflow_steps("wf-named")]

    assert names == ["support__chat_generator.run"]


@pytest.mark.usefixtures("dbos_app")
def test_one_step_is_recorded_per_model_call():
    counter: dict[str, int] = {}
    inner = FakeChatGenerator(replies=[tool_call_reply(), ChatMessage.from_assistant("finished")])
    agent = durable_agent(
        Agent(chat_generator=inner, tools=[counting_tool(counter)], tool_concurrency_limit=1),
        name="steps",
    )

    @DBOS.workflow()
    def answer() -> str:
        return agent.run(messages=[ChatMessage.from_user("hi")])["last_message"].text

    DBOS.launch()

    with SetWorkflowID("wf-steps"):
        answer()

    names = [step["function_name"] for step in DBOS.list_workflow_steps("wf-steps")]

    assert names == ["steps__chat_generator.run", "steps__chat_generator.run"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("dbos_app")
async def test_model_calls_are_replayed_in_an_async_workflow():
    inner = FakeChatGenerator(replies=[ChatMessage.from_assistant("async answer")])
    agent = durable_agent(Agent(chat_generator=inner), name="async_replay")
    crash = {"pending": True}

    @DBOS.workflow()
    async def answer(question: str) -> str:
        result = await agent.run_async(messages=[ChatMessage.from_user(question)])
        if crash["pending"]:
            crash["pending"] = False
            msg = "crash after the model call"
            raise RuntimeError(msg)
        return result["last_message"].text

    DBOS.launch()

    with pytest.raises(RuntimeError), SetWorkflowID("wf-async"):
        await answer("hi")
    assert len(inner.calls) == 1

    recovered = await DBOS.fork_workflow_async("wf-async", _SECOND_STEP)

    assert await recovered.get_result() == "async answer"
    assert len(inner.calls) == 1


@pytest.mark.usefixtures("dbos_app")
def test_no_step_is_recorded_outside_a_workflow():
    inner = FakeChatGenerator()
    wrapper = DBOSChatGenerator(inner, name="loose")

    DBOS.launch()
    result = wrapper.run(messages=[ChatMessage.from_user("hi")])

    assert result["replies"][0].text == "done"
    assert len(inner.calls) == 1
