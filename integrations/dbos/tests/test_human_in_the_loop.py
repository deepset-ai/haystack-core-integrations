# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import json
import threading
import time
from collections.abc import Callable

import pytest
from dbos import DBOS, SetWorkflowID
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.hooks.human_in_the_loop import (
    AlwaysAskPolicy,
    ConfirmationHook,
    ConfirmationUIResult,
    NeverAskPolicy,
)

from haystack_integrations.dbos import DBOSConfirmationStrategy, durable_agent
from haystack_integrations.dbos.human_in_the_loop import _to_ui_result

from .conftest import FakeChatGenerator, counting_tool, tool_call_reply

_EVENT_KEY = "haystack.pending_tool_call"


class TestMessageParsing:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("confirm", ConfirmationUIResult(action="confirm")),
            ({"action": "reject"}, ConfirmationUIResult(action="reject")),
            (
                {"action": "reject", "feedback": "too risky"},
                ConfirmationUIResult(action="reject", feedback="too risky"),
            ),
            (
                {"action": "modify", "new_tool_params": {"value": "y"}},
                ConfirmationUIResult(action="modify", new_tool_params={"value": "y"}),
            ),
        ],
    )
    def test_accepts_supported_message_shapes(self, message, expected):
        assert _to_ui_result(message) == expected

    def test_passes_a_ui_result_through(self):
        result = ConfirmationUIResult(action="confirm")

        assert _to_ui_result(result) is result

    @pytest.mark.parametrize("message", [42, {"no_action": True}, None])
    def test_rejects_unusable_messages(self, message):
        with pytest.raises(ValueError, match="must be an action string"):
            _to_ui_result(message)


class TestTopics:
    def test_defaults_to_a_topic_per_tool(self):
        assert DBOSConfirmationStrategy()._topic_for("delete_file") == "confirm.delete_file"

    def test_an_explicit_topic_wins(self):
        assert DBOSConfirmationStrategy(topic="approvals")._topic_for("delete_file") == "approvals"


class TestWithoutAWorkflow:
    def test_allows_the_call_and_warns(self, caplog):
        decision = DBOSConfirmationStrategy().run(
            tool_name="delete_file", tool_description="Delete a file", tool_params={"path": "reports/x"}
        )

        assert decision.execute is True
        assert "needs a DBOS workflow" in caplog.text

    def test_a_policy_can_skip_the_confirmation_entirely(self):
        strategy = DBOSConfirmationStrategy(confirmation_policy=NeverAskPolicy())

        decision = strategy.run(tool_name="delete_file", tool_description="d", tool_params={"path": "reports/x"})

        assert decision.execute is True
        assert decision.final_tool_params == {"path": "reports/x"}


class TestSerialization:
    def test_round_trips_without_a_policy(self):
        strategy = DBOSConfirmationStrategy(topic="approvals", timeout_seconds=30.0)
        data = strategy.to_dict()

        assert data["type"] == "haystack_integrations.dbos.human_in_the_loop.DBOSConfirmationStrategy"
        assert json.loads(json.dumps(data)) == data

        restored = DBOSConfirmationStrategy.from_dict(data)

        assert restored.topic == "approvals"
        assert restored.timeout_seconds == 30.0
        assert restored.confirmation_policy is None

    def test_round_trips_with_a_policy(self):
        strategy = DBOSConfirmationStrategy(confirmation_policy=AlwaysAskPolicy())

        restored = DBOSConfirmationStrategy.from_dict(strategy.to_dict())

        assert isinstance(restored.confirmation_policy, AlwaysAskPolicy)


def _agent_awaiting_confirmation(counter: dict[str, int], **strategy_kwargs) -> Agent:
    """An Agent that asks to run `record`, gated behind a DBOS confirmation."""
    inner = FakeChatGenerator(replies=[tool_call_reply(), ChatMessage.from_assistant("finished")])
    hook = ConfirmationHook(confirmation_strategies={"record": DBOSConfirmationStrategy(**strategy_kwargs)})
    return durable_agent(
        Agent(
            chat_generator=inner,
            tools=[counting_tool(counter)],
            tool_concurrency_limit=1,
            hooks={"before_tool": [hook]},
        ),
        name="hitl",
    )


def _answer_in_background(agent: Agent, workflow_id: str, tail: Callable[[], str] | None = None):
    @DBOS.workflow()
    def answer() -> str:
        text = agent.run(messages=[ChatMessage.from_user("delete it")])["last_message"].text
        if tail is not None:
            DBOS.run_step({"name": "tail"}, tail)
        return text

    DBOS.launch()
    with SetWorkflowID(workflow_id):
        return DBOS.start_workflow(answer)


def _wait_for_pending_call(workflow_id: str) -> dict:
    pending = DBOS.get_event(workflow_id, _EVENT_KEY, timeout_seconds=10)
    assert pending is not None, "the strategy never published the pending tool call"
    return pending


class TestDurableConfirmation:
    @pytest.mark.usefixtures("dbos_app")
    def test_publishes_the_pending_call_and_runs_the_tool_once_confirmed(self):
        counter: dict[str, int] = {}
        handle = _answer_in_background(_agent_awaiting_confirmation(counter), "wf-confirm")

        pending = _wait_for_pending_call("wf-confirm")

        assert pending["tool_name"] == "record"
        assert pending["tool_params"] == {"value": "x"}
        assert counter.get("record") is None

        DBOS.send("wf-confirm", {"action": "confirm"}, topic="confirm.record")

        assert handle.get_result() == "finished"
        assert counter["record"] == 1

    @pytest.mark.usefixtures("dbos_app")
    def test_a_rejection_stops_the_tool_and_tells_the_model_why(self):
        counter: dict[str, int] = {}
        handle = _answer_in_background(_agent_awaiting_confirmation(counter), "wf-reject")
        _wait_for_pending_call("wf-reject")

        DBOS.send("wf-reject", {"action": "reject", "feedback": "too risky"}, topic="confirm.record")

        assert handle.get_result() == "finished"
        assert counter.get("record") is None

    @pytest.mark.usefixtures("dbos_app")
    def test_a_modification_changes_the_tool_arguments(self):
        counter: dict[str, int] = {}
        handle = _answer_in_background(_agent_awaiting_confirmation(counter), "wf-modify")
        _wait_for_pending_call("wf-modify")

        DBOS.send("wf-modify", {"action": "modify", "new_tool_params": {"value": "safe"}}, topic="confirm.record")

        assert handle.get_result() == "finished"
        assert counter["record"] == 1

    @pytest.mark.usefixtures("dbos_app")
    def test_a_timeout_leaves_the_tool_unexecuted(self):
        counter: dict[str, int] = {}
        handle = _answer_in_background(_agent_awaiting_confirmation(counter, timeout_seconds=0.5), "wf-timeout")

        assert handle.get_result() == "finished"
        assert counter.get("record") is None

    @pytest.mark.usefixtures("dbos_app")
    def test_the_human_is_not_asked_twice_after_an_interrupted_run(self):
        """The decisive property: a recovered run reuses the recorded decision instead of asking again."""
        counter: dict[str, int] = {}
        # The confirmation timeout is the discriminator. Nobody sends a second decision after the interruption, so
        # a resumed run that awaited one afresh would sit here for the full timeout before giving up.
        confirmation_timeout = 30.0
        agent = _agent_awaiting_confirmation(counter, timeout_seconds=confirmation_timeout)
        reached_tail = threading.Event()
        release_tail = threading.Event()

        def tail() -> str:
            reached_tail.set()
            release_tail.wait(timeout=10)
            return "tail"

        handle = _answer_in_background(agent, "wf-hitl-replay", tail=tail)
        _wait_for_pending_call("wf-hitl-replay")

        DBOS.send("wf-hitl-replay", {"action": "confirm"}, topic="confirm.record")
        assert reached_tail.wait(timeout=10)
        assert counter["record"] == 1

        # Interrupt the run after the decision was recorded but before the workflow finished.
        DBOS.cancel_workflow("wf-hitl-replay")
        release_tail.set()
        # See test_recovery.py: how a cancelled workflow surfaces to its handle differs across dbos versions.
        with contextlib.suppress(Exception):
            handle.get_result()

        # No second DBOS.send happens here. The resumed run gets the recorded decision straight back.
        started = time.monotonic()
        result = DBOS.resume_workflow("wf-hitl-replay").get_result()
        elapsed = time.monotonic() - started

        assert result == "finished"
        assert elapsed < confirmation_timeout / 3, "the resumed run appears to have waited for a fresh decision"
        # Deliberately no assertion on the tool counter here: whether resuming re-executes the workflow body or
        # picks up after the last completed step differs across dbos versions, and this test is about the human.
