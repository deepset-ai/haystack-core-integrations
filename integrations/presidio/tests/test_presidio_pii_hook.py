# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from haystack import component
from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.dataclasses import ChatMessage, ToolCall
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine

from haystack_integrations.hooks.presidio import PresidioPIIHook

_MODULE = "haystack_integrations.hooks.presidio.presidio_pii_hook"
_SUPPORTED = ["PERSON", "LOCATION", "CREDIT_CARD", "IBAN_CODE", "EMAIL_ADDRESS"]


def _fake_analyze(spans: dict[str, tuple[str, float]]):
    """Return an `analyze` side effect that reports each substring in `spans` as (entity_type, score)."""

    def analyze(text: str, language: str, entities: list[str] | None, score_threshold: float):  # noqa: ARG001
        results = []
        for substring, (entity_type, score) in spans.items():
            start = text.find(substring)
            if start != -1 and score >= score_threshold:
                results.append(
                    RecognizerResult(entity_type=entity_type, start=start, end=start + len(substring), score=score)
                )
        return results

    return analyze


def _hook(spans: dict[str, tuple[str, float]] | None = None, **kwargs) -> PresidioPIIHook:
    """Return a hook with a fake analyzer and the real (model-free) anonymizer."""
    hook = PresidioPIIHook(**kwargs)
    hook._analyzer = MagicMock()
    hook._analyzer.analyze.side_effect = _fake_analyze(spans or {})
    hook._anonymizer = AnonymizerEngine()
    return hook


def _state(messages: list[ChatMessage], exit_reason: str | None = None) -> State:
    schema = {"exit_reason": {"type": str}, "stop_run": {"type": str}}
    state = State(schema=schema)
    state.set("messages", messages)
    state.set("exit_reason", exit_reason)
    return state


class TestInit:
    def test_defaults(self):
        hook = PresidioPIIHook()
        assert hook.language == "en"
        assert hook.entities is None
        assert hook.score_threshold == 0.35
        assert hook.entity_score_thresholds == {}
        assert hook.entity_actions == {}
        assert hook.on_error == "block"

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"score_threshold": 1.5}, "between 0 and 1"),
            ({"entity_score_thresholds": {"PERSON": -0.1}}, "between 0 and 1"),
            ({"entity_actions": {"PERSON": "mask"}}, "Unknown action 'mask'"),
            ({"on_error": "ignore"}, "Unknown `on_error` value"),
        ],
    )
    def test_invalid_config_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            PresidioPIIHook(**kwargs)

    def test_serialization_round_trip(self):
        hook = PresidioPIIHook(
            language="de",
            entities=["PERSON", "CREDIT_CARD"],
            entity_score_thresholds={"PERSON": 0.6},
            entity_actions={"CREDIT_CARD": "block"},
            on_error="stop",
            block_message="removed",
        )
        data = hook.to_dict()
        assert data["type"] == f"{_MODULE}.PresidioPIIHook"
        restored = PresidioPIIHook.from_dict(data)
        assert restored.to_dict() == data


class TestWarmUp:
    def _warm_up(self, hook: PresidioPIIHook):
        analyzer = MagicMock()
        analyzer.get_supported_entities.return_value = _SUPPORTED
        with patch(f"{_MODULE}.build_analyzer_engine", return_value=analyzer) as build:
            hook.warm_up()
        return build

    def test_loads_engines_once(self):
        hook = PresidioPIIHook(language="es")
        build = self._warm_up(hook)
        build.assert_called_once_with(language="es", models=None, default_models=PresidioPIIHook.SPACY_DEFAULT_MODELS)
        assert hook._anonymizer is not None
        self._warm_up(hook).assert_not_called()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"entities": ["PERSN"]}, "`entities` contains entity types not supported"),
            ({"entity_actions": {"CREDIT_CRAD": "block"}}, "`entity_actions` contains entity types not supported"),
            ({"entity_score_thresholds": {"PERSN": 0.5}}, "`entity_score_thresholds` contains entity types not"),
            (
                {"entities": ["PERSON"], "entity_actions": {"CREDIT_CARD": "block"}},
                "`entity_actions` contains entity types missing from `entities`: CREDIT_CARD",
            ),
        ],
    )
    def test_unknown_entities_raise(self, kwargs, match):
        hook = PresidioPIIHook(**kwargs)
        with pytest.raises(ValueError, match=match):
            self._warm_up(hook)
        assert hook._analyzer is None


class TestRun:
    def test_redacts_user_message_and_skips_system(self):
        hook = _hook({"John Smith": ("PERSON", 0.85), "Berlin": ("LOCATION", 0.4)})
        system = ChatMessage.from_system("You are talking to John Smith.")
        state = _state([system, ChatMessage.from_user("Hi, I'm John Smith from Berlin.")])

        hook.run(state)

        messages = state.data["messages"]
        assert messages[0] is system
        assert messages[1].text == "Hi, I'm <PERSON> from <LOCATION>."
        assert messages[1].meta["pii_checked"] is True
        assert state.data.get("stop_run") is None

    def test_per_entity_threshold(self):
        hook = _hook(
            {"John Smith": ("PERSON", 0.85), "Hi": ("PERSON", 0.45), "Berlin": ("LOCATION", 0.4)},
            entity_score_thresholds={"PERSON": 0.6, "LOCATION": 0.3},
        )
        state = _state([ChatMessage.from_user("Hi, I'm John Smith from Berlin.")])

        hook.run(state)

        assert state.data["messages"][0].text == "Hi, I'm <PERSON> from <LOCATION>."
        assert hook._analyzer.analyze.call_args.kwargs["score_threshold"] == 0.3

    def test_block_mid_run_adds_reminder(self):
        hook = _hook({"4111 1111 1111 1111": ("CREDIT_CARD", 1.0)}, entity_actions={"CREDIT_CARD": "block"})
        state = _state([ChatMessage.from_user("My card is 4111 1111 1111 1111")])

        hook.run(state)

        blocked, reminder = state.data["messages"]
        assert blocked.text == hook.block_message
        assert reminder.meta["system_reminder"] is True
        assert "CREDIT_CARD" in reminder.text

    def test_block_on_exit_replaces_reply_without_reminder(self):
        hook = _hook({"4111 1111 1111 1111": ("CREDIT_CARD", 1.0)}, entity_actions={"CREDIT_CARD": "block"})
        user = ChatMessage.from_user("What card did I use?")
        state = _state([user, ChatMessage.from_assistant("You used 4111 1111 1111 1111.")], exit_reason="text")

        hook.run(state)

        messages = state.data["messages"]
        assert len(messages) == 2
        assert messages[1].text == hook.block_message

    def test_stop_redacts_and_sets_stop_run(self):
        hook = _hook({"DE89370400440532013000": ("IBAN_CODE", 1.0)}, entity_actions={"IBAN_CODE": "stop"})
        state = _state([ChatMessage.from_user("Send it to DE89370400440532013000")])

        hook.run(state)

        assert state.data["messages"][0].text == "Send it to <IBAN_CODE>"
        assert state.data["stop_run"] == "pii_detected"

    def test_strictest_action_wins(self):
        hook = _hook(
            {"John": ("PERSON", 0.9), "4111 1111 1111 1111": ("CREDIT_CARD", 1.0)},
            entity_actions={"CREDIT_CARD": "block"},
        )
        state = _state([ChatMessage.from_user("I'm John, card 4111 1111 1111 1111")])

        hook.run(state)

        assert state.data["messages"][0].text == hook.block_message

    def test_checks_each_message_once(self):
        hook = _hook({"John": ("PERSON", 0.9)})
        state = _state([ChatMessage.from_user("I'm John")])

        hook.run(state)
        hook.run(state)

        assert hook._analyzer.analyze.call_count == 1

    def test_redacts_tool_result_and_keeps_tool_call(self):
        hook = _hook({"alice@example.com": ("EMAIL_ADDRESS", 1.0)})
        call = ToolCall(tool_name="lookup", arguments={"name": "alice@example.com"}, id="1")
        assistant = ChatMessage.from_assistant("Looking up alice@example.com", tool_calls=[call])
        tool = ChatMessage.from_tool(tool_result="Found alice@example.com", origin=call, error=False)
        state = _state([assistant, tool])

        hook.run(state)

        new_assistant, new_tool = state.data["messages"]
        assert new_assistant.text == "Looking up <EMAIL_ADDRESS>"
        assert new_assistant.tool_call == call
        assert new_tool.tool_call_result.result == "Found <EMAIL_ADDRESS>"
        assert new_tool.tool_call_result.origin == call

    def test_skips_system_reminders(self):
        hook = _hook({"John": ("PERSON", 0.9)})
        reminder = ChatMessage.from_user("Remind John", meta={"system_reminder": True})
        state = _state([reminder])

        hook.run(state)

        assert state.data["messages"] == [reminder]
        hook._analyzer.analyze.assert_not_called()

    @pytest.mark.parametrize(
        ("on_error", "blocked", "stopped"), [("block", True, False), ("stop", True, True), ("pass", False, False)]
    )
    def test_on_error(self, on_error, blocked, stopped, caplog):
        hook = _hook(on_error=on_error)
        hook._analyzer.analyze.side_effect = RuntimeError("model missing")
        state = _state([ChatMessage.from_user("I'm John")])

        with caplog.at_level(logging.WARNING):
            hook.run(state)

        assert (state.data["messages"][0].text == hook.block_message) is blocked
        assert (state.data.get("stop_run") == "pii_detected") is stopped
        assert "Presidio failed" in caplog.text

    def test_records_decision_on_span(self):
        hook = _hook(
            {"John": ("PERSON", 0.9), "4111 1111 1111 1111": ("CREDIT_CARD", 1.0)},
            entity_actions={"CREDIT_CARD": "block"},
        )
        state = _state([ChatMessage.from_user("I'm John"), ChatMessage.from_user("card 4111 1111 1111 1111")])
        span = MagicMock()

        with patch(f"{_MODULE}.tracing.tracer.current_span", return_value=span):
            hook.run(state)

        span.set_tags.assert_called_once_with(
            tags={
                "haystack.agent.hook.pii.checked_messages": 2,
                "haystack.agent.hook.pii.triggered": True,
                "haystack.agent.hook.pii.action": "block",
                "haystack.agent.hook.pii.entities": {"PERSON": 1, "CREDIT_CARD": 1},
                "haystack.agent.hook.pii.error": False,
            }
        )
        content_tags = {c.args[0]: c.args[1] for c in span.set_content_tag.call_args_list}
        assert content_tags["haystack.agent.hook.pii.input"] == ["I'm John", "card 4111 1111 1111 1111"]
        assert content_tags["haystack.agent.hook.pii.output"] == ["I'm <PERSON>", hook.block_message]
        # The checked text must never appear in the plain tags.
        assert "John" not in json.dumps(span.set_tags.call_args.kwargs)


@component
class _RecordingChatGenerator:
    """Replies with a fixed text and records the messages it was sent."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.received: list[list[ChatMessage]] = []

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
        self.received.append(list(messages))
        return {"replies": [ChatMessage.from_assistant(self.reply)]}


class TestWithAgent:
    def test_input_and_reply_are_redacted(self):
        hook = _hook({"John Smith": ("PERSON", 0.9), "Jane Doe": ("PERSON", 0.9)})
        generator = _RecordingChatGenerator(reply="Nice to meet you. Say hi to Jane Doe.")
        agent = Agent(chat_generator=generator, hooks={"before_llm": [hook], "after_run": [hook]})

        result = agent.run(messages=[ChatMessage.from_user("I'm John Smith.")])

        assert generator.received[0][-1].text == "I'm <PERSON>."
        assert result["last_message"].text == "Nice to meet you. Say hi to <PERSON>."

    def test_stop_ends_run_before_llm_call(self):
        hook = _hook({"DE89370400440532013000": ("IBAN_CODE", 1.0)}, entity_actions={"IBAN_CODE": "stop"})
        generator = _RecordingChatGenerator(reply="ok")
        agent = Agent(chat_generator=generator, hooks={"before_llm": [hook]})

        result = agent.run(messages=[ChatMessage.from_user("IBAN DE89370400440532013000")])

        assert generator.received == []
        assert result["messages"][-1].text == "IBAN <IBAN_CODE>"


@pytest.mark.integration
class TestIntegration:
    def test_redacts_with_real_models(self):
        hook = PresidioPIIHook(entity_actions={"CREDIT_CARD": "block"})
        hook.warm_up()
        state = _state(
            [
                ChatMessage.from_user("Hi, I am Alice, call me at 212-555-5678"),
                ChatMessage.from_user("Card 4111111111111111"),
            ]
        )

        hook.run(state)

        redacted, blocked, reminder = state.data["messages"]
        assert "Alice" not in redacted.text
        assert "212-555-5678" not in redacted.text
        assert blocked.text == hook.block_message
        assert reminder.meta["system_reminder"] is True
