import os

import pytest
from haystack.components.agents.state import State
from haystack.components.generators.chat import MockChatGenerator, OpenAIResponsesChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole, ToolCall

from haystack_integrations.agent_pack.advanced_rag.agent import _default_llm
from haystack_integrations.agent_pack.advanced_rag.hooks import BackupAnswerHook

SEARCH_CALL = ToolCall(tool_name="search_documents", arguments={"query": "x"})


class TestBackupAnswerHook:
    def test_needs_backup_detects_runs_without_a_text_answer(self):
        assert BackupAnswerHook._needs_backup([ChatMessage.from_assistant(tool_calls=[SEARCH_CALL])])
        assert BackupAnswerHook._needs_backup([ChatMessage.from_tool(tool_result="r", origin=SEARCH_CALL)])
        assert not BackupAnswerHook._needs_backup([ChatMessage.from_assistant("final answer")])
        assert not BackupAnswerHook._needs_backup([])

    def test_appends_an_answer_on_step_exhaustion(self):
        hook = BackupAnswerHook(generator=MockChatGenerator("backup answer from the gathered evidence"))
        state = State(schema={})
        state.set(
            "messages",
            [
                ChatMessage.from_system("system prompt"),
                ChatMessage.from_user("the question"),
                ChatMessage.from_assistant(tool_calls=[SEARCH_CALL]),
                ChatMessage.from_tool(tool_result="[doc abc12345] some evidence", origin=SEARCH_CALL),
            ],
        )

        hook.run(state)

        messages = state.get("messages")
        assert messages[-1].text == "backup answer from the gathered evidence"
        assert len(messages) == 5  # appended, nothing replaced

    def test_replaces_the_system_prompt_in_its_call(self):
        seen = []

        def capture(messages):
            seen.extend(messages)
            return "answer"

        hook = BackupAnswerHook(generator=MockChatGenerator(response_fn=capture))
        state = State(schema={})
        state.set(
            "messages",
            [
                ChatMessage.from_system("agent system prompt"),
                ChatMessage.from_user("q"),
                ChatMessage.from_tool(tool_result="r", origin=SEARCH_CALL),
            ],
        )

        hook.run(state)

        system_messages = [m for m in seen if m.is_from(ChatRole.SYSTEM)]
        assert len(system_messages) == 1
        assert "interrupted document-search session" in system_messages[0].text
        assert "agent system prompt" not in system_messages[0].text

    def test_is_a_noop_when_the_run_answered(self):
        generator = MockChatGenerator("should not run")
        hook = BackupAnswerHook(generator=generator)
        state = State(schema={})
        state.set("messages", [ChatMessage.from_user("q"), ChatMessage.from_assistant("proper answer")])

        hook.run(state)

        assert state.get("messages")[-1].text == "proper answer"
        assert generator._call_count == 0

    def test_lifecycle_methods_delegate_to_the_generator(self):
        class _LifecycleGenerator(MockChatGenerator):
            warmed_up = 0
            closed = 0

            def warm_up(self):
                self.warmed_up += 1

            def close(self):
                self.closed += 1

        generator = _LifecycleGenerator()
        hook = BackupAnswerHook(generator=generator)
        hook.warm_up()
        hook.close()
        assert generator.warmed_up == 1
        assert generator.closed == 1

    def test_lifecycle_methods_are_noops_without_generator_support(self):
        class _BareGenerator:
            pass

        hook = BackupAnswerHook(generator=_BareGenerator())
        hook.warm_up()  # must not raise
        hook.close()  # must not raise

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY required")
    def test_live_backup_answer_uses_the_gathered_evidence(self):
        # Simulate a run interrupted by max_agent_steps: evidence was retrieved, no answer written.
        # The Responses API requires the ids that real runs carry: an `fc_...` item id and a
        # `call_...` call id (stored in `ToolCall.extra`).
        search_call = ToolCall(
            tool_name="search_documents",
            arguments={"query": "gravitational waves detection"},
            id="fc_1",
            extra={"call_id": "call_1"},
        )
        state = State(schema={})
        state.set(
            "messages",
            [
                ChatMessage.from_system("You answer questions using ONLY documents retrieved from a document store."),
                ChatMessage.from_user("When were gravitational waves first detected?"),
                ChatMessage.from_assistant(tool_calls=[search_call]),
                ChatMessage.from_tool(
                    tool_result=(
                        '[doc 9e932483]\n  meta: {"category": "science", "year": 2016}\n  '
                        "The LIGO observatory detected gravitational waves from two merging black holes "
                        "for the first time in 2016."
                    ),
                    origin=search_call,
                ),
            ],
        )

        BackupAnswerHook(generator=_default_llm("gpt-5.4")).run(state)

        answer = state.get("messages")[-1]
        assert answer.is_from(ChatRole.ASSISTANT)
        assert answer.text
        assert "2016" in answer.text  # drawn from the gathered evidence, not general knowledge refusal

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY required")
    def test_live_backup_answer_acknowledges_insufficient_evidence(self):
        # Interrupted before any retrieval: only a metadata listing was gathered, nothing to answer from.
        list_call = ToolCall(tool_name="list_metadata_fields", arguments={}, id="fc_1", extra={"call_id": "call_1"})
        state = State(schema={})
        state.set(
            "messages",
            [
                ChatMessage.from_system("You answer questions using ONLY documents retrieved from a document store."),
                ChatMessage.from_user("What do the science documents from 2021 say?"),
                ChatMessage.from_assistant(tool_calls=[list_call]),
                ChatMessage.from_tool(
                    tool_result=(
                        "Metadata fields (add the 'meta.' prefix when filtering):\n- category (keyword)\n- year (int)"
                    ),
                    origin=list_call,
                ),
            ],
        )

        BackupAnswerHook(generator=_default_llm("gpt-5.4")).run(state)

        answer = state.get("messages")[-1]
        assert answer.is_from(ChatRole.ASSISTANT)
        assert "no matching information was found" in (answer.text or "").lower()

    def test_serialization_roundtrip(self):
        hook = BackupAnswerHook(generator=OpenAIResponsesChatGenerator(model="gpt-5.4"))
        data = hook.to_dict()
        assert data["type"].endswith("BackupAnswerHook")

        restored = BackupAnswerHook.from_dict(data)
        assert isinstance(restored, BackupAnswerHook)
        assert restored.generator.model == "gpt-5.4"
