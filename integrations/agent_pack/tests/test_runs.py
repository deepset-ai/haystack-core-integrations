import pytest
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.agent_pack.runs import AgentRunRecord, AgentRunRecorder, LocalRunStore, RunSelection


def test_recorder_persists_replayable_inputs_and_outputs(tmp_path):
    store = LocalRunStore(tmp_path)
    agent = Agent(chat_generator=MockChatGenerator([ChatMessage.from_assistant("answer")]))

    recorded = AgentRunRecorder(store).run(agent, messages=[ChatMessage.from_user("question")])

    restored = LocalRunStore(tmp_path).list()[0]
    assert recorded.record == restored
    assert restored.inputs["messages"][0].text == "question"
    assert restored.outputs["last_message"].text == "answer"
    assert restored.fingerprint() == recorded.record.fingerprint()


def test_store_selects_records_by_id_and_limit():
    store = LocalRunStore()
    for run_id in ("one", "two", "three"):
        store.add(AgentRunRecord(run_id=run_id, inputs={"messages": []}, outputs={}))

    selection = RunSelection(run_ids=frozenset({"one", "three"}), limit=1)
    assert [record.run_id for record in store.list(selection)] == ["three"]


def test_run_fingerprint_depends_on_content_not_storage_identity():
    first = AgentRunRecord(run_id="first", inputs={"messages": []}, outputs={"answer": "same"})
    second = AgentRunRecord(run_id="second", inputs=first.inputs, outputs=first.outputs)
    changed = AgentRunRecord(run_id="first", inputs=first.inputs, outputs={"answer": "changed"})

    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() != changed.fingerprint()


@pytest.mark.asyncio
async def test_recorder_supports_async_runs():
    agent = Agent(chat_generator=MockChatGenerator([ChatMessage.from_assistant("answer")]))
    recorded = await AgentRunRecorder().run_async(agent, messages=[ChatMessage.from_user("question")])
    assert recorded.result["last_message"].text == "answer"
