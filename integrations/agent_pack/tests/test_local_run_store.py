from haystack.dataclasses import ChatMessage

from haystack_integrations.agent_pack.dataclasses import AgentRunRecord
from haystack_integrations.agent_pack.local_run_store import LocalRunStore


def test_store_persists_replayable_inputs_and_outputs(tmp_path):
    """Persist and restore replayable Haystack values without changing the record."""
    store = LocalRunStore(directory=tmp_path)
    record = AgentRunRecord(
        run_id="run",
        inputs={"messages": [ChatMessage.from_user(text="question")]},
        outputs={"last_message": ChatMessage.from_assistant(text="answer")},
    )
    store.add(record=record)

    restored = LocalRunStore(directory=tmp_path).list()[0]
    assert record == restored
    assert restored.inputs["messages"][0].text == "question"
    assert restored.outputs["last_message"].text == "answer"
    assert restored.fingerprint() == record.fingerprint()


def test_store_selects_records_by_id():
    """Return only requested records while preserving newest-first ordering."""
    store = LocalRunStore()
    for run_id in ("one", "two", "three"):
        store.add(record=AgentRunRecord(run_id=run_id, inputs={"messages": []}, outputs={}))

    assert [record.run_id for record in store.list(run_ids=frozenset({"one", "three"}))] == ["three", "one"]
