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


def test_clear_removes_records_from_memory_and_disk(tmp_path):
    """A recorded run says nothing about which Agent produced it, so a store has to be emptiable."""
    store = LocalRunStore(directory=tmp_path / "runs")
    store.add(record=AgentRunRecord(run_id="first", inputs={"messages": []}, outputs={"answer": "a"}))
    store.add(record=AgentRunRecord(run_id="second", inputs={"messages": []}, outputs={"answer": "b"}))
    assert len(store.list()) == 2
    assert len(list((tmp_path / "runs").glob("*.json"))) == 2

    store.clear()

    assert store.list() == []
    assert list((tmp_path / "runs").glob("*.json")) == []
    # A store reopened on the cleared directory finds nothing either.
    assert LocalRunStore(directory=tmp_path / "runs").list() == []


def test_clear_is_safe_without_a_directory():
    """An in-memory store clears without touching a filesystem it does not have."""
    store = LocalRunStore()
    store.add(record=AgentRunRecord(run_id="only", inputs={"messages": []}, outputs={"answer": "a"}))

    store.clear()

    assert store.list() == []
