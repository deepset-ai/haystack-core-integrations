# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.writers.cognee import CogneeWriter
from haystack_integrations.memory_stores.cognee import CogneeMemoryStore


class TestCogneeWriter:
    def test_init_requires_memory_store(self):
        with pytest.raises(ValueError, match="memory_store must be an instance of CogneeMemoryStore"):
            CogneeWriter(memory_store="not a store")  # type: ignore[arg-type]

    def test_init_holds_store(self):
        store = CogneeMemoryStore(dataset_name="ds", session_id="s")
        writer = CogneeWriter(memory_store=store)
        assert writer._memory_store is store
        assert writer._session_id is None

    def test_init_with_session_id(self):
        store = CogneeMemoryStore(dataset_name="ds")
        writer = CogneeWriter(memory_store=store, session_id="override")
        assert writer._session_id == "override"

    def test_to_from_dict_roundtrip(self):
        store = CogneeMemoryStore(dataset_name="ds", session_id="s")
        writer = CogneeWriter(memory_store=store, session_id="writer_sess")

        data = writer.to_dict()
        assert data["type"] == "haystack_integrations.components.writers.cognee.memory_writer.CogneeWriter"
        assert data["init_parameters"]["session_id"] == "writer_sess"
        assert (
            data["init_parameters"]["memory_store"]["type"]
            == "haystack_integrations.memory_stores.cognee.memory_store.CogneeMemoryStore"
        )
        assert data["init_parameters"]["memory_store"]["init_parameters"]["dataset_name"] == "ds"

        restored = CogneeWriter.from_dict(data)
        assert isinstance(restored._memory_store, CogneeMemoryStore)
        assert restored._memory_store.dataset_name == "ds"
        assert restored._memory_store.session_id == "s"
        assert restored._session_id == "writer_sess"

    def test_run_delegates_to_store_and_echoes_messages(self):
        store = MagicMock(spec=CogneeMemoryStore)
        writer = CogneeWriter(memory_store=store)

        messages = [ChatMessage.from_user("hi"), ChatMessage.from_assistant("hello")]
        out = writer.run(messages=messages)

        store.add_memories.assert_called_once_with(messages=messages, user_id=None, session_id=None)
        assert out == {"messages_written": messages}

    def test_run_passes_user_id(self):
        store = MagicMock(spec=CogneeMemoryStore)
        writer = CogneeWriter(memory_store=store)

        messages = [ChatMessage.from_user("hi")]
        writer.run(messages=messages, user_id="user-abc")

        store.add_memories.assert_called_once_with(messages=messages, user_id="user-abc", session_id=None)

    def test_run_forwards_writer_session_id(self):
        store = MagicMock(spec=CogneeMemoryStore)
        writer = CogneeWriter(memory_store=store, session_id="writer_sess")

        messages = [ChatMessage.from_user("hi")]
        writer.run(messages=messages)

        store.add_memories.assert_called_once_with(messages=messages, user_id=None, session_id="writer_sess")
