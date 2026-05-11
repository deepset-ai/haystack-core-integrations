# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.types.protocol import MemoryStore


@component
class MemoryWriter:
    """
    Writes ChatMessage objects as memories to any MemoryStore implementation.

    Analogous to Haystack's DocumentWriter, this component is generic and works with
    any store that implements the MemoryStore protocol.

    Use this component in a Haystack Pipeline to persist conversation messages, or wrap it
    with ComponentTool to expose memory writing as an Agent tool.
    """

    def __init__(self, *, memory_store: MemoryStore) -> None:
        """
        Initialize the MemoryWriter.

        :param memory_store: Any object implementing the MemoryStore protocol.
        """
        self.memory_store = memory_store

    @component.output_types(memories_written=int)
    def run(
        self,
        messages: list[ChatMessage],
        *,
        user_id: str | None = None,
    ) -> dict[str, int]:
        """
        Write messages as memories to the store.

        :param messages: List of ChatMessage objects to store.
        :param user_id: User ID to scope the stored memories.
        :returns: Dictionary with key `memories_written` containing the count of stored memory items,
            or 0 if the store does not report a count.
        """
        result = self.memory_store.add_memories(
            messages=messages,
            user_id=user_id,
        )
        count = len(result) if isinstance(result, list) else 0
        return {"memories_written": count}

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            memory_store=self.memory_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryWriter":
        """Deserialize this component from a dictionary."""
        init_params = data.get("init_parameters", {})
        if init_params.get("memory_store"):
            store_data = init_params["memory_store"]
            store_type_path: str = store_data.get("type", "")
            module_path, _, class_name = store_type_path.rpartition(".")
            if module_path and class_name:
                module = importlib.import_module(module_path)
                store_cls = getattr(module, class_name)
                data["init_parameters"]["memory_store"] = default_from_dict(store_cls, store_data)
        return default_from_dict(cls, data)
