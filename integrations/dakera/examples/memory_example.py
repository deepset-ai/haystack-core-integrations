# SPDX-FileCopyrightText: 2025-present Dakera AI <hello@dakera.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal example for the Dakera memory integration.

Prerequisites:
    * A running Dakera server (see https://github.com/dakera-ai/dakera-deploy).
    * ``export DAKERA_API_KEY=dk-...``
    * ``pip install dakera-haystack``
"""

from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from haystack_integrations.components.retrievers.dakera import DakeraMemoryRetriever
from haystack_integrations.components.writers.dakera import DakeraMemoryWriter
from haystack_integrations.memory_stores.dakera import DakeraMemoryStore

store = DakeraMemoryStore(base_url="http://localhost:3000", api_key=Secret.from_env_var("DAKERA_API_KEY"))

# Persist a couple of memories.
writer = DakeraMemoryWriter(memory_store=store)
writer.run(
    messages=[
        ChatMessage.from_user("Alice prefers concise answers."),
        ChatMessage.from_user("Alice works in the Paris office."),
    ],
    session_id="alice",
)

# Recall decay-weighted memories relevant to a query.
retriever = DakeraMemoryRetriever(memory_store=store, top_k=5)
result = retriever.run(query="Where does Alice work?", session_id="alice")
for message in result["memories"]:
    print(f"{message.meta.get('score')}: {message.text}")
