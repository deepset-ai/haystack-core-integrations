"""
Example: Haystack Agent with Mem0 long-term memory tools.

The Agent can call `retrieve_memories` before answering to recall relevant context, and `store_memory` after
answering to persist new facts for future sessions. By default, the writer tool stores the model-selected memory text
directly with `infer=False`, so the Agent decides exactly what should become a memory.

The `user_id` is passed to each `agent.run()` call and injected into the tools via State, so a single Agent instance
can serve multiple users without rebuilding.

Prerequisites:
    pip install mem0-haystack openai

Environment variables:
    MEM0_API_KEY   - Your Mem0 cloud API key
    OPENAI_API_KEY - Your OpenAI API key
"""

import time

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.mem0 import Mem0MemoryStore
from haystack_integrations.tools.mem0 import Mem0MemoryRetrieverTool, Mem0MemoryWriterTool

USER_ID = "demo-user-2"
MEMORY_INDEXING_WAIT_SECONDS = 10


def wait_for_memory_indexing(seconds: int) -> None:
    """Print a countdown while Mem0 makes newly written memories searchable."""
    print("Waiting for Mem0 to make new memories searchable:")  # noqa: T201
    for remaining in range(seconds, 0, -1):
        print(f"  {remaining} second(s) remaining...", end="\r", flush=True)  # noqa: T201
        time.sleep(1)
    print("  done.\n")  # noqa: T201


def run_conversation(agent: Agent, turns: list[str], title: str) -> None:
    """Run one conversation with local chat history isolated to that session."""
    print(f"\n=== {title} ===")  # noqa: T201
    history: list[ChatMessage] = []

    for user_text in turns:
        print(f"\nUser: {user_text}\n")  # noqa: T201
        history.append(ChatMessage.from_user(user_text))
        result = agent.run(messages=history, user_id=USER_ID)
        history.append(result["last_message"])


def main() -> None:  # noqa: D103
    store = Mem0MemoryStore()

    retriever_tool = Mem0MemoryRetrieverTool(memory_store=store, top_k=5)
    writer_tool = Mem0MemoryWriterTool(memory_store=store)

    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-5.4-mini"),
        tools=[retriever_tool, writer_tool],
        system_prompt="""You are a helpful assistant with long-term memory.

Use the memory tools deliberately:
- Call `retrieve_memories` with a focused query when stored memories are likely to affect the answer.
- Call `retrieve_memories` without a query when the user asks what you remember or when a broad memory inventory is
  more useful than a targeted search.
- Do not call `retrieve_memories` when the current message is self-contained and stored context would not help.
- When the user shares durable, user-specific facts or preferences, call `store_memory` before your final answer.
  Do not wait for the user to explicitly say "remember." Write concise standalone memory text. Do not store transient
  requests or facts that are only useful inside the current conversation.
""",
        streaming_callback=print_streaming_chunk,
        state_schema={"user_id": {"type": str}},
    )

    first_conversation = [
        "Hi! My name is Alice and I work on internal developer tooling at Acme.",
        (
            "I'm building a Haystack and Mem0 assistant for our docs team. "
            "Please remember that I prefer concise answers with Python snippets."
        ),
        (
            "One more thing: my teammate Priya owns evaluation and Marco handles deployment. "
            "Can you suggest the next implementation step?"
        ),
    ]
    run_conversation(agent, first_conversation, "Conversation 1: write long-term memories")

    wait_for_memory_indexing(MEMORY_INDEXING_WAIT_SECONDS)

    second_conversation = [
        "Hi again. What do you remember about my current project and answer style?",
        "Who on my team should I ask about evaluation and deployment?",
        "Give me a short Python-oriented tip for improving the memory workflow.",
    ]
    run_conversation(agent, second_conversation, "Conversation 2: recall memories in a new session")


if __name__ == "__main__":
    main()
