"""
Example: Haystack Agent with Mem0 long-term memory tools.

The Agent can call `retrieve_memories` before answering to recall relevant context,
and `store_memory` after answering to persist new facts for future sessions.

Prerequisites:
    pip install mem0-haystack openai

Environment variables:
    MEM0_API_KEY   - Your Mem0 cloud API key
    OPENAI_API_KEY - Your OpenAI API key
"""

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.mem0 import Mem0MemoryStore
from haystack_integrations.tools.mem0 import Mem0MemoryRetrieverTool, Mem0MemoryWriterTool

USER_ID = "demo-user"


def main() -> None:  # noqa: D103
    store = Mem0MemoryStore()

    retriever_tool = Mem0MemoryRetrieverTool(memory_store=store, user_id=USER_ID, top_k=5)
    writer_tool = Mem0MemoryWriterTool(memory_store=store, user_id=USER_ID)

    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
        tools=[retriever_tool, writer_tool],
        system_prompt="""You are a helpful assistant with long-term memory.

Before answering, call `retrieve_memories` to recall relevant facts about this user.
After answering, call `store_memory` to save any important new information the user shared.
""",
    )

    turns = [
        "Hi! My name is Alice and I love working with Haystack.",
        "What do you know about me so far?",
        "I'm also a big fan of hiking in the Alps.",
        "Based on what you know about me, suggest a weekend activity.",
    ]

    history: list[ChatMessage] = []
    for user_text in turns:
        print(f"\nUser: {user_text}")  # noqa: T201
        history.append(ChatMessage.from_user(user_text))
        result = agent.run(messages=history)
        last = result["last_message"]
        reply = last.text or "(no text reply)"
        print(f"Agent: {reply}")  # noqa: T201
        history.append(last)


if __name__ == "__main__":
    main()
