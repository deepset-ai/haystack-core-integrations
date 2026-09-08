"""
Haystack Agent with EverOS long-term memory tools.

Prerequisites:
    pip install everos-haystack openai

Environment variables:
    EVEROS_CLOUD_API_KEY - API key used by EverOS Cloud
    OPENAI_API_KEY - API key used by the Haystack chat generator
"""

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.memory_stores.everos import EverOSMemoryStore
from haystack_integrations.tools.everos import EverOSMemoryRetrieverTool, EverOSMemoryWriterTool


def main() -> None:
    """Run one Agent turn with EverOS recall and explicit memory writing tools."""
    store = EverOSMemoryStore()
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-5.4-mini"),
        tools=[
            EverOSMemoryRetrieverTool(memory_store=store, top_k=5, include_profile=True),
            EverOSMemoryWriterTool(memory_store=store, flush_on_write=True),
        ],
        system_prompt=(
            "You are a helpful assistant with long-term memory. Search memory before answering when earlier "
            "context may help. Store only new durable user facts, preferences, and decisions."
        ),
        state_schema={"user_id": {"type": str}, "session_id": {"type": str}},
    )
    result = agent.run(
        messages=[ChatMessage.from_user("Remember that I prefer concise Python examples.")],
        user_id="alice",
        session_id="everos-demo-1",
    )
    print(result["last_message"].text)  # noqa: T201
    store.close()


if __name__ == "__main__":
    main()
