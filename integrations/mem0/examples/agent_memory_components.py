"""
Example: Haystack Agent pipeline with Mem0 memory components.

This demo shows a memory-augmented Agent built with Haystack components instead of using the prebuilt Agent tools:

1. Pre-seeded memories are written to Mem0 before the first conversation starts.
2. Before each turn, Mem0MemoryRetriever fetches memories relevant to the user message.
3. OutputAdapter prepends retrieved memories to live chat history and the current user message.
4. Agent answers with the combined context.
5. Mem0MemoryWriter stores the Agent's full message trace with inference enabled, giving Mem0 the
   user message, any tool-call context, and the final assistant response when extracting memories.
6. A second conversation starts with empty chat history and recalls facts introduced in the first conversation.

Pipeline structure per turn:

    query -> Mem0MemoryRetriever -> memories ---.
                                                \
    history + user message ----------------------> OutputAdapter -> Agent -> messages
                                                                                 |
                                                                                 v
                                                                           Mem0MemoryWriter

Prerequisites:
    pip install mem0-haystack openai

Environment variables:
    MEM0_API_KEY   - Your Mem0 cloud API key
    OPENAI_API_KEY - Your OpenAI API key
"""

import time

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.retrievers.mem0 import Mem0MemoryRetriever
from haystack_integrations.components.writers.mem0 import Mem0MemoryWriter
from haystack_integrations.memory_stores.mem0 import Mem0MemoryStore

USER_ID = "mem0-agent-pipeline-demo-user"
MEMORY_INDEXING_WAIT_SECONDS = 15

SEEDED_MEMORIES = [
    "My name is Alice. I am a senior data scientist at Acme Corp specializing in NLP.",
    "My current project is building an internal documentation search system powered by Haystack and Mem0.",
    "My team: Bob is the ML engineer and Carol handles infrastructure.",
    "I prefer concise answers with Python code examples over long prose explanations.",
]

FIRST_CONVERSATION_TURNS = [
    "Hi! Can you remind me what project I am currently working on?",
    (
        "Quick update for memory: I switched the prototype vector database to Qdrant Cloud, "
        "and the main blocker is extracting tables cleanly from PDFs."
    ),
    (
        "Also remember that Dana joined the team to own evaluation, and I want future code examples "
        "to be fully documented with api doc strings."
    ),
]

SECOND_CONVERSATION_TURNS = [
    "What vector database did I switch the prototype to, and what is the current blocker?",
    "Who owns evaluation now, and how should you tailor code examples for me?",
]

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to persistent memory from past conversations. "
    "System messages at the start of the conversation contain relevant memories retrieved from Mem0. "
    "Use them to personalize responses and maintain continuity across turns. "
    "Be concise and prefer short Python examples when useful."
)


def build_memory_agent_pipeline(store: Mem0MemoryStore) -> Pipeline:
    """
    Build a pipeline that retrieves memories, injects them into Agent context, and writes new memories after each turn.

    OutputAdapter merges retrieved memories, live history, and the current user message into one ChatMessage list for
    the Agent. The Agent's `messages` output is connected directly to Mem0MemoryWriter so Mem0 can infer memories
    from the full turn context.
    """
    pipeline = Pipeline()

    pipeline.add_component("retriever", Mem0MemoryRetriever(memory_store=store, top_k=5))
    pipeline.add_component(
        "memory_injector",
        OutputAdapter(
            template="{{ memories + history + user_messages }}",
            output_type=list[ChatMessage],
            unsafe=True,
        ),
    )
    pipeline.add_component(
        "agent",
        Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-5.4-mini"),
            system_prompt=SYSTEM_PROMPT,
        ),
    )
    pipeline.add_component("writer", Mem0MemoryWriter(memory_store=store, infer=True))

    pipeline.connect("retriever.memories", "memory_injector.memories")
    pipeline.connect("memory_injector.output", "agent.messages")
    pipeline.connect("agent.messages", "writer.messages")

    return pipeline


def seed_memories(store: Mem0MemoryStore) -> None:
    """Seed facts that simulate memories from a previous session."""
    messages = [ChatMessage.from_user(fact) for fact in SEEDED_MEMORIES]
    store.add_memories(messages=messages, user_id=USER_ID, infer=False)


def run_turn(pipeline: Pipeline, user_text: str, history: list[ChatMessage]) -> str:
    """Run one conversation turn, update clean user/assistant history, and return the reply."""
    user_message = ChatMessage.from_user(user_text)
    result = pipeline.run(
        {
            "retriever": {"query": user_text, "user_id": USER_ID},
            "memory_injector": {
                "history": history,
                "user_messages": [user_message],
            },
            "writer": {"user_id": USER_ID},
        }
    )

    last_message = result["agent"]["last_message"]
    reply = last_message.text or "(no text reply)"

    history.append(user_message)
    history.append(last_message)

    return reply


def wait_for_memory_indexing(seconds: int) -> None:
    """Print a countdown while Mem0 makes newly written memories searchable."""
    print("Waiting for Mem0 to make new memories searchable:")  # noqa: T201
    for remaining in range(seconds, 0, -1):
        print(f"  {remaining} second(s) remaining...", end="\r", flush=True)  # noqa: T201
        time.sleep(1)
    print("  done.\n")  # noqa: T201


def main() -> None:
    """Run the Mem0 memory Agent pipeline demo."""
    print("=== Mem0 Memory Agent Pipeline Demo ===\n")  # noqa: T201

    store = Mem0MemoryStore()
    seed_memories(store)
    print(f"Seeded {len(SEEDED_MEMORIES)} memories for {USER_ID}.\n")  # noqa: T201
    print("Seeded memories:")  # noqa: T201
    for memory in store.search_memories(query="", top_k=10, user_id=USER_ID):
        print(f"- {memory.text}")  # noqa: T201

    pipeline = build_memory_agent_pipeline(store)
    first_history: list[ChatMessage] = []

    print("\n==Starting first conversation...==\n")  # noqa: T201
    for user_text in FIRST_CONVERSATION_TURNS:
        print(f"User:  {user_text}")  # noqa: T201
        reply = run_turn(pipeline, user_text, first_history)
        print(f"Agent: {reply}\n")  # noqa: T201

    wait_for_memory_indexing(MEMORY_INDEXING_WAIT_SECONDS)

    second_history: list[ChatMessage] = []
    print("==Starting second conversation with empty local history...==\n")  # noqa: T201
    for user_text in SECOND_CONVERSATION_TURNS:
        print(f"User:  {user_text}")  # noqa: T201
        reply = run_turn(pipeline, user_text, second_history)
        print(f"Agent: {reply}\n")  # noqa: T201

    print("=== Done ===")  # noqa: T201


if __name__ == "__main__":
    main()
