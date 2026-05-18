"""
Example: Haystack Agent pipeline with Mem0 memory components.

This demo shows a memory-augmented Agent loop built with regular Haystack
components instead of prebuilt Agent tools:

1. Pre-seeded memories are written to Mem0 before the conversation starts.
2. Before each turn, Mem0MemoryRetriever fetches memories relevant to the user message.
3. OutputAdapter prepends retrieved memories to live chat history and the current user message.
4. Agent answers with the combined context.
5. Mem0MemoryWriter stores the Agent's latest reply for future turns.

Pipeline structure per turn:

    query -> Mem0MemoryRetriever -> memories -------------.
                                                            \
    history + user message ---------------------------------> OutputAdapter -> Agent -> last_message
                                                                                           |
                                                                                           v
                                                                              Mem0MemoryWriter

Prerequisites:
    pip install mem0-haystack openai

Environment variables:
    MEM0_API_KEY   - Your Mem0 cloud API key
    OPENAI_API_KEY - Your OpenAI API key
"""

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.retrievers.mem0 import Mem0MemoryRetriever
from haystack_integrations.components.writers.mem0 import Mem0MemoryWriter
from haystack_integrations.memory_stores.mem0 import Mem0MemoryStore

USER_ID = "mem0-agent-pipeline-demo-user"
AGENT_ID = "mem0-agent-pipeline-demo-agent"

SEEDED_MEMORIES = [
    "My name is Alice. I am a senior data scientist at Acme Corp specializing in NLP.",
    "My current project is building an internal documentation search system powered by Haystack and Mem0.",
    "My team: Bob is the ML engineer and Carol handles infrastructure.",
    "I prefer concise answers with Python code examples over long prose explanations.",
]

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to persistent memory from past conversations. "
    "System messages at the start of the conversation contain relevant memories retrieved from Mem0. "
    "Use them to personalize responses and maintain continuity across turns. "
    "Be concise and prefer short Python examples when useful."
)


def build_memory_agent_pipeline(store: Mem0MemoryStore) -> Pipeline:
    """
    Build a pipeline that retrieves memories, injects them into Agent context, and stores replies.

    OutputAdapter merges retrieved memories, live history, and the current user message
    into one ChatMessage list for the Agent. The Agent's `last_message` is connected
    directly to Mem0MemoryWriter, relying on smart connections to pass it to the writer's
    list[ChatMessage] input.
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
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            system_prompt=SYSTEM_PROMPT,
        ),
    )
    pipeline.add_component("writer", Mem0MemoryWriter(memory_store=store))

    pipeline.connect("retriever.memories", "memory_injector.memories")
    pipeline.connect("memory_injector.output", "agent.messages")
    pipeline.connect("agent.last_message", "writer.messages")

    return pipeline


def seed_memories(store: Mem0MemoryStore) -> None:
    """Seed facts that simulate memories from a previous session."""
    messages = [ChatMessage.from_user(fact) for fact in SEEDED_MEMORIES]
    store.add_memories(messages=messages, user_id=USER_ID, agent_id=AGENT_ID, infer=False)


def run_turn(pipeline: Pipeline, user_text: str, history: list[ChatMessage]) -> str:
    """Run one conversation turn, update clean user/assistant history, and return the reply."""
    user_message = ChatMessage.from_user(user_text)
    result = pipeline.run(
        {
            "retriever": {"query": user_text, "user_id": USER_ID, "agent_id": AGENT_ID},
            "memory_injector": {
                "history": history,
                "user_messages": [user_message],
            },
            "writer": {"user_id": USER_ID, "agent_id": AGENT_ID},
        }
    )

    last_message = result["agent"]["last_message"]
    reply = last_message.text or "(no text reply)"

    history.append(user_message)
    history.append(last_message)

    return reply


def main() -> None:
    """Run the Mem0 memory Agent pipeline demo."""
    print("=== Mem0 Memory Agent Pipeline Demo ===\n")  # noqa: T201

    store = Mem0MemoryStore(infer=False)
    seed_memories(store)
    print(f"Seeded {len(SEEDED_MEMORIES)} memories for {USER_ID}.\n")  # noqa: T201

    pipeline = build_memory_agent_pipeline(store)
    history: list[ChatMessage] = []

    turns = [
        "Hi! Can you remind me what project I am currently working on?",
        "What is the tech stack we are using for it?",
        "Who else is on my team, and what are their roles?",
        "Based on what you know about me, give me a quick tip for structuring a new Haystack pipeline component.",
    ]

    for user_text in turns:
        print(f"User:  {user_text}")  # noqa: T201
        reply = run_turn(pipeline, user_text, history)
        print(f"Agent: {reply}\n")  # noqa: T201

    print("=== Done ===")  # noqa: T201


if __name__ == "__main__":
    main()
