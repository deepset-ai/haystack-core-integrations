import os

import pytest
from haystack import Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.components.generators.chat import MockChatGenerator, OpenAIResponsesChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.hooks import hook
from haystack.tools import Toolset, tool

from haystack_integrations.agent_pack.advanced_rag.agent import _default_llm, create_advanced_rag_agent
from haystack_integrations.agent_pack.advanced_rag.hooks import BackupAnswerHook
from haystack_integrations.agent_pack.advanced_rag.tools import DocumentStoreToolset

EXPECTED_TOOL_NAMES = [
    "list_metadata_fields",
    "get_metadata_field_values",
    "get_metadata_field_range",
    "fetch_documents_by_filter",
    "search_documents",
]


def _flat_tool_names(tools):
    return [t.name for item in tools for t in (item if isinstance(item, Toolset) else [item])]


@pytest.fixture
def store():
    document_store = InMemoryDocumentStore()
    document_store.write_documents(
        [
            Document(content="CRISPR gene editing breakthrough", meta={"category": "science", "year": 2021}),
            Document(content="The fall of the Berlin Wall", meta={"category": "history", "year": 1989}),
        ]
    )
    return document_store


def test_default_llm_applies_pack_settings():
    llm = _default_llm("gpt-5.4")
    assert isinstance(llm, OpenAIResponsesChatGenerator)
    assert llm.model == "gpt-5.4"
    assert llm.timeout == 180.0
    assert llm.max_retries == 5
    assert llm.generation_kwargs == {"reasoning": {"effort": "low"}}


class TestFactoryValidation:
    def test_requires_exactly_one_retrieval_source(self, store):
        with pytest.raises(ValueError, match="exactly one"):
            create_advanced_rag_agent(document_store=store, llm=MockChatGenerator())
        with pytest.raises(ValueError, match="exactly one"):
            create_advanced_rag_agent(
                document_store=store,
                retriever=InMemoryBM25Retriever(document_store=store),
                retrieval_pipeline=Pipeline(),
                llm=MockChatGenerator(),
            )

    def test_pipeline_path_requires_input_mapping(self, store):
        with pytest.raises(ValueError, match="retrieval_pipeline_input_mapping"):
            create_advanced_rag_agent(document_store=store, retrieval_pipeline=Pipeline(), llm=MockChatGenerator())

    def test_rejects_pipeline_arguments_on_the_retriever_path(self, store):
        with pytest.raises(ValueError, match="only valid with"):
            create_advanced_rag_agent(
                document_store=store,
                retriever=InMemoryBM25Retriever(document_store=store),
                retrieval_pipeline_output_mapping={"retriever.documents": "documents"},
                llm=MockChatGenerator(),
            )


class TestCreateAdvancedRagAgent:
    def test_wires_the_tools_state_and_hooks(self, store):
        agent = create_advanced_rag_agent(
            document_store=store,
            retriever=InMemoryBM25Retriever(document_store=store),
            llm=MockChatGenerator(),
            max_agent_steps=7,
            max_fetched_docs=4,
        )
        assert isinstance(agent, Agent)
        assert _flat_tool_names(agent.tools) == EXPECTED_TOOL_NAMES
        assert isinstance(agent.tools[0], DocumentStoreToolset)
        assert agent.tools[0].max_fetched_docs == 4
        assert agent.exit_conditions == ["text"]
        assert agent.max_agent_steps == 7
        assert "documents" in agent.state_schema
        assert [type(h) for h in agent.hooks["after_run"]] == [BackupAnswerHook]
        assert "list_metadata_fields" in agent.system_prompt

    def test_supports_a_retrieval_pipeline(self, store):
        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
        agent = create_advanced_rag_agent(
            document_store=store,
            retrieval_pipeline=pipeline,
            retrieval_pipeline_input_mapping={"query": ["retriever.query"], "filters": ["retriever.filters"]},
            retrieval_pipeline_output_mapping={"retriever.documents": "documents"},
            llm=MockChatGenerator(),
        )
        assert _flat_tool_names(agent.tools) == EXPECTED_TOOL_NAMES

    def test_supports_extra_tools_state_schema_and_hooks(self, store):
        @tool
        def extra_tool(text: str) -> str:
            """Do something extra."""
            return text

        @hook
        def custom_after_run(state: State) -> None:  # noqa: ARG001
            return None

        @hook
        def custom_before_llm(state: State) -> None:  # noqa: ARG001
            return None

        agent = create_advanced_rag_agent(
            document_store=store,
            retriever=InMemoryBM25Retriever(document_store=store),
            llm=MockChatGenerator(),
            extra_tools=[extra_tool],
            state_schema={"notes": {"type": list}},
            hooks={"after_run": [custom_after_run], "before_llm": [custom_before_llm]},
            raise_on_tool_invocation_failure=True,
            tool_concurrency_limit=2,
        )

        assert _flat_tool_names(agent.tools) == [*EXPECTED_TOOL_NAMES, "extra_tool"]
        assert set(agent.state_schema) >= {"notes", "documents"}
        # The built-in backup-answer hook runs first, custom after_run hooks after it.
        assert [type(h) for h in agent.hooks["after_run"]] == [BackupAnswerHook, type(custom_after_run)]
        assert agent.hooks["before_llm"] == [custom_before_llm]
        assert agent.raise_on_tool_invocation_failure is True
        assert agent.tool_concurrency_limit == 2

    def test_built_in_documents_state_entry_wins_over_custom(self, store):
        agent = create_advanced_rag_agent(
            document_store=store,
            retriever=InMemoryBM25Retriever(document_store=store),
            llm=MockChatGenerator(),
            state_schema={"documents": {"type": str}},
        )
        assert agent.state_schema["documents"]["type"] == list[Document]

    def test_accepts_a_custom_system_prompt(self, store):
        agent = create_advanced_rag_agent(
            document_store=store,
            retriever=InMemoryBM25Retriever(document_store=store),
            llm=MockChatGenerator(),
            system_prompt="my prompt",
        )
        assert agent.system_prompt == "my prompt"

    def test_serialization_roundtrip(self, store):
        # No allow_deserialization_module needed: `haystack_integrations` is on the default allowlist.
        agent = create_advanced_rag_agent(
            document_store=store, retriever=InMemoryBM25Retriever(document_store=store), llm=MockChatGenerator()
        )

        restored = Agent.from_dict(agent.to_dict())

        assert isinstance(restored, Agent)
        assert _flat_tool_names(restored.tools) == EXPECTED_TOOL_NAMES
        assert isinstance(restored.tools[0], DocumentStoreToolset)
        assert [type(h) for h in restored.hooks["after_run"]] == [BackupAnswerHook]
        assert "documents" in restored.state_schema


class TestAdvancedRagAgentRun:
    def test_accumulates_retrieved_documents(self, store):
        responses = [
            ChatMessage.from_assistant(
                tool_calls=[
                    ToolCall(
                        tool_name="search_documents",
                        arguments={
                            "query": "gene editing",
                            "filters": {"field": "meta.category", "operator": "==", "value": "science"},
                        },
                    )
                ]
            ),
            ChatMessage.from_assistant("CRISPR corrected a mutation. [doc ref]"),
        ]
        agent = create_advanced_rag_agent(
            document_store=store,
            retriever=InMemoryBM25Retriever(document_store=store, top_k=3),
            llm=MockChatGenerator(responses=responses),
        )

        result = agent.run(messages=[ChatMessage.from_user("What do the science documents say?")])

        documents = result["documents"]
        assert [d.meta["category"] for d in documents] == ["science"]
        assert result["last_message"].text == "CRISPR corrected a mutation. [doc ref]"

    def test_writes_backup_answer_on_step_exhaustion(self, store):
        search_call = ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="search_documents", arguments={"query": "wall"})]
        )
        # One step only: the LLM requests a tool call, the step budget is spent, no answer is written.
        # The next (cycled) response is consumed by the BackupAnswerHook instead.
        agent = create_advanced_rag_agent(
            document_store=store,
            retriever=InMemoryBM25Retriever(document_store=store),
            llm=MockChatGenerator(responses=[search_call, ChatMessage.from_assistant("the backup answer")]),
            max_agent_steps=1,
        )

        result = agent.run(messages=[ChatMessage.from_user("q")])

        assert result["step_count"] == 1
        assert result["last_message"].text == "the backup answer"

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY required")
    def test_live_run(self):
        document_store = InMemoryDocumentStore()
        document_store.write_documents(
            [
                Document(
                    content="CRISPR corrected a hereditary blindness mutation.",
                    meta={"category": "science", "year": 2021},
                ),
                Document(
                    content="A quantum computer demonstrated error-corrected qubits.",
                    meta={"category": "science", "year": 2023},
                ),
                Document(content="The Berlin Wall fell.", meta={"category": "history", "year": 1989}),
            ]
        )
        agent = create_advanced_rag_agent(
            document_store=document_store, retriever=InMemoryBM25Retriever(document_store=document_store, top_k=3)
        )

        result = agent.run(messages=[ChatMessage.from_user("What do the science documents from after 2020 say?")])

        assert result["last_message"].text.strip()
        documents = result["documents"]
        assert documents and all(d.meta["category"] == "science" for d in documents)
        assert result["tool_call_counts"]["list_metadata_fields"] >= 1
