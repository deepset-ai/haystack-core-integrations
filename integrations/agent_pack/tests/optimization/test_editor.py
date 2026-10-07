import pytest
from haystack import Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.rankers import LLMRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent
from haystack_integrations.agent_pack.optimization.editor import ConfigurationEditor
from haystack_integrations.agent_pack.optimization.utils import _configuration_id, load_agent


def agent_yaml(agent=None):
    pipeline = Pipeline()
    pipeline.add_component("agent", agent or Agent(chat_generator=MockChatGenerator(model="reference")))
    return pipeline.dumps()


def model_yaml(model):
    return agent_yaml(Agent(chat_generator=MockChatGenerator(model=model)))


class TestEdit:
    def test_requires_unique_match_and_revision(self):
        editor = ConfigurationEditor(reference_yaml=agent_yaml())
        original = editor.read_config()
        with pytest.raises(ValueError, match="exactly once"):
            editor.edit_config("", "x", original["revision"])
        changed = editor.edit_config("model: reference", "model: cheap", original["revision"])
        with pytest.raises(ValueError, match="Stale"):
            editor.edit_config("model: cheap", "model: other", original["revision"])
        assert changed["revision"] != original["revision"]
        assert editor.validate_config()["valid"]
        editor.submit_candidate(changed["revision"], "cheaper model")
        assert load_agent(editor.submitted.yaml).chat_generator.model == "cheap"

    def test_revalidates_on_every_edit(self):
        editor = ConfigurationEditor(reference_yaml=agent_yaml())
        revision = editor.read_config()["revision"]
        changed = editor.edit_config("model: reference", "model: [broken", revision)
        assert editor.validate_config()["valid"] is False
        with pytest.raises(ValueError, match="Validate"):
            editor.submit_candidate(changed["revision"], "broken")
        repaired = editor.edit_config("model: [broken", "model: cheap", changed["revision"])
        assert editor.validate_config()["valid"]
        editor.edit_config("model: cheap", "model: other", repaired["revision"])
        with pytest.raises(ValueError, match="Validate"):
            editor.submit_candidate(repaired["revision"], "stale validation")

    def test_turn_ends_on_submit(self):
        editor = ConfigurationEditor(reference_yaml=agent_yaml())
        changed = editor.edit_config("model: reference", "model: cheap", editor.read_config()["revision"])
        editor.validate_config()
        editor.submit_candidate(changed["revision"], "cheaper model")
        with pytest.raises(ValueError, match="has ended"):
            editor.edit_config("model: cheap", "model: other", changed["revision"])

    def test_replacing_a_retriever_and_repairing_it(self):
        store = InMemoryDocumentStore()
        document = Document(content="Berlin is in Germany")
        store.write_documents([document])
        reference = create_advanced_rag_agent(
            document_store=store,
            retriever=InMemoryBM25Retriever(document_store=store),
            llm=MockChatGenerator("ok"),
            backup_answer_llm=MockChatGenerator("ok"),
        )
        editor = ConfigurationEditor(reference_yaml=agent_yaml(reference))
        pipeline = Pipeline()
        pipeline.add_component("bm25", InMemoryBM25Retriever(document_store=store, top_k=20))
        pipeline.add_component(
            "ranker",
            LLMRanker(
                chat_generator=MockChatGenerator('{"documents": [{"index": 1}]}', model="ranker"),
                top_k=3,
            ),
        )
        pipeline.connect("bm25.documents", "ranker.documents")
        upgraded = create_advanced_rag_agent(
            document_store=store,
            retriever=pipeline,
            retrieval_pipeline_input_mapping={"query": ["bm25.query", "ranker.query"], "filters": ["bm25.filters"]},
            retrieval_pipeline_output_mapping={"ranker.documents": "documents"},
            llm=MockChatGenerator("ok"),
            backup_answer_llm=MockChatGenerator("ok"),
        )
        proposed = agent_yaml(upgraded)
        current = editor.read_config()
        changed = editor.edit_config(
            current["yaml"], proposed.replace("ranker.documents", "ranker.missing"), current["revision"]
        )
        assert not editor.validate_config()["valid"]
        current = editor.read_config()
        changed = editor.edit_config(current["yaml"], proposed, changed["revision"])
        assert editor.validate_config()["valid"]
        editor.submit_candidate(changed["revision"], "retrieve more, rerank to three")
        tool = load_agent(editor.submitted.yaml).tools[-1]
        assert tool.invoke(query="Berlin")["documents"][0].id == document.id
        assert tool.outputs_to_state["documents"]["source"] == "documents"


class TestEarlierCandidates:
    def test_starts_from_the_base(self):
        cheap = model_yaml("cheap")
        cheap_id = _configuration_id(cheap)
        editor = ConfigurationEditor(reference_yaml=agent_yaml(), candidates={cheap_id: cheap}, base_id=cheap_id)
        assert editor.read_config() == {**editor.read_config(), "yaml": cheap, "parent_id": cheap_id}

    def test_unknown_base(self):
        with pytest.raises(ValueError, match="Unknown base"):
            ConfigurationEditor(reference_yaml=agent_yaml(), base_id="never-measured")

    def test_restore_and_refuse_duplicates(self):
        cheap = model_yaml("cheap")
        cheap_id = _configuration_id(cheap)
        editor = ConfigurationEditor(reference_yaml=agent_yaml(), candidates={cheap_id: cheap}, base_id=cheap_id)
        restored = editor.restore_candidate("reference", editor.read_config()["revision"])
        assert editor.parent_id == editor.reference_id
        assert "model: reference" in editor.read_config()["yaml"]
        # A formatting-only change is the same configuration
        text = editor.read_config()["yaml"]
        changed = editor.edit_config(text, "# comment\n" + text, restored["revision"])
        editor.validate_config()
        with pytest.raises(ValueError, match="duplicate_or_no_op"):
            editor.submit_candidate(changed["revision"], "same config")
        # So is a candidate submitted on an earlier turn
        restored = editor.restore_candidate(cheap_id, changed["revision"])
        editor.validate_config()
        with pytest.raises(ValueError, match="duplicate_or_no_op"):
            editor.submit_candidate(restored["revision"], "resubmitted")


class TestEditingTools:
    def test_tool_names(self):
        """The system prompt refers to the tools by these names."""
        assert [tool.name for tool in ConfigurationEditor(reference_yaml=agent_yaml())] == [
            "read_config",
            "edit_config",
            "validate_config",
            "submit_candidate",
            "restore_candidate",
            "finish",
        ]

    def test_methods_are_bound(self):
        """The tools are the editor's own methods, so the optimizer never sees `self`."""
        for editing_tool in ConfigurationEditor(reference_yaml=agent_yaml()):
            assert "self" not in editing_tool.parameters.get("properties", {})

    def test_every_parameter_is_described(self):
        """
        `create_tool_from_function` builds the schema from `Annotated` metadata and never reads `:param` lines, so a
        parameter documented only in the docstring reaches the model as a bare string with no explanation of it.
        """
        described = {
            f"{editing_tool.name}.{name}": specification.get("description")
            for editing_tool in ConfigurationEditor(reference_yaml=agent_yaml())
            for name, specification in editing_tool.parameters.get("properties", {}).items()
        }
        assert described
        assert [parameter for parameter, description in described.items() if not description] == []
        # The constraint that actually fails at runtime has to be in the schema, not only in the error it raises.
        assert "exactly once" in described["edit_config.old"]
