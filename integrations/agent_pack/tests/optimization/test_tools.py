import importlib
import json

import pytest
from haystack import Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.rankers import LLMRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.core.errors import DeserializationError
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent
from haystack_integrations.agent_pack.optimization.tools import (
    ConfigurationEditorToolset,
    _documentation_result,
    _make_haystack_documentation_toolset,
    inspect_component,
)
from haystack_integrations.agent_pack.optimization.utils import _configuration_id, load_agent


def agent_yaml(agent=None):
    pipeline = Pipeline()
    pipeline.add_component("agent", agent or Agent(chat_generator=MockChatGenerator(model="reference")))
    return pipeline.dumps()


def model_yaml(model):
    return agent_yaml(Agent(chat_generator=MockChatGenerator(model=model)))


def docs_payload(**body):
    """The shape the documentation server answers with: a payload serialized inside an MCP envelope."""
    return json.dumps({"meta": None, "content": [{"type": "text", "text": json.dumps(body)}]})


class TestDocumentationToolset:
    def test_toolset(self):
        pytest.importorskip("haystack_integrations.tools.mcp")
        docs = _make_haystack_documentation_toolset()
        assert docs.tool_names == ["search_haystack_docs"]
        assert docs.eager_connect is False


class TestDocumentationSearch:
    def test_strips_the_servers_debug_output(self):
        """Measured against the live server, the debug payload is 94% of the answer and says nothing about Haystack."""
        payload = docs_payload(
            documents=[{"content": "LLMRanker reorders documents.", "meta": {"url": "https://docs/llmranker"}}],
            _debug={"pipeline": "x" * 5000},
        )
        result = _documentation_result(payload)
        assert result == "[https://docs/llmranker]\nLLMRanker reorders documents."
        assert "_debug" not in result

    def test_unexpected_answer(self):
        """A server that changes shape must not silently look like an empty search."""
        assert _documentation_result("not json at all") == "not json at all"
        assert _documentation_result(docs_payload(unexpected=1)).startswith('{"meta"')

    def test_no_match(self):
        assert _documentation_result(docs_payload(documents=[])) == "No documentation matched."


class TestInspectComponent:
    def test_reports_the_installed_path(self):
        """The answer carries the string the YAML has to use, not the one that happened to be asked for."""
        answer = inspect_component.function(type_name="haystack.components.rankers.llm_ranker.LLMRanker")
        assert answer["import_path"] == "haystack.components.rankers.llm_ranker.LLMRanker"
        assert "top_k" in answer["constructor"]

    def test_finds_a_class_asked_for_elsewhere(self):
        """
        The reference names its generator in `...chat.openai`, so reaching for the responses one by changing the class
        on the end of that path is the natural mistake. Uncorrected it reaches the YAML and fails deserialization.
        """
        answer = inspect_component.function(
            type_name="haystack.components.generators.chat.openai.OpenAIResponsesChatGenerator"
        )
        assert answer["import_path"] == (
            "haystack.components.generators.chat.openai_responses.OpenAIResponsesChatGenerator"
        )

    def test_class_that_is_not_installed(self):
        with pytest.raises(ImportError):
            inspect_component.function(type_name="haystack.components.nonsense.NoSuchComponent")

    def test_cannot_import_outside_the_allowlist(self, monkeypatch):
        """
        Recovering from a wrong path must not become a way to import anything: the search asks for shorter and
        shorter paths, and a module off the allowlist has to be refused before it is executed rather than after.
        """

        def refuse(name, *_args, **_kwargs):
            pytest.fail(f"searching for a class imported {name}")

        monkeypatch.setattr(importlib, "import_module", refuse)
        with pytest.raises(DeserializationError):
            inspect_component.function(type_name="subprocess.check_output.Popen")

    def test_tool_name(self):
        """The system prompt refers to the tool by this name."""
        assert inspect_component.name == "inspect_component"


class TestEditorEdits:
    def test_requires_unique_match_and_revision(self):
        editor = ConfigurationEditorToolset(reference_yaml=agent_yaml())
        original = editor._read_config()
        with pytest.raises(ValueError, match="exactly once"):
            editor._edit_config("", "x", original["revision"])
        changed = editor._edit_config("model: reference", "model: cheap", original["revision"])
        with pytest.raises(ValueError, match="Stale"):
            editor._edit_config("model: cheap", "model: other", original["revision"])
        assert changed["revision"] != original["revision"]
        assert editor._validate_config()["valid"]
        editor._submit_candidate(changed["revision"], "cheaper model")
        assert load_agent(editor.submitted.yaml).chat_generator.model == "cheap"

    def test_revalidates_on_every_edit(self):
        editor = ConfigurationEditorToolset(reference_yaml=agent_yaml())
        revision = editor._read_config()["revision"]
        changed = editor._edit_config("model: reference", "model: [broken", revision)
        assert editor._validate_config()["valid"] is False
        with pytest.raises(ValueError, match="Validate"):
            editor._submit_candidate(changed["revision"], "broken")
        repaired = editor._edit_config("model: [broken", "model: cheap", changed["revision"])
        assert editor._validate_config()["valid"]
        editor._edit_config("model: cheap", "model: other", repaired["revision"])
        with pytest.raises(ValueError, match="Validate"):
            editor._submit_candidate(repaired["revision"], "stale validation")

    def test_turn_ends_on_submit(self):
        editor = ConfigurationEditorToolset(reference_yaml=agent_yaml())
        changed = editor._edit_config("model: reference", "model: cheap", editor._read_config()["revision"])
        editor._validate_config()
        editor._submit_candidate(changed["revision"], "cheaper model")
        with pytest.raises(ValueError, match="has ended"):
            editor._edit_config("model: cheap", "model: other", changed["revision"])

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
        editor = ConfigurationEditorToolset(reference_yaml=agent_yaml(reference))
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
        current = editor._read_config()
        changed = editor._edit_config(
            current["yaml"], proposed.replace("ranker.documents", "ranker.missing"), current["revision"]
        )
        assert not editor._validate_config()["valid"]
        current = editor._read_config()
        changed = editor._edit_config(current["yaml"], proposed, changed["revision"])
        assert editor._validate_config()["valid"]
        editor._submit_candidate(changed["revision"], "retrieve more, rerank to three")
        tool = load_agent(editor.submitted.yaml).tools[-1]
        assert tool.invoke(query="Berlin")["documents"][0].id == document.id
        assert tool.outputs_to_state["documents"]["source"] == "documents"


class TestEditorEarlierCandidates:
    def test_starts_from_the_base(self):
        cheap = model_yaml("cheap")
        cheap_id = _configuration_id(cheap)
        editor = ConfigurationEditorToolset(reference_yaml=agent_yaml(), candidates={cheap_id: cheap}, base_id=cheap_id)
        assert editor._read_config() == {**editor._read_config(), "yaml": cheap, "parent_id": cheap_id}

    def test_unknown_base(self):
        with pytest.raises(ValueError, match="Unknown base"):
            ConfigurationEditorToolset(reference_yaml=agent_yaml(), base_id="never-measured")

    def test_restore_and_refuse_duplicates(self):
        cheap = model_yaml("cheap")
        cheap_id = _configuration_id(cheap)
        editor = ConfigurationEditorToolset(reference_yaml=agent_yaml(), candidates={cheap_id: cheap}, base_id=cheap_id)
        restored = editor._restore_candidate("reference", editor._read_config()["revision"])
        assert editor.parent_id == editor.reference_id
        assert "model: reference" in editor._read_config()["yaml"]
        # A formatting-only change is the same configuration
        text = editor._read_config()["yaml"]
        changed = editor._edit_config(text, "# comment\n" + text, restored["revision"])
        editor._validate_config()
        with pytest.raises(ValueError, match="duplicate_or_no_op"):
            editor._submit_candidate(changed["revision"], "same config")
        # So is a candidate submitted on an earlier turn
        restored = editor._restore_candidate(cheap_id, changed["revision"])
        editor._validate_config()
        with pytest.raises(ValueError, match="duplicate_or_no_op"):
            editor._submit_candidate(restored["revision"], "resubmitted")


class TestEditorTools:
    def test_tool_names(self):
        """The system prompt refers to the tools by these names."""
        assert [tool.name for tool in ConfigurationEditorToolset(reference_yaml=agent_yaml())] == [
            "read_config",
            "edit_config",
            "validate_config",
            "submit_candidate",
            "restore_candidate",
            "finish",
        ]

    def test_methods_are_bound(self):
        """The tools are the editor's own methods, so the optimizer never sees `self`."""
        for editing_tool in ConfigurationEditorToolset(reference_yaml=agent_yaml()):
            assert "self" not in editing_tool.parameters.get("properties", {})

    def test_every_parameter_is_described(self):
        """
        `create_tool_from_function` builds the schema from `Annotated` metadata and never reads `:param` lines, so a
        parameter documented only in the docstring reaches the model as a bare string with no explanation of it.
        """
        described = {
            f"{editing_tool.name}.{name}": specification.get("description")
            for editing_tool in ConfigurationEditorToolset(reference_yaml=agent_yaml())
            for name, specification in editing_tool.parameters.get("properties", {}).items()
        }
        assert described
        assert [parameter for parameter, description in described.items() if not description] == []
        # The constraint that actually fails at runtime has to be in the schema, not only in the error it raises.
        assert "exactly once" in described["edit_config.old"]
