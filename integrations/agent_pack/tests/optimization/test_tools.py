import importlib
import json

import pytest
from haystack import Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.rankers import LLMRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.core.errors import DeserializationError
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent
from haystack_integrations.agent_pack.evaluation import RetrievalHarnessEvaluator
from haystack_integrations.agent_pack.optimization.dataclasses import ConfigurationDraft, KnownConfigurations
from haystack_integrations.agent_pack.optimization.tools import (
    ValidateConfig,
    _documentation_result,
    _make_haystack_documentation_toolset,
    edit_config,
    finish,
    inspect_component,
    read_config,
    restore_candidate,
    submit_candidate,
)
from haystack_integrations.agent_pack.optimization.utils import _configuration_id, load_agent, load_pipeline


def agent_yaml(agent=None):
    pipeline = Pipeline()
    pipeline.add_component("agent", agent or Agent(chat_generator=MockChatGenerator(model="reference")))
    return pipeline.dumps()


def model_yaml(model):
    return agent_yaml(Agent(chat_generator=MockChatGenerator(model=model)))


class AcceptEvaluator:
    """A harness evaluator whose check accepts every configuration."""

    def validate(self, target):
        pass

    def to_dict(self):
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data):
        return default_from_dict(cls, data)


class OnlyCheapEvaluator(AcceptEvaluator):
    """A harness evaluator whose check rejects every model but `cheap`."""

    def validate(self, target):
        if target.chat_generator.model != "cheap":
            msg = "Only the cheap model is allowed."
            raise ValueError(msg)


def turn_start(yaml=None, candidates=None, base_id=None):
    """The draft and known configurations a turn starts from, as `propose_candidate` builds them."""
    yaml = yaml or agent_yaml()
    reference_id = _configuration_id(yaml)
    known = KnownConfigurations(reference_id=reference_id, yaml_by_id={reference_id: yaml, **(candidates or {})})
    parent_id = base_id or reference_id
    return ConfigurationDraft(yaml=known.yaml_by_id[parent_id], parent_id=parent_id), known


def edited(draft, old, new):
    return edit_config.function(old=old, new=new, expected_revision=draft.revision, draft=draft)["draft"]


def validated(draft, evaluator=None):
    return ValidateConfig(evaluator=evaluator or AcceptEvaluator()).function(draft=draft)["draft"]


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


class TestReadConfig:
    def test_returns_the_draft(self):
        draft, _ = turn_start()
        assert read_config.function(draft=draft) == {
            "yaml": draft.yaml,
            "revision": draft.revision,
            "parent_id": draft.parent_id,
        }


class TestEditConfig:
    def test_requires_unique_match_and_current_revision(self):
        draft, _ = turn_start()
        with pytest.raises(ValueError, match="exactly once"):
            edit_config.function(old="", new="x", expected_revision=draft.revision, draft=draft)
        changed = edited(draft, "model: reference", "model: cheap")
        assert "model: cheap" in changed.yaml
        with pytest.raises(ValueError, match="Stale"):
            edit_config.function(
                old="model: cheap", new="model: other", expected_revision=draft.revision, draft=changed
            )

    def test_clears_the_validation(self):
        draft = validated(turn_start()[0])
        assert draft.validated_revision == draft.revision
        assert edited(draft, "model: reference", "model: cheap").validated_revision is None


class TestValidateConfig:
    def test_records_the_valid_revision(self):
        draft, _ = turn_start()
        result = ValidateConfig(evaluator=AcceptEvaluator()).function(draft=draft)
        assert result["result"] == {"valid": True, "revision": draft.revision, "tools": []}
        assert result["draft"].validated_revision == draft.revision
        assert result["validation_failures"] == []

    def test_records_a_failure(self):
        draft = edited(turn_start()[0], "model: reference", "model: [broken")
        result = ValidateConfig(evaluator=AcceptEvaluator()).function(draft=draft)
        assert result["result"]["valid"] is False
        assert result["draft"].validated_revision is None
        assert result["validation_failures"] == [{"revision": draft.revision, "error": result["result"]["error"]}]

    def test_runs_the_evaluators_check(self):
        validate = ValidateConfig(evaluator=OnlyCheapEvaluator())
        draft, _ = turn_start()
        assert validate.function(draft=draft)["result"]["error"] == "ValueError: Only the cheap model is allowed."
        cheap = edited(draft, "model: reference", "model: cheap")
        assert validate.function(draft=cheap)["result"]["valid"] is True

    def test_requires_a_serializable_evaluator(self):
        class Unserializable:
            def validate(self, target):
                pass

        with pytest.raises(TypeError, match="to_dict, from_dict"):
            ValidateConfig(evaluator=Unserializable())

    def test_serialization_roundtrip(self):
        restored = ValidateConfig.from_dict(
            ValidateConfig(evaluator=RetrievalHarnessEvaluator(k=3), loader=load_pipeline).to_dict()
        )
        assert isinstance(restored.evaluator, RetrievalHarnessEvaluator)
        assert restored.evaluator.k == 3
        assert restored.loader is load_pipeline

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
        draft, _ = turn_start(yaml=agent_yaml(reference))
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
        broken = validated(edited(draft, draft.yaml, proposed.replace("ranker.documents", "ranker.missing")))
        assert broken.validated_revision is None
        repaired = validated(edited(broken, broken.yaml, proposed))
        assert repaired.validated_revision == repaired.revision
        tool = load_agent(repaired.yaml).tools[-1]
        assert tool.invoke(query="Berlin")["documents"][0].id == document.id
        assert tool.outputs_to_state["documents"]["source"] == "documents"


class TestSubmitCandidate:
    def test_requires_the_current_revision_to_be_validated(self):
        draft, known = turn_start()
        changed = edited(draft, "model: reference", "model: cheap")
        with pytest.raises(ValueError, match="Validate"):
            submit_candidate.function(expected_revision=changed.revision, rationale="r", draft=changed, known=known)

    def test_submits_with_its_parent_and_diff(self):
        draft, known = turn_start()
        changed = validated(edited(draft, "model: reference", "model: cheap"))
        submitted = submit_candidate.function(
            expected_revision=changed.revision, rationale="cheaper model", draft=changed, known=known
        )["submitted"]
        assert submitted.parent_id == known.reference_id
        assert submitted.rationale == "cheaper model"
        assert "+          model: cheap" in submitted.diff
        assert load_agent(submitted.yaml).chat_generator.model == "cheap"

    def test_refuses_known_configurations(self):
        cheap = model_yaml("cheap")
        draft, known = turn_start(candidates={_configuration_id(cheap): cheap})
        # A formatting-only change is the same configuration as the reference
        commented = validated(edited(draft, draft.yaml, "# comment\n" + draft.yaml))
        with pytest.raises(ValueError, match="duplicate_or_no_op"):
            submit_candidate.function(expected_revision=commented.revision, rationale="r", draft=commented, known=known)
        # So is a candidate submitted on an earlier turn
        again = validated(edited(draft, "model: reference", "model: cheap"))
        with pytest.raises(ValueError, match="duplicate_or_no_op"):
            submit_candidate.function(expected_revision=again.revision, rationale="r", draft=again, known=known)


class TestRestoreCandidate:
    def test_restores_the_reference_and_earlier_candidates(self):
        cheap = model_yaml("cheap")
        cheap_id = _configuration_id(cheap)
        draft, known = turn_start(candidates={cheap_id: cheap}, base_id=cheap_id)
        restored = restore_candidate.function(
            candidate_id="reference", expected_revision=draft.revision, draft=draft, known=known
        )["draft"]
        assert (restored.yaml, restored.parent_id) == (known.yaml_by_id[known.reference_id], known.reference_id)
        back = restore_candidate.function(
            candidate_id=cheap_id, expected_revision=restored.revision, draft=restored, known=known
        )["draft"]
        assert (back.yaml, back.parent_id) == (cheap, cheap_id)

    def test_unknown_candidate(self):
        draft, known = turn_start()
        with pytest.raises(ValueError, match="Unknown candidate"):
            restore_candidate.function(candidate_id="nope", expected_revision=draft.revision, draft=draft, known=known)


class TestFinish:
    def test_records_the_reason(self):
        assert finish.function(reason="nothing left")["finish_reason"] == "nothing left"


class TestEditingTools:
    TOOLS = (
        read_config,
        edit_config,
        ValidateConfig(evaluator=AcceptEvaluator()),
        submit_candidate,
        restore_candidate,
        finish,
    )

    def test_state_is_never_shown(self):
        """The draft and known configurations come from the agent's state, so the optimizer never passes them."""
        for editing_tool in self.TOOLS:
            assert not {"draft", "known"} & set(editing_tool.parameters.get("properties", {}))

    def test_every_parameter_is_described(self):
        """
        `create_tool_from_function` builds the schema from `Annotated` metadata and never reads `:param` lines, so a
        parameter documented only in the docstring reaches the model as a bare string with no explanation of it.
        """
        described = {
            f"{editing_tool.name}.{name}": specification.get("description")
            for editing_tool in self.TOOLS
            for name, specification in editing_tool.parameters.get("properties", {}).items()
        }
        assert described
        assert [parameter for parameter, description in described.items() if not description] == []
        # The constraint that actually fails at runtime has to be in the schema, not only in the error it raises.
        assert "exactly once" in described["edit_config.old"]
