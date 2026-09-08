import importlib

import pytest
import yaml
from haystack import Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.rankers import LLMRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.core.errors import DeserializationError
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import ComponentTool

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent
from haystack_integrations.agent_pack.optimization.workspace import (
    ConfigurationWorkspace,
    configuration_id,
    dump_agent,
    inspect_component,
    load_agent,
)


def agent_yaml(agent=None):
    pipeline = Pipeline()
    pipeline.add_component("agent", agent or Agent(chat_generator=MockChatGenerator(model="reference")))
    return pipeline.dumps()


def test_generated_yaml_has_editable_multiline_prompts_and_preserves_values():
    prompt = "First line — literal Unicode.\nSecond line: keep this exactly.\n"
    original = Agent(chat_generator=MockChatGenerator(), system_prompt=prompt)
    serialized = dump_agent(original)
    assert "system_prompt: |" in serialized
    assert "First line — literal Unicode." in serialized
    assert load_agent(serialized).system_prompt == prompt


def test_existing_draft_is_preserved(tmp_path):
    path = tmp_path / "provided.yaml"
    draft = dump_agent(Agent(chat_generator=MockChatGenerator(model="draft")))
    path.write_text(draft)
    workspace = ConfigurationWorkspace(path, agent_yaml())
    assert workspace.read_config()["yaml"] == draft
    assert workspace.validate_config()["valid"]


def test_edit_requires_unique_match_and_current_revision(tmp_path):
    workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())
    original = workspace.read_config()
    with pytest.raises(ValueError, match="exactly once"):
        workspace.edit_config("", "x", original["revision"])
    changed = workspace.edit_config("model: reference", "model: cheap", original["revision"])
    with pytest.raises(ValueError, match="Stale"):
        workspace.edit_config("model: cheap", "model: other", original["revision"])
    assert changed["revision"] != original["revision"]
    assert workspace.validate_config()["valid"]
    workspace.submit_candidate(changed["revision"], "cheaper model")
    assert load_agent(workspace.submitted.yaml).chat_generator.model == "cheap"


def test_validation_repairs_and_invalidates_on_every_edit(tmp_path):
    workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())
    revision = workspace.read_config()["revision"]
    changed = workspace.edit_config("model: reference", "model: [broken", revision)
    assert workspace.validate_config()["valid"] is False
    with pytest.raises(ValueError, match="Validate"):
        workspace.submit_candidate(changed["revision"], "broken")
    repaired = workspace.edit_config("model: [broken", "model: cheap", changed["revision"])
    assert workspace.validate_config()["valid"]
    workspace.edit_config("model: cheap", "model: other", repaired["revision"])
    with pytest.raises(ValueError, match="Validate"):
        workspace.submit_candidate(repaired["revision"], "stale validation")


def test_duplicate_yaml_keys_and_untrusted_classes_rejected():
    with pytest.raises(yaml.constructor.ConstructorError, match="Duplicate YAML key"):
        load_agent("components: {}\ncomponents: {}\n")
    with pytest.raises(DeserializationError):
        load_agent("components:\n  agent:\n    type: subprocess.Popen\n    init_parameters: {}\nconnections: []")


def test_restore_tracks_ancestry_and_formatting_does_not_create_candidate(tmp_path):
    workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())
    revision = workspace.read_config()["revision"]
    changed = workspace.edit_config("model: reference", "model: cheap", revision)
    workspace.validate_config()
    first = workspace.submit_candidate(changed["revision"], "one")
    snapshot = workspace.submitted
    workspace.begin_turn()
    assert workspace.parent_id == first["candidate_id"]
    current = workspace.restore_candidate("reference", workspace.read_config()["revision"])
    text = workspace.read_config()["yaml"]
    current = workspace.edit_config(text, "# comment\n" + text, current["revision"])
    workspace.validate_config()
    with pytest.raises(ValueError, match="duplicate_or_no_op"):
        workspace.submit_candidate(current["revision"], "same config")
    assert load_agent(snapshot.yaml).chat_generator.model == "cheap"


def test_store_identity_is_not_discarded():
    first = InMemoryDocumentStore(index="first")
    second = InMemoryDocumentStore(index="second")

    def configured(store):
        return agent_yaml(
            Agent(
                chat_generator=MockChatGenerator(),
                tools=[ComponentTool(component=InMemoryBM25Retriever(document_store=store))],
            )
        )

    assert configuration_id(configured(first)) != configuration_id(configured(second))


def test_only_bound_file_can_be_edited(tmp_path):
    outside = tmp_path / "outside.yaml"
    outside.write_text("untouched")
    workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())
    assert all("path" not in tool.parameters.get("properties", {}) for tool in workspace.tools())
    workspace.path.unlink()
    workspace.path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        workspace.read_config()
    assert outside.read_text() == "untouched"


def test_replace_retriever_with_bm25_ranker_pipeline_and_repair_connection(tmp_path):
    store = InMemoryDocumentStore()
    document = Document(content="Berlin is in Germany")
    store.write_documents([document])
    reference = create_advanced_rag_agent(
        document_store=store,
        retriever=InMemoryBM25Retriever(document_store=store),
        llm=MockChatGenerator("ok"),
        backup_answer_llm=MockChatGenerator("ok"),
    )
    workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml(reference))
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
    current = workspace.read_config()
    changed = workspace.edit_config(
        current["yaml"], proposed.replace("ranker.documents", "ranker.missing"), current["revision"]
    )
    assert not workspace.validate_config()["valid"]
    current = workspace.read_config()
    changed = workspace.edit_config(current["yaml"], proposed, changed["revision"])
    assert workspace.validate_config()["valid"]
    workspace.submit_candidate(changed["revision"], "retrieve more, rerank to three")
    tool = load_agent(workspace.submitted.yaml).tools[-1]
    assert tool.invoke(query="Berlin")["documents"][0].id == document.id
    assert tool.outputs_to_state["documents"]["source"] == "documents"


def test_a_turn_can_rebase_on_the_best_candidate_rather_than_the_last(tmp_path):
    """A search that always edits its last attempt carries a regression into everything after it."""
    workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())

    current = workspace.read_config()
    good = workspace.edit_config("model: reference", "model: good", current["revision"])
    workspace.validate_config()
    best = workspace.submit_candidate(good["revision"], "the one that scored well")["candidate_id"]

    workspace.begin_turn()
    current = workspace.read_config()
    worse = workspace.edit_config("model: good", "model: worse", current["revision"])
    workspace.validate_config()
    workspace.submit_candidate(worse["revision"], "a regression")

    # Without a base the next turn would continue from the regression.
    workspace.begin_turn()
    assert "model: worse" in workspace.read_config()["yaml"]

    # Naming the best candidate rebases the file and the ancestry onto it.
    workspace.begin_turn(base_id=best)
    assert "model: good" in workspace.read_config()["yaml"]
    assert workspace.read_config()["parent_id"] == best

    # Every earlier snapshot is still reachable.
    workspace.restore_candidate("reference", workspace.read_config()["revision"])
    assert "model: reference" in workspace.read_config()["yaml"]


def test_rebasing_on_an_unknown_candidate_falls_back_rather_than_failing(tmp_path):
    workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())

    workspace.begin_turn(base_id="never-measured")

    assert workspace.read_config()["parent_id"] == workspace.reference_id


def test_inspecting_a_component_reports_the_path_it_is_installed_at():
    """The answer carries the string the YAML has to use, not the one that happened to be asked for."""
    answer = inspect_component("haystack.components.rankers.llm_ranker.LLMRanker")

    assert answer["import_path"] == "haystack.components.rankers.llm_ranker.LLMRanker"
    assert "top_k" in answer["constructor"]


def test_a_class_asked_for_beside_its_neighbour_is_found_where_it_really_lives():
    """
    The reference names its generator in `...chat.openai`, so reaching for the responses one by changing the class
    on the end of that path is the natural mistake. Uncorrected it reaches the YAML and fails deserialization.
    """
    answer = inspect_component("haystack.components.generators.chat.openai.OpenAIResponsesChatGenerator")

    assert answer["import_path"] == (
        "haystack.components.generators.chat.openai_responses.OpenAIResponsesChatGenerator"
    )


def test_a_class_that_is_not_installed_still_fails():
    with pytest.raises(ImportError):
        inspect_component("haystack.components.nonsense.NoSuchComponent")


def test_searching_for_a_class_cannot_import_outside_the_deserialization_allowlist(monkeypatch):
    """
    Recovering from a wrong path must not become a way to import anything: the search asks for shorter and
    shorter paths, and a module off the allowlist has to be refused before it is executed rather than after.
    """

    def refuse(name, *_args, **_kwargs):
        pytest.fail(f"searching for a class imported {name}")

    monkeypatch.setattr(importlib, "import_module", refuse)

    with pytest.raises(DeserializationError):
        inspect_component("subprocess.check_output.Popen")


def test_every_tool_parameter_describes_itself_to_the_model(tmp_path):
    """
    `create_tool_from_function` builds the schema from `Annotated` metadata and never reads `:param` lines, so a
    parameter documented only in the docstring reaches the model as a bare string with no explanation of it.
    """
    workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())

    described = {
        f"{tool.name}.{name}": specification.get("description")
        for tool in workspace.tools()
        for name, specification in tool.parameters.get("properties", {}).items()
    }

    assert described
    assert [parameter for parameter, description in described.items() if not description] == []
    # The constraint that actually fails at runtime has to be in the schema, not only in the error it raises.
    assert "exactly once" in described["edit_config.old"]
