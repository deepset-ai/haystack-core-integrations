import pytest
from haystack import Document, Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.rankers import LLMRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.agent_pack.advanced_rag import create_advanced_rag_agent
from haystack_integrations.agent_pack.optimization.utils import dump_agent, load_agent
from haystack_integrations.agent_pack.optimization.workspace import ConfigurationWorkspace


def agent_yaml(agent=None):
    pipeline = Pipeline()
    pipeline.add_component("agent", agent or Agent(chat_generator=MockChatGenerator(model="reference")))
    return pipeline.dumps()


class TestDraft:
    def test_existing_draft_is_preserved(self, tmp_path):
        path = tmp_path / "provided.yaml"
        draft = dump_agent(Agent(chat_generator=MockChatGenerator(model="draft")))
        path.write_text(draft)
        workspace = ConfigurationWorkspace(path, agent_yaml())
        assert workspace._read_config()["yaml"] == draft
        assert workspace._validate_config()["valid"]


class TestEdit:
    def test_requires_unique_match_and_revision(self, tmp_path):
        workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())
        original = workspace._read_config()
        with pytest.raises(ValueError, match="exactly once"):
            workspace._edit_config("", "x", original["revision"])
        changed = workspace._edit_config("model: reference", "model: cheap", original["revision"])
        with pytest.raises(ValueError, match="Stale"):
            workspace._edit_config("model: cheap", "model: other", original["revision"])
        assert changed["revision"] != original["revision"]
        assert workspace._validate_config()["valid"]
        workspace._submit_candidate(changed["revision"], "cheaper model")
        assert load_agent(workspace.submitted.yaml).chat_generator.model == "cheap"

    def test_revalidates_on_every_edit(self, tmp_path):
        workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())
        revision = workspace._read_config()["revision"]
        changed = workspace._edit_config("model: reference", "model: [broken", revision)
        assert workspace._validate_config()["valid"] is False
        with pytest.raises(ValueError, match="Validate"):
            workspace._submit_candidate(changed["revision"], "broken")
        repaired = workspace._edit_config("model: [broken", "model: cheap", changed["revision"])
        assert workspace._validate_config()["valid"]
        workspace._edit_config("model: cheap", "model: other", repaired["revision"])
        with pytest.raises(ValueError, match="Validate"):
            workspace._submit_candidate(repaired["revision"], "stale validation")

    def test_rejects_a_symlinked_path(self, tmp_path):
        outside = tmp_path / "outside.yaml"
        outside.write_text("untouched")
        linked = tmp_path / "candidate.yaml"
        linked.symlink_to(outside)
        with pytest.raises(ValueError, match="symlink"):
            ConfigurationWorkspace(linked, agent_yaml())
        assert outside.read_text() == "untouched"

    def test_replacing_a_retriever_and_repairing_it(self, tmp_path):
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
        current = workspace._read_config()
        changed = workspace._edit_config(
            current["yaml"], proposed.replace("ranker.documents", "ranker.missing"), current["revision"]
        )
        assert not workspace._validate_config()["valid"]
        current = workspace._read_config()
        changed = workspace._edit_config(current["yaml"], proposed, changed["revision"])
        assert workspace._validate_config()["valid"]
        workspace._submit_candidate(changed["revision"], "retrieve more, rerank to three")
        tool = load_agent(workspace.submitted.yaml).tools[-1]
        assert tool.invoke(query="Berlin")["documents"][0].id == document.id
        assert tool.outputs_to_state["documents"]["source"] == "documents"


class TestLoad:
    def test_restore_tracks_ancestry(self, tmp_path):
        workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())
        revision = workspace._read_config()["revision"]
        changed = workspace._edit_config("model: reference", "model: cheap", revision)
        workspace._validate_config()
        first = workspace._submit_candidate(changed["revision"], "one")
        snapshot = workspace.submitted
        workspace.begin_turn()
        assert workspace.parent_id == first["candidate_id"]
        current = workspace._restore_candidate("reference", workspace._read_config()["revision"])
        text = workspace._read_config()["yaml"]
        current = workspace._edit_config(text, "# comment\n" + text, current["revision"])
        workspace._validate_config()
        with pytest.raises(ValueError, match="duplicate_or_no_op"):
            workspace._submit_candidate(current["revision"], "same config")
        assert load_agent(snapshot.yaml).chat_generator.model == "cheap"

    def test_rebase_on_the_best_candidate(self, tmp_path):
        """A search that always edits its last attempt carries a regression into everything after it."""
        workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())
        current = workspace._read_config()
        good = workspace._edit_config("model: reference", "model: good", current["revision"])
        workspace._validate_config()
        best = workspace._submit_candidate(good["revision"], "the one that scored well")["candidate_id"]
        workspace.begin_turn()
        current = workspace._read_config()
        worse = workspace._edit_config("model: good", "model: worse", current["revision"])
        workspace._validate_config()
        workspace._submit_candidate(worse["revision"], "a regression")
        # Without a base the next turn would continue from the regression.
        workspace.begin_turn()
        assert "model: worse" in workspace._read_config()["yaml"]
        # Naming the best candidate rebases the file and the ancestry onto it.
        workspace.begin_turn(base_id=best)
        assert "model: good" in workspace._read_config()["yaml"]
        assert workspace._read_config()["parent_id"] == best
        # Every earlier snapshot is still reachable.
        workspace._restore_candidate("reference", workspace._read_config()["revision"])
        assert "model: reference" in workspace._read_config()["yaml"]

    def test_rebase_on_an_unknown_candidate(self, tmp_path):
        workspace = ConfigurationWorkspace(tmp_path / "candidate.yaml", agent_yaml())
        workspace.begin_turn(base_id="never-measured")
        assert workspace._read_config()["parent_id"] == workspace.reference_id
