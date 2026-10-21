# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
from haystack.dataclasses import ByteStream
from haystack.utils import Secret
from pydantic import BaseModel

from haystack_integrations.components.extractors.docling_graph import DoclingGraphExtractor

EXTRACTOR_MODULE = "haystack_integrations.components.extractors.docling_graph.extractor"


class Invoice(BaseModel):
    invoice_number: str = ""
    total: float = 0.0


def make_graph(num_nodes: int = 2) -> nx.DiGraph:
    graph = nx.DiGraph()
    for i in range(num_nodes):
        graph.add_node(f"node_{i}", label=f"Node {i}")
    for i in range(num_nodes - 1):
        graph.add_edge(f"node_{i}", f"node_{i + 1}", label="connects_to")
    return graph


class RunPipelineSpy:
    def __init__(self, graph_factory=make_graph):
        self.calls = []
        self.graph_factory = graph_factory

    def __call__(self, config, mode="api"):
        self.calls.append({"config": dict(config), "mode": mode, "source_exists": Path(config["source"]).exists()})
        return SimpleNamespace(knowledge_graph=self.graph_factory())


@pytest.fixture
def run_pipeline_spy(monkeypatch) -> RunPipelineSpy:
    spy = RunPipelineSpy()
    monkeypatch.setattr(f"{EXTRACTOR_MODULE}.run_pipeline", spy)
    return spy


class TestRun:
    def test_run_with_path_sources(self, run_pipeline_spy):
        extractor = DoclingGraphExtractor(template="templates.Invoice")
        result = extractor.run(sources=["sample.pdf", Path("other.pdf")])

        assert len(result["graphs"]) == 2
        assert all(isinstance(graph, nx.DiGraph) for graph in result["graphs"])
        assert len(result["documents"]) == 2

        document = result["documents"][0]
        content = json.loads(document.content)
        assert content["directed"] is True
        assert {node["id"] for node in content["nodes"]} == {"node_0", "node_1"}
        assert content["edges"] == [{"source": "node_0", "target": "node_1", "label": "connects_to"}]
        assert document.meta["num_nodes"] == 2
        assert document.meta["num_edges"] == 1
        assert document.meta["source"] == "sample.pdf"
        assert document.meta["template"] == "templates.Invoice"
        assert result["documents"][1].meta["source"] == "other.pdf"

        call = run_pipeline_spy.calls[0]
        assert call["mode"] == "api"
        assert call["config"]["source"] == "sample.pdf"
        assert call["config"]["template"] == "templates.Invoice"
        assert call["config"]["backend"] == "llm"
        assert call["config"]["inference"] == "local"
        assert call["config"]["processing_mode"] == "many-to-one"
        assert call["config"]["dump_to_disk"] is False
        assert "model_override" not in call["config"]
        assert "provider_override" not in call["config"]
        assert "docling_serve_url" not in call["config"]

    def test_run_with_bytestream_source(self, run_pipeline_spy):
        source = ByteStream(data=b"fake pdf bytes", meta={"file_path": "some/dir/sample.pdf"})
        extractor = DoclingGraphExtractor(template="templates.Invoice")
        result = extractor.run(sources=[source])

        call = run_pipeline_spy.calls[0]
        temp_path = Path(call["config"]["source"])
        assert temp_path.suffix == ".pdf"
        assert call["source_exists"] is True
        assert not temp_path.exists()  # temporary file is cleaned up

        document = result["documents"][0]
        assert document.meta["source"] == "sample.pdf"
        assert document.meta["file_path"] == "some/dir/sample.pdf"

    def test_run_with_bytestream_mime_type_fallback(self, run_pipeline_spy):
        source = ByteStream(data=b"fake bytes", mime_type="application/pdf")
        extractor = DoclingGraphExtractor(template="templates.Invoice")
        extractor.run(sources=[source])

        assert Path(run_pipeline_spy.calls[0]["config"]["source"]).suffix == ".pdf"

    def test_run_temp_file_cleaned_up_on_error(self, monkeypatch):
        captured = {}

        def failing_run_pipeline(config, mode="api"):
            captured["source"] = config["source"]
            captured["mode"] = mode
            msg = "extraction failed"
            raise RuntimeError(msg)

        monkeypatch.setattr(f"{EXTRACTOR_MODULE}.run_pipeline", failing_run_pipeline)
        extractor = DoclingGraphExtractor(template="templates.Invoice")
        with pytest.raises(RuntimeError, match="extraction failed"):
            extractor.run(sources=[ByteStream(data=b"fake bytes", mime_type="application/pdf")])
        assert not Path(captured["source"]).exists()

    @pytest.mark.usefixtures("run_pipeline_spy")
    def test_run_meta_single_dict(self):
        extractor = DoclingGraphExtractor(template="templates.Invoice")
        result = extractor.run(sources=["a.pdf", "b.pdf"], meta={"category": "invoices"})

        assert all(document.meta["category"] == "invoices" for document in result["documents"])

    @pytest.mark.usefixtures("run_pipeline_spy")
    def test_run_meta_list_of_dicts(self):
        extractor = DoclingGraphExtractor(template="templates.Invoice")
        result = extractor.run(sources=["a.pdf", "b.pdf"], meta=[{"idx": 1}, {"idx": 2}])

        assert result["documents"][0].meta["idx"] == 1
        assert result["documents"][1].meta["idx"] == 2

    @pytest.mark.usefixtures("run_pipeline_spy")
    def test_run_meta_list_length_mismatch_raises(self):
        extractor = DoclingGraphExtractor(template="templates.Invoice")
        with pytest.raises(ValueError, match="length of the metadata list"):
            extractor.run(sources=["a.pdf"], meta=[{"idx": 1}, {"idx": 2}])

    def test_run_without_knowledge_graph_returns_empty_graph(self, monkeypatch):
        monkeypatch.setattr(
            f"{EXTRACTOR_MODULE}.run_pipeline",
            lambda *_args, **_kwargs: SimpleNamespace(knowledge_graph=None),
        )
        extractor = DoclingGraphExtractor(template="templates.Invoice")
        result = extractor.run(sources=["a.pdf"])

        assert result["graphs"][0].number_of_nodes() == 0
        assert result["documents"][0].meta["num_nodes"] == 0

    def test_run_config_kwargs_passthrough_and_precedence(self, run_pipeline_spy):
        extractor = DoclingGraphExtractor(
            template="templates.Invoice",
            inference="local",
            config_kwargs={"use_chunking": False, "inference": "remote", "source": "ignored.pdf"},
        )
        extractor.run(sources=["a.pdf"])

        config = run_pipeline_spy.calls[0]["config"]
        assert config["use_chunking"] is False
        assert config["inference"] == "remote"  # config_kwargs win over init parameters
        assert config["source"] == "a.pdf"  # the per-run source cannot be overridden

    def test_run_model_provider_and_docling_serve_options(self, run_pipeline_spy, monkeypatch):
        monkeypatch.setenv("DOCLING_SERVE_API_KEY", "test-api-key")
        extractor = DoclingGraphExtractor(
            template="templates.Invoice",
            model="gpt-5.2",
            provider="openai",
            docling_serve_url="http://localhost:5001",
        )
        extractor.run(sources=["a.pdf"])

        config = run_pipeline_spy.calls[0]["config"]
        assert config["model_override"] == "gpt-5.2"
        assert config["provider_override"] == "openai"
        assert config["docling_serve_url"] == "http://localhost:5001"
        assert config["docling_serve_api_key"] == "test-api-key"

    @pytest.mark.usefixtures("run_pipeline_spy")
    def test_run_merge_graphs(self, monkeypatch):
        merged_graph = make_graph(num_nodes=3)
        merge_calls = []

        def fake_merge_graphs(inputs, template=None, policy=None):
            merge_calls.append({"inputs": list(inputs), "template": template, "policy": policy})
            return merged_graph, SimpleNamespace(merged_nodes=1)

        monkeypatch.setattr(f"{EXTRACTOR_MODULE}.merge_graphs", fake_merge_graphs)
        extractor = DoclingGraphExtractor(template="templates.Invoice", merge_graphs=True)
        result = extractor.run(sources=["a.pdf", "b.pdf"])

        assert len(merge_calls) == 1
        assert len(merge_calls[0]["inputs"]) == 2
        assert merge_calls[0]["template"] == "templates.Invoice"
        assert result["graphs"] == [merged_graph]
        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["source"] == ["a.pdf", "b.pdf"]
        assert result["documents"][0].meta["num_nodes"] == 3

    @pytest.mark.usefixtures("run_pipeline_spy")
    def test_run_merge_graphs_single_source_skips_merge(self, monkeypatch):
        def fail_merge(*_args, **_kwargs):
            msg = "merge_graphs should not be called for a single source"
            raise AssertionError(msg)

        monkeypatch.setattr(f"{EXTRACTOR_MODULE}.merge_graphs", fail_merge)
        extractor = DoclingGraphExtractor(template="templates.Invoice", merge_graphs=True)
        result = extractor.run(sources=["a.pdf"])

        assert len(result["graphs"]) == 1


class TestSerialization:
    def test_to_dict_defaults(self):
        extractor = DoclingGraphExtractor(template="templates.Invoice")
        data = extractor.to_dict()

        assert data == {
            "type": f"{EXTRACTOR_MODULE}.DoclingGraphExtractor",
            "init_parameters": {
                "template": "templates.Invoice",
                "backend": "llm",
                "inference": "local",
                "processing_mode": "many-to-one",
                "model": None,
                "provider": None,
                "docling_serve_url": None,
                "docling_serve_api_key": {
                    "type": "env_var",
                    "env_vars": ["DOCLING_SERVE_API_KEY"],
                    "strict": False,
                },
                "merge_graphs": False,
                "config_kwargs": {},
            },
        }

    def test_to_dict_template_class_uses_dotted_path(self):
        extractor = DoclingGraphExtractor(template=Invoice)
        data = extractor.to_dict()

        assert data["init_parameters"]["template"] == f"{Invoice.__module__}.Invoice"

    def test_from_dict(self):
        data = {
            "type": f"{EXTRACTOR_MODULE}.DoclingGraphExtractor",
            "init_parameters": {
                "template": "templates.Invoice",
                "backend": "vlm",
                "inference": "remote",
                "processing_mode": "one-to-one",
                "model": "mistral-small-latest",
                "provider": "mistral",
                "docling_serve_url": "http://localhost:5001",
                "docling_serve_api_key": {
                    "type": "env_var",
                    "env_vars": ["DOCLING_SERVE_API_KEY"],
                    "strict": False,
                },
                "merge_graphs": True,
                "config_kwargs": {"use_chunking": False},
            },
        }
        extractor = DoclingGraphExtractor.from_dict(data)

        assert extractor.template == "templates.Invoice"
        assert extractor.backend == "vlm"
        assert extractor.inference == "remote"
        assert extractor.processing_mode == "one-to-one"
        assert extractor.model == "mistral-small-latest"
        assert extractor.provider == "mistral"
        assert extractor.docling_serve_url == "http://localhost:5001"
        assert extractor.docling_serve_api_key == Secret.from_env_var("DOCLING_SERVE_API_KEY", strict=False)
        assert extractor.merge_graphs is True
        assert extractor.config_kwargs == {"use_chunking": False}

    def test_to_dict_from_dict_roundtrip(self):
        extractor = DoclingGraphExtractor(
            template="templates.Invoice",
            inference="remote",
            provider="mistral",
            merge_graphs=True,
            config_kwargs={"provenance": "detailed"},
        )
        restored = DoclingGraphExtractor.from_dict(extractor.to_dict())

        assert restored.to_dict() == extractor.to_dict()


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY is not set")
def test_extraction_end_to_end(tmp_path):
    source = tmp_path / "invoice.md"
    source.write_text("# Invoice\n\nInvoice number: INV-42\n\nTotal: 123.45 EUR\n")

    extractor = DoclingGraphExtractor(
        template=Invoice,
        inference="remote",
        provider="mistral",
        model="mistral-small-latest",
    )
    result = extractor.run(sources=[source])

    assert len(result["graphs"]) == 1
    graph = result["graphs"][0]
    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() >= 1
    document = result["documents"][0]
    assert document.meta["num_nodes"] == graph.number_of_nodes()
    json.loads(document.content)
