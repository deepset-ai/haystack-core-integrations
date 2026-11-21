# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""Docling Graph Haystack extractor module."""

import importlib
import json
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any

import networkx as nx
from haystack import Document, component, logging
from haystack.components.converters.utils import normalize_metadata
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ByteStream
from haystack.utils import Secret, deserialize_secrets_inplace
from pydantic import BaseModel

from docling_graph import merge_graphs, run_pipeline

logger = logging.getLogger(__name__)


def _template_to_dotted_path(template: str | type[BaseModel]) -> str:
    """
    Return the dotted import path for a template.

    A string template is returned as-is. A Pydantic model class is converted to its
    `module.qualname` dotted path; a warning is logged if the class cannot be re-imported
    from that path (e.g. it was defined inside a function or an interactive session).
    """
    if isinstance(template, str):
        return template

    dotted_path = f"{template.__module__}.{template.__qualname__}"
    try:
        module = importlib.import_module(template.__module__)
        resolved = getattr(module, template.__qualname__, None)
    except ImportError:
        resolved = None
    if resolved is not template:
        logger.warning(
            "The template class '{dotted_path}' cannot be re-imported from its dotted path. "
            "Deserializing this component will fail unless the class is importable from that path.",
            dotted_path=dotted_path,
        )
    return dotted_path


def _bytestream_file_name(source: ByteStream) -> str:
    """
    Resolve a file name (with extension) for a Haystack `ByteStream`.

    Checks common metadata keys (`file_path`, `file_name`, `name`) and falls back to MIME-type
    extension guessing so that docling-graph can reliably detect the input format.
    """
    meta = source.meta or {}
    raw_name = meta.get("file_path") or meta.get("file_name") or meta.get("name")
    name = Path(raw_name).name if raw_name else "document"

    if not Path(name).suffix and source.mime_type:
        ext = mimetypes.guess_extension(source.mime_type)
        if ext:
            name = f"{name}{ext}"

    return name


def _graph_to_json(graph: "nx.DiGraph") -> str:
    """
    Serialize a NetworkX directed graph to a JSON string.

    Uses a plain node-link structure (`nodes` with `id`, `edges` with `source`/`target`) that is
    stable across NetworkX versions. Attribute values that are not JSON-serializable are
    converted with `str()`.
    """
    data = {
        "directed": True,
        "nodes": [{"id": node, **attrs} for node, attrs in graph.nodes(data=True)],
        "edges": [{"source": u, "target": v, **attrs} for u, v, attrs in graph.edges(data=True)],
    }
    return json.dumps(data, default=str)


@component
class DoclingGraphExtractor:
    """
    Extract knowledge graphs from documents using [docling-graph](https://github.com/docling-project/docling-graph).

    For each source document, docling-graph converts the document with Docling, extracts instances
    of a user-defined Pydantic template with an LLM or VLM backend, and builds a directed
    knowledge graph with explicit semantic relationships.

    Usage example:
    ```python
    from haystack_integrations.components.extractors.docling_graph import DoclingGraphExtractor

    extractor = DoclingGraphExtractor(
        template="my_templates.Invoice",
        inference="remote",
        provider="mistral",
        model="mistral-small-latest",
    )
    result = extractor.run(sources=["invoice.pdf"])
    graph = result["graphs"][0]  # networkx.DiGraph
    document = result["documents"][0]  # haystack Document with the JSON-serialized graph
    ```
    """

    def __init__(
        self,
        template: str | type[BaseModel],
        *,
        backend: str = "llm",
        inference: str = "local",
        processing_mode: str = "many-to-one",
        model: str | None = None,
        provider: str | None = None,
        docling_serve_url: str | None = None,
        docling_serve_api_key: Secret = Secret.from_env_var("DOCLING_SERVE_API_KEY", strict=False),
        merge_graphs: bool = False,
        config_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a DoclingGraphExtractor component.

        :param template: The extraction template: either a Pydantic model class or its dotted
            import path (e.g. `"my_package.templates.Invoice"`). Using a dotted path is
            recommended, as it survives pipeline serialization.
        :param backend: The extraction backend: `"llm"` (text-based) or `"vlm"` (vision-based).
        :param inference: Where model inference runs: `"local"` or `"remote"`. Remote inference
            reads provider API keys from environment variables (e.g. `OPENAI_API_KEY`,
            `MISTRAL_API_KEY`), following LiteLLM conventions.
        :param processing_mode: How document pages are processed: `"many-to-one"` builds one graph
            per document, `"one-to-one"` extracts one instance per page.
        :param model: Optional model name overriding the docling-graph default for the selected
            backend and inference mode.
        :param provider: Optional provider identifier (e.g. `"openai"`, `"mistral"`, `"ollama"`)
            overriding the docling-graph default.
        :param docling_serve_url: Optional base URL of a docling-serve instance to use for remote
            document conversion.
        :param docling_serve_api_key: API key for the docling-serve instance.
        :param merge_graphs: If `True`, the graphs extracted from all sources are merged into a
            single knowledge graph, and `run()` returns single-element lists.
        :param config_kwargs: Additional parameters passed through to the docling-graph
            `PipelineConfig` (e.g. `use_chunking`, `provenance`, `llm_input_format`). Values here
            take precedence over the equivalent init parameters, except for the per-run `source`.
        """
        self.template = template
        self.backend = backend
        self.inference = inference
        self.processing_mode = processing_mode
        self.model = model
        self.provider = provider
        self.docling_serve_url = docling_serve_url
        self.docling_serve_api_key = docling_serve_api_key
        self.merge_graphs = merge_graphs
        self.config_kwargs = config_kwargs if config_kwargs is not None else {}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        A template passed as a Pydantic model class is serialized as its dotted import path.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            template=_template_to_dotted_path(self.template),
            backend=self.backend,
            inference=self.inference,
            processing_mode=self.processing_mode,
            model=self.model,
            provider=self.provider,
            docling_serve_url=self.docling_serve_url,
            docling_serve_api_key=self.docling_serve_api_key.to_dict(),
            merge_graphs=self.merge_graphs,
            config_kwargs=self.config_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DoclingGraphExtractor":
        """
        Deserialize this component from a dictionary.

        :param data: Dictionary with keys `type` and `init_parameters`, as produced by `to_dict`.
        :returns: A new `DoclingGraphExtractor` instance.
        """
        deserialize_secrets_inplace(data.get("init_parameters", {}), keys=["docling_serve_api_key"])
        return default_from_dict(cls, data)

    def _build_config(self, source: str | Path) -> dict[str, Any]:
        """Build the docling-graph pipeline config for a single source."""
        config: dict[str, Any] = {
            "template": self.template,
            "backend": self.backend,
            "inference": self.inference,
            "processing_mode": self.processing_mode,
            # The component works fully in memory; exports are the caller's concern.
            "dump_to_disk": False,
        }
        if self.model is not None:
            config["model_override"] = self.model
        if self.provider is not None:
            config["provider_override"] = self.provider
        if self.docling_serve_url is not None:
            config["docling_serve_url"] = self.docling_serve_url
            api_key = self.docling_serve_api_key.resolve_value()
            if api_key is not None:
                config["docling_serve_api_key"] = api_key
        config.update(self.config_kwargs)
        config["source"] = str(source)
        return config

    def _extract_graph(self, source: str | Path, display_name: str) -> "nx.DiGraph":
        """Run the docling-graph pipeline on a single source and return its knowledge graph."""
        context = run_pipeline(self._build_config(source), mode="api")
        graph = context.knowledge_graph
        if graph is None:
            logger.warning("docling-graph produced no knowledge graph for source '{source}'.", source=display_name)
            graph = nx.DiGraph()
        return graph

    def _graph_document(self, graph: "nx.DiGraph", meta: dict[str, Any]) -> Document:
        """Build a Haystack `Document` carrying the JSON-serialized graph."""
        full_meta = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "template": _template_to_dotted_path(self.template),
            **meta,
        }
        return Document(content=_graph_to_json(graph), meta=full_meta)

    @component.output_types(graphs=list[nx.DiGraph], documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Extract a knowledge graph from each source document.

        :param sources: List of file paths, URLs, or ByteStream objects to process.
        :param meta:
            Optional metadata to attach to the output Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will
            be zipped.
            If a source is a ByteStream, its own metadata is also merged into the output.
        :returns:
            A dictionary with these keys:
            - `graphs`: The extracted knowledge graphs, one `networkx.DiGraph` per source
              (a single merged graph if `merge_graphs` is enabled).
            - `documents`: One Haystack `Document` per graph, with the JSON-serialized graph as
              content and graph statistics in the metadata.
        :raises ValueError: If `meta` is a list whose length does not match the number of sources.
        """
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))

        graphs: list[nx.DiGraph] = []
        metas: list[dict[str, Any]] = []
        for source, source_meta in zip(sources, meta_list, strict=True):
            if isinstance(source, ByteStream):
                # docling-graph only accepts paths and URLs, so ByteStreams are written to a
                # temporary file first. delete=False + manual cleanup because on Windows the
                # file cannot be re-opened by docling while it is held open here.
                file_name = _bytestream_file_name(source)
                suffix = Path(file_name).suffix
                temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                try:
                    temp_file.write(source.data)
                    temp_file.close()
                    graph = self._extract_graph(temp_file.name, display_name=file_name)
                finally:
                    temp_file.close()
                    os.unlink(temp_file.name)
                merged_meta = {"source": file_name, **(source.meta or {}), **source_meta}
            else:
                graph = self._extract_graph(source, display_name=str(source))
                merged_meta = {"source": str(source), **source_meta}
            graphs.append(graph)
            metas.append(merged_meta)

        if self.merge_graphs and len(graphs) > 1:
            merged_graph, merge_report = merge_graphs(graphs, template=self.template)
            merged_sources = [m.get("source") for m in metas]
            combined_meta: dict[str, Any] = {}
            for source_meta in metas:
                combined_meta.update(source_meta)
            combined_meta["source"] = merged_sources
            logger.debug("Merged {count} graphs into one: {report}", count=len(graphs), report=merge_report)
            graphs = [merged_graph]
            metas = [combined_meta]

        documents = [self._graph_document(graph, meta=doc_meta) for graph, doc_meta in zip(graphs, metas, strict=True)]
        return {"graphs": graphs, "documents": documents}
