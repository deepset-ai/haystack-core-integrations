# docling-graph-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/docling-graph-haystack.svg)](https://pypi.org/project/docling-graph-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling-graph-haystack.svg)](https://pypi.org/project/docling-graph-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/docling-graph)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/docling_graph/CHANGELOG.md)

---

[docling-graph](https://github.com/docling-project/docling-graph) transforms unstructured documents into validated
knowledge graphs: you define the target schema as a Pydantic model, docling-graph converts the document with
[Docling](https://github.com/docling-project/docling), extracts instances with an LLM or VLM backend, and builds a
[NetworkX](https://networkx.org) directed graph with explicit semantic relationships.

This integration provides the `DoclingGraphExtractor` component, which extracts one knowledge graph per source
document and returns both the raw `networkx.DiGraph` objects and Haystack `Document` objects carrying the
JSON-serialized graphs.

## Installation

```console
pip install docling-graph-haystack
```

## Usage

Define an extraction template as a Pydantic model (see the
[docling-graph documentation](https://docling-project.github.io/docling-graph/) for details on templates and the
`edge()` relationship utility), make it importable, and pass its dotted path to the component:

```python
from haystack_integrations.components.extractors.docling_graph import DoclingGraphExtractor

extractor = DoclingGraphExtractor(
    template="my_templates.Invoice",  # dotted path to your Pydantic template
    inference="remote",
    provider="mistral",
    model="mistral-small-latest",
)
result = extractor.run(sources=["invoice.pdf"])

graph = result["graphs"][0]  # networkx.DiGraph
document = result["documents"][0]  # Haystack Document with the JSON-serialized graph
```

`sources` accepts file paths, URLs, and Haystack `ByteStream` objects. With `merge_graphs=True`, the graphs of all
sources are merged into a single knowledge graph.

### Model configuration

- `inference="local"` (default) runs extraction with a local model; `inference="remote"` uses a hosted provider.
- Provider API keys are read from environment variables following [LiteLLM](https://docs.litellm.ai) conventions,
  e.g. `OPENAI_API_KEY`, `MISTRAL_API_KEY`.
- `backend="vlm"` uses a vision language model instead of the text-based LLM pipeline
  (requires `pip install "docling-graph[vlm]"` for local VLM inference).
- Additional [docling-graph pipeline options](https://docling-project.github.io/docling-graph/) (chunking,
  provenance, gleaning, ...) can be passed via `config_kwargs`.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

Integration tests require a Mistral API key in the `MISTRAL_API_KEY` environment variable; they are skipped
otherwise. Unit tests run without credentials.
