# fastembed-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/fastembed-haystack.svg)](https://pypi.org/project/fastembed-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fastembed-haystack.svg)](https://pypi.org/project/fastembed-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/fastembed)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/fastembed/CHANGELOG.md)

## Components

- **FastembedTextEmbedder / FastembedDocumentEmbedder / Sparse embedders** – embedding components for dense and sparse retrieval (see docs).
- **FastembedRanker** – cross-encoder reranker (see docs).
- **FastembedColbertReranker** – *new* ColBERT late-interaction (MaxSim) reranker for reordering 100–500 candidates on CPU via ONNX.  
  → See the docs for installation and examples.

### Documentation

- Integration page: https://haystack.deepset.ai/integrations/fastembed
- Component docs:
  - Fastembed embedders: https://docs.haystack.deepset.ai/docs/fastembedtextembedder
  - FastembedRanker (cross-encoder): https://docs.haystack.deepset.ai/docs/fastembedranker
  - **FastembedColbertReranker (ColBERT)**: *[link to your new docs section]*

### Install

```bash
pip install fastembed-haystack

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).