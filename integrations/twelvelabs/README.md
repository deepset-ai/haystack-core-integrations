# twelvelabs-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/twelvelabs-haystack.svg)](https://pypi.org/project/twelvelabs-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/twelvelabs-haystack.svg)](https://pypi.org/project/twelvelabs-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/twelvelabs)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/twelvelabs/CHANGELOG.md)

---

## Components

TwelveLabs **Marengo** embeds text, images, audio, and video into a single shared vector space, so
embeddings from any modality are directly comparable — enabling cross-modal retrieval.

- `TwelveLabsTextEmbedder` — embeds a text query.
- `TwelveLabsDocumentEmbedder` — embeds the text `content` of Documents.
- `TwelveLabsMultimodalEmbedder` — embeds a single image, audio, or video (local path or URL) into
  the shared Marengo space; the cross-modal companion to `TwelveLabsTextEmbedder`.
- `TwelveLabsDocumentMultimodalEmbedder` — embeds the media referenced by each Document
  (via `meta["file_path"]`) for indexing.
- `TwelveLabsVideoConverter` — analyzes video with **Pegasus** and converts the result to Documents.

```python
from haystack import Document
from haystack_integrations.components.embedders.twelvelabs import (
    TwelveLabsMultimodalEmbedder,
    TwelveLabsDocumentMultimodalEmbedder,
)

# Set the TWELVELABS_API_KEY environment variable

# Embed a single media query (image / audio / video, local path or URL).
query_embedding = TwelveLabsMultimodalEmbedder().run(source="cat.jpg")["embedding"]

# Embed media Documents for indexing (modality inferred from the file extension).
docs = [Document(meta={"file_path": "clip.mp4"})]
docs = TwelveLabsDocumentMultimodalEmbedder().run(documents=docs)["documents"]
```

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run the integration tests locally, export a `TWELVELABS_API_KEY` environment variable (get a key at [playground.twelvelabs.io](https://playground.twelvelabs.io)).
