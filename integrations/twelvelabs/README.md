# twelvelabs-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/twelvelabs-haystack.svg)](https://pypi.org/project/twelvelabs-haystack)

[Haystack](https://haystack.deepset.ai/) integration for [TwelveLabs](https://twelvelabs.io) video understanding — **Marengo** multimodal embeddings and **Pegasus** video analysis.

```bash
pip install twelvelabs-haystack
```

Set the `TWELVELABS_API_KEY` environment variable (get a key at [playground.twelvelabs.io](https://playground.twelvelabs.io)).

## Components

### `TwelveLabsVideoConverter` (Pegasus)

Converts videos into Haystack `Document`s. Pegasus analyzes each video on the fly (visuals **and** its own audio ASR) and returns text — so a video becomes a `Document` whose content is the analysis (e.g. a description + transcript), with no frame extraction.

```python
from haystack_integrations.components.converters.twelvelabs import TwelveLabsVideoConverter

converter = TwelveLabsVideoConverter()
docs = converter.run(sources=["https://example.com/clip.mp4"])["documents"]
print(docs[0].content)
```

Sources may be publicly accessible direct video URLs or local file paths (uploaded to TwelveLabs, ≤ 200 MB).

### `TwelveLabsTextEmbedder` and `TwelveLabsDocumentEmbedder` (Marengo)

Embed text/Documents with Marengo. Because Marengo embeds text, images, audio, and video into one shared vector space, these embeddings support cross-modal retrieval (e.g. text-to-image/video search).

```python
from haystack import Document
from haystack_integrations.components.embedders.twelvelabs import (
    TwelveLabsTextEmbedder,
    TwelveLabsDocumentEmbedder,
)

text_embedder = TwelveLabsTextEmbedder()
print(text_embedder.run(text="a cat playing piano")["embedding"])

doc_embedder = TwelveLabsDocumentEmbedder()
docs = doc_embedder.run(documents=[Document(content="a cat playing piano")])["documents"]
print(docs[0].embedding)
```

## License

`twelvelabs-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
