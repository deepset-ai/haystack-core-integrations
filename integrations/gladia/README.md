# gladia-haystack

Haystack integration for [Gladia](https://www.gladia.io/) batch audio transcription API.

## Installation

```bash
pip install gladia-haystack
```

## Usage Example

```python
from haystack_integrations.components.audio.gladia import GladiaTranscriber

transcriber = GladiaTranscriber()
results = transcriber.run(sources=["path/to/audio.mp3"])

for doc in results["documents"]:
    print(doc.content)
```
