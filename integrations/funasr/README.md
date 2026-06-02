# FunASR Haystack Integration

Use the `FunASRTranscriber` component to transcribe audio files with [FunASR](https://github.com/modelscope/FunASR).

## Installation

```bash
pip install funasr-haystack
```

## Usage

```python
from haystack_integrations.components.audio.funasr import FunASRTranscriber

transcriber = FunASRTranscriber(
    model="paraformer-zh",
    device="cpu",
    speaker_diarization=True,
)

result = transcriber.run(sources=["meeting.wav"])
document = result["documents"][0]

print(document.content)
print(document.meta["sentence_info"])
```

Use `device="cuda:0"` for GPU inference. To use a model that emits emotion or audio-event tags, such as SenseVoice,
enable tag extraction with `emotion_detection=True`.
