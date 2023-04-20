# text2speech

[![PyPI - Version](https://img.shields.io/pypi/v/farm-haystack-text2speech.svg)](https://pypi.org/project/farm-haystack-text2speech)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/farm-haystack-text2speech.svg)](https://pypi.org/project/farm-haystack-text2speech)

---

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation


Only for Debian-based systems: first install the audio system dependencies:
```console
sudo apt-get install libsndfile1 ffmpeg
```

Install the text2speech package:
```console
pip install farm-haystack-text2speech
```


## Usage

For a full example of how to use the `AnswerToSpeech` Node, you may try out our "[Make Your QA Pipelines Talk Tutorial](https://haystack.deepset.ai/tutorials/17_audio)"

For example, in a simple Extractive QA Pipeline:

```
from haystack.nodes import BM25Retriever, FARMReader
from text2speech import AnswerToSpeech

retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
answer2speech = AnswerToSpeech(
    model_name_or_path="espnet/kan-bayashi_ljspeech_vits", generated_audio_dir=Path("./audio_answers")
)

audio_pipeline = Pipeline()
audio_pipeline.add_node(retriever, name="Retriever", inputs=["Query"])
audio_pipeline.add_node(reader, name="Reader", inputs=["Retriever"])
audio_pipeline.add_node(answer2speech, name="AnswerToSpeech", inputs=["Reader"])
```

## License

`haystack-text2speech` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
