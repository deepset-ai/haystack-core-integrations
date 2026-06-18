# whisper-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/whisper-haystack.svg)](https://pypi.org/project/whisper-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/whisper-haystack.svg)](https://pypi.org/project/whisper-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/whisper/CHANGELOG.md)

---

**Table of Contents**

- [Installation](#installation)
- [Contributing](#contributing)

## Installation

```console
pip install whisper-haystack
```

This installs `RemoteWhisperTranscriber`, which transcribes audio with the OpenAI Whisper API and only needs an
`OPENAI_API_KEY`.

To also use `LocalWhisperTranscriber`, which runs Whisper on your machine, install the optional `openai-whisper`
dependency and make sure [`ffmpeg`](https://ffmpeg.org/) is available on your system:

```console
pip install "openai-whisper>=20231106"
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run the integration tests locally, export an `OPENAI_API_KEY` environment variable (for `RemoteWhisperTranscriber`)
and install `openai-whisper` and `ffmpeg` (for `LocalWhisperTranscriber`).
