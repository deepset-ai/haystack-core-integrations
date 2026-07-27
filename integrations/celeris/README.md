# celeris-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/celeris-haystack.svg)](https://pypi.org/project/celeris-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/celeris-haystack.svg)](https://pypi.org/project/celeris-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/celeris/CHANGELOG.md)

---

[Celeris](https://celeris.ai) serves diffusion-based language models over an OpenAI-compatible API, optimized for
low latency on short responses. This integration provides `CelerisChatGenerator`, which is a good fit for
classification, extraction, judging, query rewriting, and other short structured responses.

## Installation

```console
pip install celeris-haystack
```

## Usage

Set the `CELERIS_API_KEY` environment variable and run the generator:

```python
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.celeris import CelerisChatGenerator

generator = CelerisChatGenerator()
result = generator.run([ChatMessage.from_user("What's Natural Language Processing?")])
print(result["replies"][0].text)
```

The default model is `celeris-1` and the default base URL is `https://inference.celeris.ai/celeris-1/v1`.
Celeris encodes the model in the endpoint path, so a different model also requires a different `api_base_url`.

## Celeris-specific behaviour

Celeris has a few hard API constraints. The component enforces them so you get a clear Python error instead of an
opaque HTTP 400:

| Constraint | What the component does |
| --- | --- |
| `max_tokens` must be `1` or a positive multiple of 256 | Rounds your value **up** to the next multiple of 256; defaults to `1024` when unset. `1` is passed through as a warm ping. |
| Prompt and completion share a single 8192-token context | Caps `max_tokens` **down** to a multiple of 256 that fits the remaining budget, and raises `ValueError` if the prompt leaves less than 256 tokens of room. |
| No `response_format` (no JSON mode, no JSON schema) | Raises `ValueError`. Use tool calling for structured output. |
| No image / multimodal input | Raises `ValueError` if a message carries an image. |
| No `tool_choice="required"` | Raises `ValueError`. Use `"auto"`, `"none"`, or a named tool. |

Streaming and tool calling are supported. Celeris generates in blocks, so streamed chunks arrive in large groups
rather than one at a time.

`temperature`, `top_p`, `seed`, `stop`, `n`, `presence_penalty`, `frequency_penalty` and `logprobs` /
`top_logprobs` are passed through to the API. The Celeris API reference lists `logprobs` as unsupported, but the
endpoint accepts it and returns per-token log probabilities, so the component does not reject it.

Because Celeris does not publish a tokenizer, the prompt size used for the context check is a conservative
character-based estimate. It errs on the side of over-estimating, which shortens the completion rather than
letting the request fail.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to export the `CELERIS_API_KEY` environment variable.
