# vllm-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/vllm-haystack.svg)](https://pypi.org/project/vllm-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vllm-haystack.svg)](https://pypi.org/project/vllm-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/vllm/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to have a running vLLM server. Refer to the [workflow file](https://github.com/deepset-ai/haystack-core-integrations/blob/main/.github/workflows/vllm.yml) for more details.

For example, on macOs, you can install [vLLM-metal](https://github.com/vllm-project/vllm-metal) and run the server with:

```bash
source ~/.venv-vllm-metal/bin/activate && vllm serve Qwen/Qwen3.5-0.8B --reasoning-parser qwen3 --max-model-len 1024 --enforce-eager  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```