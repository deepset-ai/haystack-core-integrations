# nvidia-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/nvidia-haystack.svg)](https://pypi.org/project/nvidia-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nvidia-haystack.svg)](https://pypi.org/project/nvidia-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/nvidia)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/nvidia/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to export the `NVIDIA_API_KEY` environment variable. Some tests require additional environment variables:
- `NVIDIA_NIM_EMBEDDER_MODEL`
- `NVIDIA_NIM_ENDPOINT_URL`
- `NVIDIA_NIM_GENERATOR_MODEL`
- `NVIDIA_NIM_RANKER_MODEL`
- `NVIDIA_NIM_RANKER_ENDPOINT_URL`