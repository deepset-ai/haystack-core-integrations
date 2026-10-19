# huggingface-api-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/huggingface-api-haystack.svg)](https://pypi.org/project/huggingface-api-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/huggingface-api-haystack.svg)](https://pypi.org/project/huggingface-api-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/huggingface-api)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/huggingface_api/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need Docker containers running a few Text Embeddings Inference (TEI) servers. Use the provided `docker-compose.yml` file to start the containers: `docker compose up -d`.

The tests that call the Hugging Face Inference API are skipped unless `HF_TOKEN` is set as an evironment variable.
