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

### Regenerate gRPC files

Generate the files from the upstream [TEI proto](https://github.com/huggingface/text-embeddings-inference/blob/main/proto/tei.proto):

```bash
curl -Lo src/haystack_integrations/components/embedders/huggingface_api/_grpc/tei.proto https://raw.githubusercontent.com/huggingface/text-embeddings-inference/main/proto/tei.proto
pip install grpcio-tools mypy-protobuf
python -m grpc_tools.protoc --proto_path=src --python_out=src --pyi_out=src --grpc_python_out=src --mypy_grpc_out=src haystack_integrations/components/embedders/huggingface_api/_grpc/tei.proto
```
