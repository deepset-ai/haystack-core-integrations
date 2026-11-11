# vllm-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/vllm-haystack.svg)](https://pypi.org/project/vllm-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/vllm-haystack.svg)](https://pypi.org/project/vllm-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/vllm)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/vllm/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need vLLM servers running in parallel: one for the chat generator on port `8000`, one for the embedders on port `8001`, one for the ranker on port `8002`, and one for the sparse embedders on port `8003`. Refer to the [workflow file](https://github.com/deepset-ai/haystack-core-integrations/blob/main/.github/workflows/vllm.yml) for more details.

For example, on macOs, you can install [vLLM-metal](https://github.com/vllm-project/vllm-metal) and start the chat generator server with:

```bash
# chat generator server (port 8000)
source ~/.venv-vllm-metal/bin/activate && vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3 --max-model-len 1024 --enforce-eager --enable-auto-tool-choice --tool-call-parser hermes
```

vLLM-metal does not support embedding models. On macOS, you can run the embedding server via CPU Docker image:

```bash
# embedders server (port 8001)
docker run --rm -p 8001:8000 -e VLLM_CPU_OMP_THREADS_BIND=0-3 vllm/vllm-openai-cpu:latest \
    --model sentence-transformers/all-MiniLM-L6-v2 --enforce-eager
```

To run the ranker server, use CPU Docker image:
```bash
# ranker server (port 8002)
docker run --rm -p 8002:8000 -e VLLM_CPU_OMP_THREADS_BIND=0-3 vllm/vllm-openai-cpu:latest \
    --model BAAI/bge-reranker-base --enforce-eager
```

To run the sparse embedders server, use CPU Docker image. SPLADE checkpoints need the `--hf-overrides` flag to be served as sparse embedding models:
```bash
# sparse embedders server (port 8003)
docker run --rm -p 8003:8000 --shm-size=1g -e VLLM_CPU_KVCACHE_SPACE=1 vllm/vllm-openai-cpu:latest \
    prithivida/Splade_PP_en_v1 --runner pooling \
    --hf-overrides '{"architectures": ["BertSpladeSparseEmbeddingModel"]}' --enforce-eager
```
