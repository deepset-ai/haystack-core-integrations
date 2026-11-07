# minicpm-ocr-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/minicpm-ocr-haystack.svg)](https://pypi.org/project/minicpm-ocr-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/minicpm-ocr-haystack.svg)](https://pypi.org/project/minicpm-ocr-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/minicpm-ocr)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/minicpm_ocr/CHANGELOG.md)

---

## Installation

```console
pip install minicpm-ocr-haystack
```

## Usage

`MiniCPMOCRDocumentConverter` turns PDF and image files into Haystack `Document`s
using [MiniCPM-OCR](https://github.com/OpenBMB/MiniCPM-OCR), a 1.3B open OCR model
specialized in document parsing.

The converter runs the official `minicpm-ocr` SDK in-process: it detects each
page's layout with PP-DocLayout, crops the detected blocks, and asks a
MiniCPM-OCR inference server to recognize them. The server can be a local
vLLM, SGLang, llama.cpp or Ollama deployment, or any OpenAI-compatible
endpoint serving the same model.

```python
from haystack_integrations.components.converters.minicpm_ocr import MiniCPMOCRDocumentConverter

converter = MiniCPMOCRDocumentConverter(
    server_url="http://localhost:8000",
    model="minicpm-ocr",
)
result = converter.run(sources=["invoice.pdf", "scan.png"])
documents = result["documents"]
```

Each `Document` contains the page Markdown, plus `page_count` and
`ocr_engine: "minicpm-ocr"` metadata. The original source metadata is preserved.

Supported inputs are PDF, PNG, JPEG, BMP, TIFF and WebP.

### Running the inference server

```console
vllm serve openbmb/MiniCPM-OCR-1.0 --port 8000
```

Set `backend` to `ollama`, `llamacpp`, `vllm`, or `openai` (SGLang and LMDeploy
expose the OpenAI-compatible API, so use `openai`). When the server requires a
key, pass `api_key=Secret.from_env_var("MINICPM_OCR_API_KEY")`.

### Layout model

The layout detector runs locally. By default the SDK downloads PP-DocLayout into
`~/.cache/minicpm-ocr/layout` on first use; set `layout_model_dir` to reuse a
directory that already contains the model. Layout detection runs on CPU unless
`layout_device` is set to a CUDA device.

### Limitations

- OCR quality depends on the inference server; tables and formulas require a
  model and prompt set that support them.
- The first call downloads the layout model, which can take a minute on a slow
  connection.
- Scanned pages are recognized block by block, so very large PDFs take time and
  memory; adjust `page_workers` and `max_workers` to the server capacity.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

## License

`minicpm-ocr-haystack` is distributed under the terms of the
[Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
