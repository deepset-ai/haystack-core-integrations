# azure-form-recognizer-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/azure-form-recognizer-haystack.svg)](https://pypi.org/project/azure-form-recognizer-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/azure-form-recognizer-haystack.svg)](https://pypi.org/project/azure-form-recognizer-haystack)

Haystack integration for the `AzureOCRDocumentConverter`, which converts files to Haystack Documents using
Azure's Document Intelligence service via the `azure-ai-formrecognizer` SDK.

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/azure_form_recognizer/CHANGELOG.md)

---

## Installation

```bash
pip install azure-form-recognizer-haystack
```

## Usage

```python
from haystack_integrations.components.converters.azure_form_recognizer import AzureOCRDocumentConverter
from haystack.utils import Secret

converter = AzureOCRDocumentConverter(
    endpoint="<your-azure-resource-endpoint>",
    api_key=Secret.from_env_var("AZURE_AI_API_KEY"),
)
results = converter.run(sources=["document.pdf"])
documents = results["documents"]
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
