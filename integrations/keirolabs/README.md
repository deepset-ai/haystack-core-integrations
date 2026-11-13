# keirolabs-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/keirolabs-haystack.svg)](https://pypi.org/project/keirolabs-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/keirolabs-haystack.svg)](https://pypi.org/project/keirolabs-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/keirolabs)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/keirolabs/CHANGELOG.md)

---

`KeirolabsWebSearch` searches the web with [Keirolabs](https://keirolabs.cloud) and returns the results as Haystack Documents (the snippet as content, with title and URL in the metadata) plus a list of links.

```bash
pip install keirolabs-haystack
export KEIROLABS_API_KEY="keiro_..."  # https://platform.keirolabs.cloud
```

```python
from haystack_integrations.components.websearch.keirolabs import KeirolabsWebSearch

websearch = KeirolabsWebSearch(top_k=5)
result = websearch.run(query="What is Haystack by deepset?")
print(result["links"])
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

Integration tests call the live Keirolabs API and need a `KEIROLABS_API_KEY` environment variable. They are skipped when it isn't set.
