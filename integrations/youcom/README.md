# youcom-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/youcom-haystack.svg)](https://pypi.org/project/youcom-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/youcom-haystack.svg)](https://pypi.org/project/youcom-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/youcom/CHANGELOG.md)

---

## Usage

`YouComWebSearch` runs without any configuration: with no API key available it uses the You.com
[keyless free tier](https://you.com/docs/api-reference/search/v1-agents-search) (rate limited per IP).
Set the `YOUDOTCOM_API_KEY` environment variable to use the keyed
[You.com Search API](https://you.com/docs/api-reference/search/v1-search) with higher limits.

```python
from haystack_integrations.components.websearch.youcom import YouComWebSearch

websearch = YouComWebSearch(top_k=5)
result = websearch.run(query="What is Haystack by deepset?")
result["documents"]  # list[Document]
result["links"]  # list[str]
```

To require a key instead of silently degrading to the keyless tier, pass `keyless_fallback=False`;
the component then raises `YouComError` when no key resolves.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

The unit tests are fully mocked and need no credentials. The integration tests include one live keyless
search that needs no key; to also run the keyed tests, export the `YOUDOTCOM_API_KEY` environment variable.
