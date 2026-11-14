# everos-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/everos-haystack.svg)](https://pypi.org/project/everos-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/everos-haystack.svg)](https://pypi.org/project/everos-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/everos)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/everos/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

Run unit tests with `hatch run test:unit`. Live integration tests require an
EverOS Cloud API key in `EVEROS_CLOUD_API_KEY` and run with `hatch run test:integration`.
They write synthetic test memories to your account and may consume quota. Use a dedicated test account.
Optionally set `EVEROS_TEST_BASE_URL` to an authorized test endpoint.

The integration class in `tests/test_memory_store.py` covers add/flush/search, default-add recall,
user isolation, metadata filters, and agent-case recall with agent isolation. Extraction is asynchronous
in some paths, so the tests use bounded polling. A timeout is a test failure, not a skipped assertion.
Synthetic test memories remain in the dedicated test account; use an approved retention/cleanup policy.
