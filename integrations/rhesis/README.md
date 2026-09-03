# rhesis-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/rhesis-haystack.svg)](https://pypi.org/project/rhesis-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rhesis-haystack.svg)](https://pypi.org/project/rhesis-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/rhesis)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/rhesis/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to export the following environment variables:

- `RHESIS_API_KEY`
- `RHESIS_BASE_URL` — optional, defaults to `http://localhost:8080`

```bash
cd integrations/rhesis
hatch run test:integration
```

Runnable examples live in [`example/`](example/).
