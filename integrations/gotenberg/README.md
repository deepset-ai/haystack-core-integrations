# gotenberg-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/gotenberg-haystack.svg)](https://pypi.org/project/gotenberg-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gotenberg-haystack.svg)](https://pypi.org/project/gotenberg-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/gotenberg)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/gotenberg/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

The integration tests require a local Gotenberg service. Start the provided Gotenberg v8 container from this directory:

```bash
docker compose up -d --wait
```

Then run the integration tests with `hatch run test:integration`. Stop the service afterward with `docker compose down`.
