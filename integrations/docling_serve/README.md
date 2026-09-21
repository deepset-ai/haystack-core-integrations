# docling-serve-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/docling-serve-haystack.svg)](https://pypi.org/project/docling-serve-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/docling-serve-haystack.svg)](https://pypi.org/project/docling-serve-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/docling-serve)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/docling_serve/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need a Docker container running docling-serve.
You can start it using Docker:

```console
docker run -d -p 5001:5001 ghcr.io/docling-project/docling-serve-cpu:latest
```

Then export the `DOCLING_SERVE_URL` environment variable: `export DOCLING_SERVE_URL=http://localhost:5001`.
