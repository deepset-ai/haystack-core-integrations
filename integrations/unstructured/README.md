# unstructured-fileconverter-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/unstructured-fileconverter-haystack.svg)](https://pypi.org/project/unstructured-fileconverter-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/unstructured-fileconverter-haystack.svg)](https://pypi.org/project/unstructured-fileconverter-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/unstructured-file-converter)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/unstructured/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to have a Docker container running the Unstructured API.
You can start it as follows:

```console
docker run -p 8000:8000 -d --rm --name unstructured-api quay.io/unstructured-io/unstructured-api:latest --port 8000 --host 0.0.0.0
```