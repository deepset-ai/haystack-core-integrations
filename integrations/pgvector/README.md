# pgvector-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/pgvector-haystack.svg)](https://pypi.org/project/pgvector-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pgvector-haystack.svg)](https://pypi.org/project/pgvector-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/pgvector-documentstore)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/pgvector/CHANGELOG.md)

---

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to have a PostgreSQL DB running with the `pgvector` extension.
You can start it using Docker:

```console
docker run -d -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=postgres pgvector/pgvector:pg17
```
