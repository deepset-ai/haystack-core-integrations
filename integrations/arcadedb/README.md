# arcadedb-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/arcadedb-haystack.svg)](https://pypi.org/project/arcadedb-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/arcadedb-haystack.svg)](https://pypi.org/project/arcadedb-haystack)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.txt)

- [Integration page](https://haystack.deepset.ai/integrations/arcadedb)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/arcadedb/CHANGELOG.md)

---

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need to have an ArcadeDB server running.
You can start it using Docker:

```console
docker run -d -p 2480:2480 -e JAVA_OPTS="-Darcadedb.server.rootPassword=arcadedb" arcadedata/arcadedb:latest
```

Then run the integration tests, providing the credentials via environment variables:

```console
ARCADEDB_USERNAME=root ARCADEDB_PASSWORD=arcadedb hatch run test:integration
```
