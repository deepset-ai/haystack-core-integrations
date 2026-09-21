# alloydb-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/alloydb-haystack.svg)](https://pypi.org/project/alloydb-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/alloydb-haystack.svg)](https://pypi.org/project/alloydb-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/alloydb-documentstore)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/alloydb/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need a running AlloyDB instance.
Export the connection settings and run the tests:

```console
export ALLOYDB_INSTANCE_URI="projects/MY_PROJECT/locations/MY_REGION/clusters/MY_CLUSTER/instances/MY_INSTANCE"
export ALLOYDB_USER="my-db-user"
export ALLOYDB_PASSWORD="my-db-password"

hatch run test:integration
```
