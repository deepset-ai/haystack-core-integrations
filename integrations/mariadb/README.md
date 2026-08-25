# mariadb-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/mariadb-haystack.svg)](https://pypi.org/project/mariadb-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mariadb-haystack.svg)](https://pypi.org/project/mariadb-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/mariadb-documentstore)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/mariadb/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

The `mariadb` connector is a C extension built from source, so it needs the MariaDB Connector/C system library
(`mariadb_config`): `sudo apt-get install -y libmariadb-dev` on Ubuntu, or `brew install mariadb-connector-c`
on macOS.

To run integration tests locally, you need a MariaDB 11.7+ instance running.
You can start one using Docker:

```console
docker run -d --name mariadb-haystack -e MARIADB_ROOT_PASSWORD=password -e MARIADB_DATABASE=haystack -p 3306:3306 mariadb:11.7
```
