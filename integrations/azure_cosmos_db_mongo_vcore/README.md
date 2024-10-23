# azure-cosmos-db-mongo-vcore-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/azure-cosmos-db-mongo-vcore-haystack.svg)](https://pypi.org/project/azure-cosmos-db-mongo-vcore-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/azure-cosmos-db-mongo-vcore-haystack.svg)](https://pypi.org/project/azure-cosmos-db-mongo-vcore-haystack)

-----

**Table of Contents**

- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Installation

```console
pip install azure-cosmos-db-mongo-vcore-haystack
```

## Contributing

`hatch` is the best way to interact with this project, to install it:
```sh
pip install hatch
```

To run the linters `ruff` and `mypy`:
```
hatch run lint:all
```

To run all the tests:
```
hatch run test
```

Note: you need your own Azure CosmosDB Mongo vCore account to run the tests: you can make one here: 
https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/vcore/vector-search. Once you have it, export the connection string
to the env var `AZURE_COSMOS_MONGO_CONNECTION_STRING`. If you forget to do so, all the tests will be skipped.

## License

`mongodb-atlas-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
