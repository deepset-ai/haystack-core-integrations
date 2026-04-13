# oracle-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/oracle-haystack.svg)](https://pypi.org/project/oracle-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/oracle-haystack.svg)](https://pypi.org/project/oracle-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/oracle/CHANGELOG.md)

---

## Installation

```bash
pip install oracle-haystack
```

## Usage

```python
from haystack.utils import Secret
from haystack_integrations.document_stores.oracle import OracleDocumentStore, OracleConnectionConfig

document_store = OracleDocumentStore(
    connection_config=OracleConnectionConfig(
        user="admin",
        password=Secret.from_env_var("ORACLE_PASSWORD"),
        dsn="localhost:1521/freepdb1",
    ),
    embedding_dim=1536,
)
```

For Oracle Autonomous Database (wallet connections):

```python
document_store = OracleDocumentStore(
    connection_config=OracleConnectionConfig(
        user="admin",
        password=Secret.from_env_var("ORACLE_PASSWORD"),
        dsn="mydb_low",
        wallet_location="/path/to/wallet",
        wallet_password=Secret.from_env_var("ORACLE_WALLET_PASSWORD"),
    ),
    embedding_dim=1536,
)
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

Integration tests require a live Oracle Database 23ai/26ai instance. Set the following environment variables:

```bash
export ORACLE_USER=admin
export ORACLE_PASSWORD=yourpassword
export ORACLE_DSN=localhost:1521/freepdb1

# Optional (for Autonomous Database):
export ORACLE_WALLET_LOCATION=/path/to/wallet
export ORACLE_WALLET_PASSWORD=walletpassword
```

Then run:

```bash
hatch run test:integration
```
