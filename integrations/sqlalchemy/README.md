# sqlalchemy-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/sqlalchemy-haystack.svg)](https://pypi.org/project/sqlalchemy-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sqlalchemy-haystack.svg)](https://pypi.org/project/sqlalchemy-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/sqlalchemy/CHANGELOG.md)

---

A Haystack integration for querying SQL databases via [SQLAlchemy](https://www.sqlalchemy.org/).
Provides a `SQLAlchemyTableRetriever` component that connects to any SQLAlchemy-supported database,
executes a SQL query, and returns results as a Pandas DataFrame and an optional Markdown table.

## Installation

```bash
pip install sqlalchemy-haystack
```

You also need to install the appropriate database driver for your backend:

| Backend | Driver package |
|---------|---------------|
| PostgreSQL | `psycopg2-binary` or `psycopg[binary]` |
| MySQL / MariaDB | `pymysql` or `mysqlclient` |
| SQLite | built-in (no extra package needed) |
| MSSQL | `pyodbc` |
| Oracle | `cx_oracle` or `oracledb` |

## Usage

```python
from haystack_integrations.components.retrievers.sqlalchemy import SQLAlchemyTableRetriever

# SQLite in-memory example (no driver needed)
retriever = SQLAlchemyTableRetriever(
    drivername="sqlite",
    database=":memory:",
    init_script=[
        "CREATE TABLE products (id INTEGER, name TEXT, price REAL)",
        "INSERT INTO products VALUES (1, 'Widget', 9.99)",
        "INSERT INTO products VALUES (2, 'Gadget', 19.99)",
    ],
)
retriever.warm_up()

result = retriever.run(query="SELECT * FROM products WHERE price < 15")
print(result["dataframe"])
print(result["table"])
```

For PostgreSQL:

```python
from haystack.utils import Secret

retriever = SQLAlchemyTableRetriever(
    drivername="postgresql+psycopg2",
    host="localhost",
    port=5432,
    database="mydb",
    username="myuser",
    password=Secret.from_env_var("DB_PASSWORD"),
)
```

## Security

This component executes raw SQL queries passed at runtime. Keep the following in mind:

- **Never pass unsanitised user input** directly as a query — this exposes you to SQL injection.
- **Use a read-only database user.** This is the most effective mitigation. Even if a malicious
  query is executed, a read-only user cannot modify or delete data.
- **Restrict database permissions** to the minimum required — specific tables and schemas only,
  no DDL privileges (no `CREATE`, `DROP`, `ALTER`).

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
