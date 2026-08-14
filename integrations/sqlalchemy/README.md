# sqlalchemy-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/sqlalchemy-haystack.svg)](https://pypi.org/project/sqlalchemy-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sqlalchemy-haystack.svg)](https://pypi.org/project/sqlalchemy-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/sqlalchemy)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/sqlalchemy/CHANGELOG.md)

---

## Security

`SQLAlchemyTableRetriever` executes raw SQL queries passed at runtime. Keep the following in mind:

- **Never pass unsanitised user input** directly as a query — this exposes you to SQL injection.
- **Use a read-only database user.** This is the most effective mitigation. Even if a malicious
  query is executed, a read-only user cannot modify or delete data.
- **Restrict database permissions** to the minimum required — specific tables and schemas only,
  no DDL privileges (no `CREATE`, `DROP`, `ALTER`).

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
