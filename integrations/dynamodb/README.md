# dynamodb-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/dynamodb-haystack.svg)](https://pypi.org/project/dynamodb-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dynamodb-haystack.svg)](https://pypi.org/project/dynamodb-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/dynamodb)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/dynamodb/CHANGELOG.md)

---

## Requirements

DynamoDB's native vector search (`SearchVectors`, vector indexes) is only available in
`boto3 >= 1.43.66`, which requires Python 3.10+. Both are enforced by this package's metadata.

## Running the integration tests

The integration tests run against **real AWS** — there is no local DynamoDB emulator that
supports vector indexes. They are skipped unless you opt in:

```bash
export AWS_DEFAULT_REGION=<your-region>
export HAYSTACK_DYNAMODB_INTEGRATION_TESTS=1
hatch run test:integration
```

You also need credentials (any standard boto3 credential source) with permission to
`CreateTable`, `DeleteTable`, `DescribeTable`, `ListTables`, the item-level operations, and
`SearchVectors`. Each test creates a uniquely-named `haystack_test_*` table and deletes it
afterwards; a class-scoped safety net sweeps any table whose delete was rejected while its
index was still settling. Expect the suite to create and destroy real, billable tables.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).