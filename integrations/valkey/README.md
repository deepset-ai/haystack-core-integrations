# valkey-haystack

 - [Integration page](https://haystack.deepset.ai/integrations/valkey)

### Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

### Running Tests

To run integration tests locally, you need a running Valkey instance. You can start one using Docker:

```bash
docker run -d -p 6379:6379 valkey/valkey-bundle:latest
```

Navigate to the integration directory and set up environment variables:

```bash
cd integrations/valkey

# Sync dependencies including test dependencies
uv sync --group test

# Run unit tests only
hatch run test:unit

# Run integration tests only (requires Valkey instance)
hatch run test:integration

# Run all tests
hatch run test:all
```
