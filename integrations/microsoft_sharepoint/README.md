# microsoft-sharepoint-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/microsoft-sharepoint-haystack.svg)](https://pypi.org/project/microsoft-sharepoint-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/microsoft-sharepoint-haystack.svg)](https://pypi.org/project/microsoft-sharepoint-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/microsoft_sharepoint/CHANGELOG.md)

## Running the tests

```bash
hatch run test:unit          # offline, no credentials needed
hatch run test:integration   # hits live Microsoft Graph (see below)
```

The integration tests (the `TestLive` classes in `tests/`) exercise the retriever and the fetcher
against live Microsoft Graph. They **skip themselves** when their inputs are not set, so the default
local and CI runs stay offline. To run them you need a delegated Microsoft Graph access token.
