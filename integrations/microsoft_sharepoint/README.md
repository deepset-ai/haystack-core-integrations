# microsoft-sharepoint-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/microsoft-sharepoint-haystack.svg)](https://pypi.org/project/microsoft-sharepoint-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/microsoft-sharepoint-haystack.svg)](https://pypi.org/project/microsoft-sharepoint-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/microsoft-sharepoint)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/microsoft_sharepoint/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

The integration tests run against live Microsoft Graph and are skipped unless their
environment variables are set:

- `MS_SHAREPOINT_ACCESS_TOKEN` — a delegated Microsoft Graph access token; required by the
  retriever and fetcher tests.
- `MS_SHAREPOINT_TEST_FILE_URL` — a SharePoint file URL; also required by the fetcher tests.
- `MS_GRAPH_TENANT_ID`, `MS_GRAPH_CLIENT_ID` and `MS_GRAPH_CLIENT_SECRET` — required by the
  app-only retriever tests.
