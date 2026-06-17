# oauth-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/oauth-haystack.svg)](https://pypi.org/project/oauth-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/oauth-haystack.svg)](https://pypi.org/project/oauth-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/oauth/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

The integration test resolves a token against a real identity provider and is skipped unless the required environment variables are set. To run it locally, export:

- `OAUTH_TOKEN_URL` — the provider's OAuth 2.0 token endpoint.
- `OAUTH_CLIENT_ID` — the OAuth client identifier.
- `OAUTH_REFRESH_TOKEN` — a valid refresh token for that client.
- `OAUTH_CLIENT_SECRET` — *(optional)* the client secret for confidential clients; omit it for public clients.
- `OAUTH_SCOPES` — *(optional)* space-separated scopes to request.

Then run:

```bash
hatch run test:integration
```
