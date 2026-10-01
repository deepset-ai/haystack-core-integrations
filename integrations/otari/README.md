# otari-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/otari-haystack.svg)](https://pypi.org/project/otari-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/otari-haystack.svg)](https://pypi.org/project/otari-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/otari)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/otari/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

The integration tests run against a local Otari gateway. To start one, export `OPENAI_API_KEY` and `COHERE_API_KEY`
for the providers it routes to, and run:

```bash
docker compose up -d --wait
```

Then create an API key with the gateway's master key, export it as `OTARI_API_KEY`, and run the integration tests:

```bash
export OTARI_API_KEY=$(curl -sSf -X POST http://localhost:8000/api/v1/keys \
  -H "Authorization: Bearer otari-haystack-tests" -H "Content-Type: application/json" \
  -d '{"key_name": "haystack-tests"}' | jq -r .key)
hatch run test:integration
```

Stop the gateway afterward with `docker compose down`.

To run the chat generator tests against otari.ai instead, export an otari.ai API key as `OTARI_API_KEY` and the
API root of your account's region as `OTARI_API_BASE_URL`, for example `https://eu.api.otari.ai/api/v1`, and run:

```bash
hatch run test:integration -k chat_generator
```
