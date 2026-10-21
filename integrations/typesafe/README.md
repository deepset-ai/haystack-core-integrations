# typesafe-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/typesafe-haystack.svg)](https://pypi.org/project/typesafe-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/typesafe-haystack.svg)](https://pypi.org/project/typesafe-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/typesafe)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/typesafe/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

### Integration tests

Integration tests run against a local [Ollaya](https://ollaya.dev) server, which serves open decision models through the TypeSafe API:

```bash
docker run -d --name ollaya -p 11435:11435 -v ollaya:/home/ollaya/.ollaya ghcr.io/ollaya-dev/ollaya
docker exec ollaya ollaya pull laya:en
TYPESAFE_BASE_URL=http://localhost:11435 TYPESAFE_API_KEY=local hatch run test:integration
```
