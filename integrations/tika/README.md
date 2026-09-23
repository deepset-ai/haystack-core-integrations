# tika-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/tika-haystack.svg)](https://pypi.org/project/tika-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tika-haystack.svg)](https://pypi.org/project/tika-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/tika)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/tika/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

Integration tests require a running Tika server. Start one with:

```shell
docker run -d -p 127.0.0.1:9998:9998 apache/tika:3.3.1.0
```

Use a 3.x image, not `apache/tika:latest`: the `tika` Python client does not yet support Tika Server
4.x (TIKA-4816). `3.3.1.0` is the last 3.x image.
