# solr-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/solr-haystack.svg)](https://pypi.org/project/solr-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/solr-haystack.svg)](https://pypi.org/project/solr-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/solr)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/solr/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need a Docker container running Solr.
Use the provided `docker-compose.yml` file to start the container: `docker compose up -d --wait`.

The document store requires Solr 9.6 or newer. To test against the oldest supported release, override
the image. Recreate the volume as well, because an older Solr cannot open a newer one's index:

```console
docker compose down -v
SOLR_IMAGE=solr:9.10.1 docker compose up -d --wait
```
