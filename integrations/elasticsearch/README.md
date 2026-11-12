# elasticsearch-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/elasticsearch-haystack.svg)](https://pypi.org/project/elasticsearch-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/elasticsearch-haystack.svg)](https://pypi.org/project/elasticsearch-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/elasticsearch-document-store)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/elasticsearch/CHANGELOG.md)

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need a Docker container running ElasticSearch.
Use the provided `docker-compose.yml` file to start the container: `docker compose up -d`.

The `test_cloud_*` integration tests run against a managed Elastic Cloud project and are skipped unless the
required environment variables are set:

| Variable | Required | Default | Description |
|---|---|---|---|
| `ELASTICSEARCH_URL` | yes | – | Elastic Cloud endpoint, e.g. `https://<id>.es.<region>.elastic.cloud:443` |
| `ELASTIC_API_KEY` | yes | – | Base64-encoded API key |
| `ELASTICSEARCH_INFERENCE_ID` | no | `.elser-2-elastic` | Sparse (ELSER) inference endpoint used by the sparse and hybrid tests |
| `ELASTICSEARCH_DENSE_INFERENCE_ID` | no | `.multilingual-e5-small-elasticsearch` | Dense text-embedding inference endpoint used by the dense tests |
| `ELASTICSEARCH_DENSE_EMBEDDING_DIMS` | no | `384` | Output dimension of the dense model, used by the dense ingest pipeline tests |
