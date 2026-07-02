# elasticsearch-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/elasticsearch-haystack.svg)](https://pypi.org/project/elasticsearch-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/elasticsearch-haystack.svg)](https://pypi.org/project/elasticsearch-haystack)

- [Integration page](https://haystack.deepset.ai/integrations/elasticsearch-document-store)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/elasticsearch/CHANGELOG.md)

---

## Connecting with security enabled

Since Elasticsearch 8.0, **security is on by default**: the cluster listens on HTTPS with a self-signed
certificate and requires authentication. Connecting to `http://localhost:9200` without credentials will fail
with a `ConnectionError` or `AuthenticationException`.

All extra keyword arguments passed to `ElasticsearchDocumentStore` are forwarded directly to the
underlying `elasticsearch-py` client, so every authentication and TLS option the client supports is
available here.

### Option 1 — API key (recommended for Elastic Cloud and self-managed clusters)

```python
import os
from haystack.utils import Secret
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

# base64-encoded "id:secret" string (the value shown in the Kibana "Create API key" dialog)
document_store = ElasticsearchDocumentStore(
    hosts="https://my-cluster.es.io:443",
    api_key=Secret.from_env_var("ELASTIC_API_KEY"),
)

# or pass it directly as a string (not recommended for production)
document_store = ElasticsearchDocumentStore(
    hosts="https://my-cluster.es.io:443",
    api_key="<base64-encoded-id:secret>",
)
```

### Option 2 — Username / password with TLS verification via CA certificate

This is the default setup for a self-managed cluster with a custom CA. Copy the CA certificate
from the cluster (usually at `/etc/elasticsearch/certs/http_ca.crt`) to your application host.

```python
document_store = ElasticsearchDocumentStore(
    hosts="https://localhost:9200",
    basic_auth=("elastic", os.environ["ELASTIC_PASSWORD"]),
    ca_certs="/path/to/http_ca.crt",
)
```

### Option 3 — Username / password with certificate fingerprint

A lighter alternative to copying the CA file. Obtain the fingerprint with:

```sh
openssl s_client -connect localhost:9200 -showcerts </dev/null 2>/dev/null \
  | openssl x509 -fingerprint -sha256 -noout
```

```python
document_store = ElasticsearchDocumentStore(
    hosts="https://localhost:9200",
    basic_auth=("elastic", os.environ["ELASTIC_PASSWORD"]),
    ssl_assert_fingerprint="AA:BB:CC:...",  # SHA-256 fingerprint without colons also accepted
)
```

### Option 4 — Disable TLS verification (development only)

Only use this in a local dev environment where the certificate cannot be trusted. **Do not use in production.**

```python
document_store = ElasticsearchDocumentStore(
    hosts="https://localhost:9200",
    basic_auth=("elastic", os.environ["ELASTIC_PASSWORD"]),
    verify_certs=False,
)
```

### Elastic Cloud (cloud_id)

```python
document_store = ElasticsearchDocumentStore(
    cloud_id="deployment-name:dXMtZWFzdC0x...",
    api_key=Secret.from_env_var("ELASTIC_API_KEY"),
)
```

For the full list of connection and authentication options, see the official
[elasticsearch-py connecting guide](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/connecting.html).

---

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run integration tests locally, you need a Docker container running ElasticSearch.
Use the provided `docker-compose.yml` file to start the container: `docker compose up -d`.
