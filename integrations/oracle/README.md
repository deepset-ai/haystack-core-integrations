# oracle-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/oracle-haystack.svg)](https://pypi.org/project/oracle-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/oracle-haystack.svg)](https://pypi.org/project/oracle-haystack)

Haystack DocumentStore backed by [Oracle AI Vector Search](https://www.oracle.com/database/ai-vector-search/), available in Oracle Database 23ai and later.

---

## Installation

## Usage


## Contributing


### Running tests

#### Unit tests


#### Integration tests against a live Oracle instance

#### Integration tests via Docker (local Oracle 23ai Free)

A `docker-compose.yml` is provided that runs [`gvenzl/oracle-free:23-slim`](https://hub.docker.com/r/gvenzl/oracle-free) (Oracle Database 23ai Free edition).

```bash
docker compose up -d
```

On first boot Oracle initialises its data files, which takes roughly 60 seconds. 
Wait until the container reports `(healthy)`:

Run the Docker-backed integration tests

```bash
hatch run test:integration -vvv tests/integration/test_docker_document_store.py
```