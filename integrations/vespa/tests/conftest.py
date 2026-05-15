# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: S603,S607

import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from haystack_integrations.document_stores.vespa import VespaDocumentStore

HOSTS_XML = """<?xml version="1.0" encoding="utf-8" ?>
<hosts>
  <host name="localhost">
    <alias>node1</alias>
  </host>
</hosts>
"""

SERVICES_XML = """<?xml version="1.0" encoding="utf-8" ?>
<services version="1.0">
  <container id="default" version="1.0">
    <search />
    <document-api />
  </container>
  <content id="content" version="1.0">
    <redundancy>1</redundancy>
    <documents>
      <document type="doc" mode="index" />
    </documents>
    <nodes>
      <node hostalias="node1" distribution-key="0" />
    </nodes>
  </content>
</services>
"""

DOC_SCHEMA = """schema doc {
    document doc {
        field content type string {
            indexing: index | summary
            index: enable-bm25
        }

        field embedding type tensor<float>(x[3]) {
            indexing: attribute | summary
            attribute {
                distance-metric: angular
            }
        }

        field category type string {
            indexing: attribute | summary
        }

        field author type string {
            indexing: attribute | summary
        }

        field name type string {
            indexing: attribute | summary
        }

        field page type string {
            indexing: attribute | summary
        }

        field chapter type string {
            indexing: attribute | summary
        }

        field number type int {
            indexing: attribute | summary
        }

        field date type string {
            indexing: attribute | summary
        }

        field no_embedding type bool {
            indexing: attribute | summary
        }

        field year type int {
            indexing: attribute | summary
        }

        field status type string {
            indexing: attribute | summary
        }

        field updated type bool {
            indexing: attribute | summary
        }

        field extra_field type string {
            indexing: attribute | summary
        }

        field featured type bool {
            indexing: attribute | summary
        }

        field priority type int {
            indexing: attribute | summary
        }

        field rating type double {
            indexing: attribute | summary
        }

        field age type int {
            indexing: attribute | summary
        }
    }

    fieldset default {
        fields: content
    }

    rank-profile bm25 {
        first-phase {
            expression: bm25(content)
        }
    }

    rank-profile semantic {
        inputs {
            query(query_embedding) tensor<float>(x[3])
        }

        first-phase {
            expression: closeness(field, embedding)
        }
    }
}
"""


@pytest.fixture(scope="session")
def deployed_vespa_app(tmp_path_factory):
    """Deploy the Vespa test schema into the Dockerized Vespa instance (integration tests only)."""
    if os.environ.get("VESPA_RUN_INTEGRATION_TESTS") != "1":
        pytest.skip("Set VESPA_RUN_INTEGRATION_TESTS=1 and start Vespa with docker compose.")

    app_dir = _write_vespa_app(tmp_path_factory.mktemp("vespa_app"))
    subprocess.run(["docker", "cp", str(app_dir), "vespa:/tmp/vespa_app"], check=True)
    subprocess.run(
        [
            "docker",
            "exec",
            "vespa",
            "bash",
            "-lc",
            "/opt/vespa/bin/vespa-deploy prepare /tmp/vespa_app && /opt/vespa/bin/vespa-deploy activate",
        ],
        check=True,
    )
    _wait_for_vespa_endpoint("http://localhost:8080/ApplicationStatus", deadline_s=300)


def _write_vespa_app(app_dir: Path) -> Path:
    schemas_dir = app_dir / "schemas"
    schemas_dir.mkdir()
    (app_dir / "hosts.xml").write_text(HOSTS_XML, encoding="utf-8")
    (app_dir / "services.xml").write_text(SERVICES_XML, encoding="utf-8")
    (schemas_dir / "doc.sd").write_text(DOC_SCHEMA, encoding="utf-8")
    return app_dir


def _wait_for_vespa_endpoint(url: str, *, deadline_s: float) -> None:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1)
    msg = f"Timed out waiting for Vespa endpoint {url}"
    raise AssertionError(msg)


def wait_until_documents_count(document_store, expected_count: int, *, deadline_s: float = 90) -> None:
    """Poll until Vespa search visibility matches `expected_count` (best-effort for CI)."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if document_store.count_documents() == expected_count:
            return
        time.sleep(0.5)
    msg = f"Timed out waiting for {expected_count} documents to be visible in Vespa."
    raise AssertionError(msg)


@pytest.fixture
def document_store(deployed_vespa_app, request):  # noqa: ARG001
    """Shared populated Vespa store for Haystack integration tests (see DocumentStoreBaseTests)."""
    _metadata_fields = [
        "category",
        "author",
        "name",
        "page",
        "chapter",
        "number",
        "date",
        "no_embedding",
        "year",
        "status",
        "updated",
        "extra_field",
        "featured",
        "priority",
        "rating",
        "age",
    ]
    store = VespaDocumentStore(
        url=os.environ.get("VESPA_URL", "http://localhost"),
        schema="doc",
        namespace="doc",
        content_field="content",
        embedding_field="embedding",
        metadata_fields=_metadata_fields,
    )
    store.delete_all_documents()
    yield store
    store.delete_all_documents()
