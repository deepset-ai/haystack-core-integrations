# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
from unittest.mock import Mock

import pytest

from haystack_integrations.document_stores.vespa import VespaDocumentStore

VESPA_URL = os.environ.get("VESPA_URL", "http://localhost")
METADATA_FIELDS = [
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


@pytest.fixture
def document_store():
    """Real Vespa store used by integration tests.

    Assumes Vespa is reachable at `VESPA_URL` (default `http://localhost`) with the test schema
    deployed — run `docker compose up -d --wait` from `integrations/vespa` to bring it up.
    """
    store = VespaDocumentStore(
        url=VESPA_URL,
        schema="doc",
        namespace="doc",
        content_field="content",
        embedding_field="embedding",
        metadata_fields=METADATA_FIELDS,
    )
    store.delete_all_documents()
    yield store
    store.delete_all_documents()


class DummyResponse:
    """Stand-in for `pyvespa`'s `VespaResponse` used in unit tests."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def is_successful(self):
        return 200 <= self.status_code < 300

    def get_json(self):
        return self._payload


@pytest.fixture
def mock_store():
    """`VespaDocumentStore` with a mocked underlying `pyvespa.Vespa` app."""
    document_store = VespaDocumentStore(
        url="http://localhost",
        schema="docs",
        namespace="docs",
        metadata_fields=["category", "author"],
    )
    document_store._app = Mock()
    return document_store
