# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for ArcadeDB integration tests."""

import os

import pytest

from haystack_integrations.document_stores.arcadedb import ArcadeDBDocumentStore

ARCADEDB_URL = os.getenv("ARCADEDB_URL", "http://localhost:2480")


@pytest.fixture
def document_store():
    """
    ArcadeDB document store instance for integration tests.

    """
    store = ArcadeDBDocumentStore(
        url=ARCADEDB_URL,
        database="haystack_test",
        embedding_dimension=768,
        recreate_type=True,
    )
    return store
