# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os

import pytest
from haystack.utils import Secret

from haystack_integrations.document_stores.arangodb import ArangoDocumentStore


@pytest.fixture
def document_store(request):
    host = os.environ.get("ARANGO_HOST")
    password = os.environ.get("ARANGO_PASSWORD")
    if not host or not password:
        pytest.skip("Set ARANGO_HOST and ARANGO_PASSWORD to run integration tests.")

    store = ArangoDocumentStore(
        host=host,
        database="haystack_test",
        username=Secret.from_env_var("ARANGO_USERNAME", strict=False),
        password=Secret.from_env_var("ARANGO_PASSWORD"),
        collection_name=f"test_{request.node.name}",
        embedding_dimension=768,
        recreate_collection=True,
    )
    yield store
    with contextlib.suppress(Exception):
        store._ensure_connected()
        if store._db and store._db.has_collection(store.collection_name):
            store._db.delete_collection(store.collection_name)
