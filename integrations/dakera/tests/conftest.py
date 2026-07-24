# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
import uuid

import pytest

from haystack_integrations.document_stores.dakera import DakeraDocumentStore


@pytest.fixture
def document_store(request):
    """
    Provides a DakeraDocumentStore connected to a live server for integration tests.

    Each test gets its own namespace so tests can run in parallel; the namespace is
    populated and torn down within the test. Requires ``DAKERA_URL`` (and usually
    ``DAKERA_API_KEY``) to be set.
    """
    url = os.environ.get("DAKERA_URL", "http://localhost:3000")
    namespace = f"haystack-test-{request.node.name}-{uuid.uuid4().hex[:8]}"
    store = DakeraDocumentStore(url=url, namespace=namespace, dimension=8, metric="cosine")

    yield store

    try:
        client = store._initialize_client()
        client.delete(namespace, delete_all=True)
    except Exception:  # noqa: S110
        # Best-effort cleanup: a namespace that was never created raises, which is fine.
        pass
    finally:
        store.close()
