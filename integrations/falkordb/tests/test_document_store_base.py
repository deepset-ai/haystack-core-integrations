# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

from haystack.testing.document_store import DocumentStoreBaseTests
from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore


@pytest.mark.integration
class TestDocumentStore(DocumentStoreBaseTests):
    """
    Test FalkorDBDocumentStore against the standard Haystack DocumentStore tests.
    """

    @pytest.fixture
    def document_store(self, request):
        host = os.environ.get("FALKORDB_HOST", "localhost")
        port = int(os.environ.get("FALKORDB_PORT", "6379"))

        # Use a unique graph name for each test to avoid interference
        # request.node.name is unique per test
        graph_name = f"test_graph_{request.node.name[:30]}"
        
        store = FalkorDBDocumentStore(
            host=host,
            port=port,
            graph_name=graph_name,
            embedding_dim=768,
            recreate_index=True,
            verify_connectivity=True,
        )
        yield store
        
        # Teardown: delete the graph
        try:
            store._client.delete(graph_name)
        except Exception:
            pass
