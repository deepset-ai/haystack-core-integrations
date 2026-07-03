# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Db2EmbeddingRetriever using live DB2 instance."""

import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.components.retrievers.ibm_db import Db2EmbeddingRetriever
from haystack_integrations.document_stores.ibm_db import Db2DocumentStore


@pytest.fixture
def document_store(connection_config, request):
    """Create a fresh document store for each test."""
    # Use test name to create unique table name per test
    table_name = f"test_retriever_emb_{request.node.name}_{sys.version_info.major}_{sys.version_info.minor}"

    store = Db2DocumentStore(
        connection_config=connection_config,
        table_name=table_name,
        embedding_dim=4,  # Small dimension for testing
        distance_metric="COSINE",
        recreate_table=True,
    )
    yield store
    # Cleanup after test
    try:
        conn = store._get_connection()
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE {store.table_name}")
            conn.commit()
    except Exception:
        pass


@pytest.fixture
def sample_documents():
    """Create sample documents with embeddings for testing."""
    return [
        Document(
            id="doc1",
            content="Python programming language",
            meta={"category": "programming", "language": "python"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        ),
        Document(
            id="doc2",
            content="Java development",
            meta={"category": "programming", "language": "java"},
            embedding=[0.5, 0.6, 0.7, 0.8],
        ),
        Document(
            id="doc3",
            content="Data science with Python",
            meta={"category": "data-science", "language": "python"},
            embedding=[0.15, 0.25, 0.35, 0.45],
        ),
    ]


class TestDb2EmbeddingRetrieverUnit:
    """Unit tests for Db2EmbeddingRetriever that don't require a database."""

    def test_invalid_document_store_raises_type_error(self):
        """Test that invalid document store raises TypeError."""
        with pytest.raises(TypeError, match="must be an instance of Db2DocumentStore"):
            Db2EmbeddingRetriever(document_store="not_a_store")

    def test_init_with_filter_policy_string(self):
        """Test initialization with filter_policy as string."""
        mock_store = Mock(spec=Db2DocumentStore)

        retriever_replace = Db2EmbeddingRetriever(document_store=mock_store, filter_policy="replace")
        assert retriever_replace.filter_policy == FilterPolicy.REPLACE

        retriever_merge = Db2EmbeddingRetriever(document_store=mock_store, filter_policy="merge")
        assert retriever_merge.filter_policy == FilterPolicy.MERGE

    def test_init_with_filter_policy_enum(self):
        """Test initialization with FilterPolicy enum directly."""
        mock_store = Mock(spec=Db2DocumentStore)

        retriever_replace = Db2EmbeddingRetriever(document_store=mock_store, filter_policy=FilterPolicy.REPLACE)
        assert retriever_replace.filter_policy == FilterPolicy.REPLACE

        retriever_merge = Db2EmbeddingRetriever(document_store=mock_store, filter_policy=FilterPolicy.MERGE)
        assert retriever_merge.filter_policy == FilterPolicy.MERGE

    def test_to_dict(self):
        """Test serialization to dictionary."""
        mock_store = Mock(spec=Db2DocumentStore)
        retriever = Db2EmbeddingRetriever(
            document_store=mock_store,
            top_k=7,
            filters={"operator": "==", "field": "meta.x", "value": "y"},
        )
        d = retriever.to_dict()

        assert d["init_parameters"]["top_k"] == 7
        assert d["init_parameters"]["filters"] == {"operator": "==", "field": "meta.x", "value": "y"}
        assert d["init_parameters"]["filter_policy"] == "replace"
        assert "document_store" in d["init_parameters"]

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        mock_store = Mock(spec=Db2DocumentStore)
        data = {
            "type": "haystack_integrations.components.retrievers.ibm_db.embedding_retriever.Db2EmbeddingRetriever",
            "init_parameters": {
                "document_store": {"type": "x", "init_parameters": {}},
                "filters": {"operator": "==", "field": "meta.x", "value": "y"},
                "top_k": 7,
                "filter_policy": "replace",
            },
        }
        with patch.object(Db2DocumentStore, "from_dict", return_value=mock_store):
            restored = Db2EmbeddingRetriever.from_dict(data)

        assert restored.top_k == 7
        assert restored.filters == {"operator": "==", "field": "meta.x", "value": "y"}
        assert restored.filter_policy == FilterPolicy.REPLACE
        assert restored.document_store is mock_store


class TestDb2EmbeddingRetrieverRun:
    """Unit tests for run/run_async using a mocked document store (no database)."""

    def test_run_delegates_to_embedding_retrieval(self):
        expected = [Document(id="1", content="a"), Document(id="2", content="b")]
        mock_store = Mock(spec=Db2DocumentStore)
        mock_store._embedding_retrieval.return_value = expected

        retriever = Db2EmbeddingRetriever(document_store=mock_store, top_k=4)
        result = retriever.run(query_embedding=[0.1, 0.2, 0.3, 0.4])

        assert result == {"documents": expected}
        mock_store._embedding_retrieval.assert_called_once()
        _, kwargs = mock_store._embedding_retrieval.call_args
        assert kwargs["top_k"] == 4
        assert kwargs["filters"] == {}

    def test_run_top_k_override_and_runtime_filters(self):
        mock_store = Mock(spec=Db2DocumentStore)
        mock_store._embedding_retrieval.return_value = []

        runtime_filters = {"operator": "==", "field": "meta.x", "value": "y"}
        retriever = Db2EmbeddingRetriever(document_store=mock_store, top_k=10)
        retriever.run(query_embedding=[0.1, 0.2], filters=runtime_filters, top_k=2)

        _, kwargs = mock_store._embedding_retrieval.call_args
        assert kwargs["top_k"] == 2
        # REPLACE policy => runtime filters win
        assert kwargs["filters"] == apply_filter_policy(FilterPolicy.REPLACE, {}, runtime_filters)

    @pytest.mark.asyncio
    async def test_run_async_delegates(self):
        expected = [Document(id="1", content="a")]
        mock_store = Mock(spec=Db2DocumentStore)
        mock_store._embedding_retrieval_async = AsyncMock(return_value=expected)

        retriever = Db2EmbeddingRetriever(document_store=mock_store, top_k=3)
        result = await retriever.run_async(query_embedding=[0.1, 0.2, 0.3, 0.4])
        assert result == {"documents": expected}
        mock_store._embedding_retrieval_async.assert_awaited_once()
