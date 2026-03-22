# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.astra import AstraEmbeddingRetriever
from haystack_integrations.document_stores.astra import AstraDocumentStore


@pytest.mark.asyncio
@patch.dict(
    "os.environ",
    {"ASTRA_DB_APPLICATION_TOKEN": "fake-token", "ASTRA_DB_API_ENDPOINT": "http://fake-url.apps.astra.datastax.com"},
)
@patch("haystack_integrations.document_stores.astra.document_store.AstraClient")
async def test_run_async_calls_search_async(*_):
    ds = AstraDocumentStore()
    ds.search_async = AsyncMock(return_value=[])
    retriever = AstraEmbeddingRetriever(ds, filters={"foo": "bar"}, top_k=5)

    result = await retriever.run_async(query_embedding=[0.1, 0.2, 0.3])

    ds.search_async.assert_called_once_with([0.1, 0.2, 0.3], 5, filters={"foo": "bar"})
    assert result == {"documents": []}


@pytest.mark.asyncio
@patch.dict(
    "os.environ",
    {"ASTRA_DB_APPLICATION_TOKEN": "fake-token", "ASTRA_DB_API_ENDPOINT": "http://fake-url.apps.astra.datastax.com"},
)
@patch("haystack_integrations.document_stores.astra.document_store.AstraClient")
async def test_run_async_filter_policy_merge(*_):
    ds = AstraDocumentStore()
    ds.search_async = AsyncMock(return_value=[])
    retriever = AstraEmbeddingRetriever(ds, filters={"foo": "bar"}, top_k=5, filter_policy=FilterPolicy.MERGE)

    await retriever.run_async(query_embedding=[0.1, 0.2, 0.3], filters={"baz": "qux"})

    # apply_filter_policy(MERGE, {"foo": "bar"}, {"baz": "qux"}) returns {"baz": "qux"}
    # because the runtime filters override the init filters for overlapping / non-nested keys
    ds.search_async.assert_called_once_with([0.1, 0.2, 0.3], 5, filters={"baz": "qux"})
