# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.dataclasses import Document

from haystack_integrations.document_stores.supabase import SupabaseGroongaDocumentStore


@pytest.fixture
def mock_supabase_client():
    """Creates a mock Supabase client so we never hit a real database."""
    with patch("haystack_integrations.document_stores.supabase.groonga_document_store.create_client") as mock_create:
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        mock_client.rpc.return_value.execute.return_value = MagicMock(data=[], count=0)

        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.upsert.return_value = mock_table
        mock_table.delete.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.in_.return_value = mock_table
        mock_table.execute.return_value = MagicMock(data=[], count=0)

        yield mock_client


@pytest.fixture
def groonga_store(mock_supabase_client, monkeypatch):  # noqa: ARG001
    """Creates a SupabaseGroongaDocumentStore with mocked client and calls warm_up()."""
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
    store = SupabaseGroongaDocumentStore(
        supabase_url="https://fake-project.supabase.co",
        table_name="test_groonga_documents",
        recreate_table=False,
    )
    store.warm_up()
    return store


class TestDocumentStore:
    def test_init_defaults(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(supabase_url="https://fake-project.supabase.co")
        assert store.table_name == "haystack_groonga_documents"
        assert store.recreate_table is False
        assert store.supabase_url == "https://fake-project.supabase.co"
        assert store._client is None

    def test_init_custom_params(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(
            supabase_url="https://fake-project.supabase.co",
            table_name="my_custom_table",
            recreate_table=True,
        )
        assert store.table_name == "my_custom_table"
        assert store.recreate_table is True
        assert store._client is None

    def test_invalid_table_name_raises(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        with pytest.raises(ValueError, match="Invalid table_name"):
            SupabaseGroongaDocumentStore(
                supabase_url="https://fake-project.supabase.co",
                table_name="bad-name; DROP TABLE users;",
            )

    def test_table_name_with_numbers_allowed(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(
            supabase_url="https://fake-project.supabase.co",
            table_name="my_table_123",
        )
        assert store.table_name == "my_table_123"

    def test_warm_up_initializes_client(self, mock_supabase_client, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(supabase_url="https://fake-project.supabase.co")
        assert store._client is None
        store.warm_up()
        assert store._client is not None

    def test_write_documents_empty(self, groonga_store):
        assert groonga_store.write_documents([]) == 0

    def test_delete_documents_empty(self, groonga_store, mock_supabase_client):
        groonga_store.delete_documents([])
        mock_supabase_client.table.return_value.delete.assert_not_called()

    def test_to_dict(self, groonga_store):
        result = groonga_store.to_dict()
        assert result["type"] == (
            "haystack_integrations.document_stores.supabase.groonga_document_store.SupabaseGroongaDocumentStore"
        )
        assert result["init_parameters"]["table_name"] == "test_groonga_documents"
        assert result["init_parameters"]["supabase_url"] == "https://fake-project.supabase.co"
        assert result["init_parameters"]["recreate_table"] is False

    def test_from_dict(self, mock_supabase_client, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        data = {
            "type": (
                "haystack_integrations.document_stores.supabase.groonga_document_store.SupabaseGroongaDocumentStore"
            ),
            "init_parameters": {
                "supabase_url": "https://fake-project.supabase.co",
                "supabase_key": {
                    "type": "env_var",
                    "env_vars": ["SUPABASE_SERVICE_KEY"],
                    "strict": True,
                },
                "table_name": "test_groonga_documents",
                "recreate_table": False,
            },
        }
        store = SupabaseGroongaDocumentStore.from_dict(data)
        assert store.table_name == "test_groonga_documents"
        assert store.supabase_url == "https://fake-project.supabase.co"
        assert store._client is None


@pytest.fixture
def mock_async_supabase_client():
    """Creates a mock async Supabase client for async tests."""
    with patch(
        "haystack_integrations.document_stores.supabase.groonga_document_store.acreate_client",
        new_callable=AsyncMock,
    ) as mock_acreate:
        mock_client = MagicMock()
        mock_acreate.return_value = mock_client

        mock_client.rpc.return_value.execute = AsyncMock(return_value=MagicMock(data=[], count=0))

        mock_table = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.select.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.upsert.return_value = mock_table
        mock_table.delete.return_value = mock_table
        mock_table.eq.return_value = mock_table
        mock_table.in_.return_value = mock_table
        mock_table.neq.return_value = mock_table
        mock_table.execute = AsyncMock(return_value=MagicMock(data=[], count=0))

        yield mock_client


@pytest.fixture
def groonga_store_async(mock_async_supabase_client, monkeypatch):  # noqa: ARG001
    """Returns a store whose async client will be initialized lazily on first use."""
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
    return SupabaseGroongaDocumentStore(
        supabase_url="https://fake-project.supabase.co",
        table_name="test_groonga_documents",
        recreate_table=False,
    )


@pytest.mark.asyncio
class TestDocumentStoreAsync:
    async def test_lazy_async_client_initialization(self, mock_async_supabase_client, monkeypatch):  # noqa: ARG002
        """Async client must be None at construction and set after the first async call."""
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(supabase_url="https://fake-project.supabase.co")
        assert store._async_client is None
        await store.count_documents_async()
        assert store._async_client is not None

    async def test_write_documents_async_empty(self, groonga_store_async):
        written = await groonga_store_async.write_documents_async([])
        assert written == 0

    async def test_delete_documents_async_empty(self, groonga_store_async, mock_async_supabase_client):
        await groonga_store_async.delete_documents_async([])
        mock_async_supabase_client.table.return_value.delete.assert_not_called()

    async def test_async_client_initialized_only_once(self, mock_async_supabase_client, monkeypatch):  # noqa: ARG002
        """_initialize_async_client must not replace the client on subsequent calls."""
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "fake-test-key")
        store = SupabaseGroongaDocumentStore(supabase_url="https://fake-project.supabase.co")
        await store.count_documents_async()
        first_client = store._async_client
        await store.count_documents_async()
        assert store._async_client is first_client


# ---------------------------------------------------------------------------
# Unit tests for the in-memory filter helpers (no live service required)
# ---------------------------------------------------------------------------

_filter = SupabaseGroongaDocumentStore._filter_documents_in_memory


def _docs():
    return [
        Document(id="1", content="alpha", meta={"lang": "en", "score": 1, "active": True}),
        Document(id="2", content="beta", meta={"lang": "fr", "score": 2, "active": False}),
        Document(id="3", content="gamma", meta={"lang": "en", "score": 3}),
    ]


class TestFilterDocumentsInMemory:
    # --- flat (simple) condition — the previously broken path -----------------

    def test_flat_condition_eq(self):
        result = _filter(_docs(), {"field": "meta.lang", "operator": "==", "value": "en"})
        assert [d.id for d in result] == ["1", "3"]

    def test_flat_condition_neq(self):
        result = _filter(_docs(), {"field": "meta.lang", "operator": "!=", "value": "en"})
        assert [d.id for d in result] == ["2"]

    def test_flat_condition_in(self):
        result = _filter(_docs(), {"field": "meta.lang", "operator": "in", "value": ["en", "de"]})
        assert [d.id for d in result] == ["1", "3"]

    def test_flat_condition_not_in(self):
        result = _filter(_docs(), {"field": "meta.lang", "operator": "not in", "value": ["en"]})
        assert [d.id for d in result] == ["2"]

    # --- comparison operators -------------------------------------------------

    def test_gt(self):
        result = _filter(_docs(), {"field": "meta.score", "operator": ">", "value": 1})
        assert [d.id for d in result] == ["2", "3"]

    def test_gte(self):
        result = _filter(_docs(), {"field": "meta.score", "operator": ">=", "value": 2})
        assert [d.id for d in result] == ["2", "3"]

    def test_lt(self):
        result = _filter(_docs(), {"field": "meta.score", "operator": "<", "value": 3})
        assert [d.id for d in result] == ["1", "2"]

    def test_lte(self):
        result = _filter(_docs(), {"field": "meta.score", "operator": "<=", "value": 2})
        assert [d.id for d in result] == ["1", "2"]

    def test_gt_excludes_missing_field(self):
        # doc "3" has no "score" key — should be excluded since None > value is False
        docs = [
            Document(id="a", meta={"score": 5}),
            Document(id="b", meta={}),
        ]
        result = _filter(docs, {"field": "meta.score", "operator": ">", "value": 0})
        assert [d.id for d in result] == ["a"]

    # --- logical operators ----------------------------------------------------

    def test_and(self):
        result = _filter(
            _docs(),
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.lang", "operator": "==", "value": "en"},
                    {"field": "meta.score", "operator": ">", "value": 1},
                ],
            },
        )
        assert [d.id for d in result] == ["3"]

    def test_or(self):
        result = _filter(
            _docs(),
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.lang", "operator": "==", "value": "fr"},
                    {"field": "meta.score", "operator": "==", "value": 3},
                ],
            },
        )
        assert [d.id for d in result] == ["2", "3"]

    def test_not(self):
        result = _filter(
            _docs(),
            {"operator": "NOT", "conditions": [{"field": "meta.lang", "operator": "==", "value": "en"}]},
        )
        assert [d.id for d in result] == ["2"]

    def test_nested_and_inside_or(self):
        # (lang==en AND score==1) OR lang==fr
        result = _filter(
            _docs(),
            {
                "operator": "OR",
                "conditions": [
                    {
                        "operator": "AND",
                        "conditions": [
                            {"field": "meta.lang", "operator": "==", "value": "en"},
                            {"field": "meta.score", "operator": "==", "value": 1},
                        ],
                    },
                    {"field": "meta.lang", "operator": "==", "value": "fr"},
                ],
            },
        )
        assert [d.id for d in result] == ["1", "2"]

    def test_empty_filters_returns_all(self):
        docs = _docs()
        assert _filter(docs, {}) == docs
