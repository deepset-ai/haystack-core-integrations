# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import random
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from azure.core.credentials import TokenCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes.models import (
    CustomAnalyzer,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchResourceEncryptionKey,
    SimpleField,
)
from haystack.dataclasses.document import Document
from haystack.errors import FilterError
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountDocumentsTest,
    CountUniqueMetadataByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    FilterDocumentsTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
    UpdateByFilterTest,
    WriteDocumentsTest,
)
from haystack.utils.auth import EnvVarSecret, Secret

from haystack_integrations.document_stores.azure_ai_search import (
    DEFAULT_VECTOR_SEARCH,
    AzureAISearchDocumentStore,
)


def test_to_dict(monkeypatch):
    monkeypatch.setenv("AZURE_AI_SEARCH_API_KEY", "test-api-key")
    monkeypatch.setenv("AZURE_AI_SEARCH_ENDPOINT", "test-endpoint")
    document_store = AzureAISearchDocumentStore()
    res = document_store.to_dict()
    assert res == {
        "type": "haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore",
        "init_parameters": {
            "azure_endpoint": {"env_vars": ["AZURE_AI_SEARCH_ENDPOINT"], "strict": True, "type": "env_var"},
            "api_key": {"env_vars": ["AZURE_AI_SEARCH_API_KEY"], "strict": False, "type": "env_var"},
            "index_name": "default",
            "embedding_dimension": 768,
            "metadata_fields": {},
            "vector_search_configuration": {
                "profiles": [{"name": "default-vector-config", "algorithm": "cosine-algorithm-config"}],
                "algorithms": [
                    {
                        "name": "cosine-algorithm-config",
                        "hnswParameters": {"metric": "cosine"},
                        "kind": "hnsw",
                    }
                ],
            },
        },
    }


def test_to_dict_with_params(monkeypatch):
    monkeypatch.setenv("AZURE_AI_SEARCH_API_KEY", "test-api-key")
    monkeypatch.setenv("AZURE_AI_SEARCH_ENDPOINT", "test-endpoint")
    encryption_key = SearchResourceEncryptionKey(
        key_name="my-key",
        key_version="my-version",
        vault_uri="my-uri",
    )
    analyzer = CustomAnalyzer(
        name="url-analyze",
        tokenizer_name="uax_url_email",
        token_filters=["lowercase"],  # Using token filter name directly as string
    )
    document_store = AzureAISearchDocumentStore(
        index_name="my_index",
        embedding_dimension=15,
        metadata_fields={
            "Title": SearchField(name="Title", type="Edm.String", searchable=True, filterable=True),
            "Pages": int,
        },
        encryption_key=encryption_key,
        analyzers=[analyzer],
    )

    res = document_store.to_dict()
    assert res == {
        "type": "haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore",
        "init_parameters": {
            "azure_endpoint": {"env_vars": ["AZURE_AI_SEARCH_ENDPOINT"], "strict": True, "type": "env_var"},
            "api_key": {"env_vars": ["AZURE_AI_SEARCH_API_KEY"], "strict": False, "type": "env_var"},
            "index_name": "my_index",
            "embedding_dimension": 15,
            "metadata_fields": {
                "Title": SearchField(name="Title", type="Edm.String", searchable=True, filterable=True).as_dict(),
                "Pages": SimpleField(name="Pages", type="Edm.Int32", filterable=True).as_dict(),
            },
            "encryption_key": {
                "keyVaultKeyName": "my-key",
                "keyVaultKeyVersion": "my-version",
                "keyVaultUri": "my-uri",
            },
            "analyzers": [
                {
                    "name": "url-analyze",
                    "tokenizer": "uax_url_email",
                    "tokenFilters": ["lowercase"],
                    "@odata.type": "#Microsoft.Azure.Search.CustomAnalyzer",
                }
            ],
            "vector_search_configuration": {
                "profiles": [{"name": "default-vector-config", "algorithm": "cosine-algorithm-config"}],
                "algorithms": [
                    {
                        "name": "cosine-algorithm-config",
                        "hnswParameters": {"metric": "cosine"},
                        "kind": "hnsw",
                    }
                ],
            },
        },
    }


def test_to_dict_emits_warning_when_token_credential_is_used(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("AZURE_AI_SEARCH_API_KEY", "test-api-key")
    monkeypatch.setenv("AZURE_AI_SEARCH_ENDPOINT", "test-endpoint")

    mock_token_credential = Mock(spec=TokenCredential)
    document_store = AzureAISearchDocumentStore(azure_token_credential=mock_token_credential)

    with caplog.at_level(logging.WARNING):
        result = document_store.to_dict()

    assert "`azure_token_credential`, which cannot be serialized." in caplog.text

    # token credential should not appear in the serialized output
    assert "azure_token_credential" not in result["init_parameters"]


def test_from_dict(monkeypatch):
    monkeypatch.setenv("AZURE_AI_SEARCH_API_KEY", "test-api-key")
    monkeypatch.setenv("AZURE_AI_SEARCH_ENDPOINT", "test-endpoint")

    data = {
        "type": "haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore",
        "init_parameters": {
            "azure_endpoint": {"env_vars": ["AZURE_AI_SEARCH_ENDPOINT"], "strict": True, "type": "env_var"},
            "api_key": {"env_vars": ["AZURE_AI_SEARCH_API_KEY"], "strict": False, "type": "env_var"},
            "embedding_dimension": 768,
            "index_name": "default",
            "metadata_fields": None,
            "vector_search_configuration": DEFAULT_VECTOR_SEARCH,
        },
    }
    document_store = AzureAISearchDocumentStore.from_dict(data)
    assert isinstance(document_store._api_key, EnvVarSecret)
    assert isinstance(document_store._azure_endpoint, EnvVarSecret)
    assert document_store._index_name == "default"
    assert document_store._embedding_dimension == 768
    assert document_store._metadata_fields == {}
    assert document_store._vector_search_configuration == DEFAULT_VECTOR_SEARCH


def test_from_dict_with_params(monkeypatch):
    monkeypatch.setenv("AZURE_AI_SEARCH_API_KEY", "test-api-key")
    monkeypatch.setenv("AZURE_AI_SEARCH_ENDPOINT", "test-endpoint")
    encryption_key = SearchResourceEncryptionKey(
        key_name="my-key",
        key_version="my-version",
        vault_uri="my-uri",
    )

    data = {
        "type": "haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore",
        "init_parameters": {
            "azure_endpoint": {"env_vars": ["AZURE_AI_SEARCH_ENDPOINT"], "strict": True, "type": "env_var"},
            "api_key": {"env_vars": ["AZURE_AI_SEARCH_API_KEY"], "strict": False, "type": "env_var"},
            "index_name": "my_index",
            "embedding_dimension": 15,
            "metadata_fields": {
                "Title": SearchField(name="Title", type="Edm.String", filterable=True).as_dict(),
                "Pages": SimpleField(name="Pages", type="Edm.Int32", filterable=True).as_dict(),
            },
            "encryption_key": {
                "keyVaultKeyName": "my-key",
                "keyVaultKeyVersion": "my-version",
                "keyVaultUri": "my-uri",
            },
            "analyzers": [
                {
                    "name": "url-analyze",
                    "@odata.type": "#Microsoft.Azure.Search.CustomAnalyzer",
                    "tokenizer": "uax_url_email",
                    "tokenFilters": ["lowercase"],
                }
            ],
            "vector_search_configuration": {
                "profiles": [{"name": "default-vector-config", "algorithm": "cosine-algorithm-config"}],
                "algorithms": [
                    {
                        "name": "cosine-algorithm-config",
                        "kind": "hnsw",
                        "hnswParameters": {"metric": "cosine"},
                    }
                ],
            },
        },
    }
    document_store = AzureAISearchDocumentStore.from_dict(data)
    assert isinstance(document_store._api_key, EnvVarSecret)
    assert isinstance(document_store._azure_endpoint, EnvVarSecret)
    assert document_store._index_name == "my_index"
    assert document_store._embedding_dimension == 15
    assert document_store._metadata_fields == {
        "Title": SearchField(name="Title", type="Edm.String", filterable=True),
        "Pages": SimpleField(name="Pages", type="Edm.Int32", filterable=True),
    }
    assert document_store._index_creation_kwargs["encryption_key"] == encryption_key
    assert document_store._index_creation_kwargs["analyzers"][0].name == "url-analyze"
    assert document_store._index_creation_kwargs["analyzers"][0].token_filters == ["lowercase"]
    assert "CustomAnalyzer" in document_store._index_creation_kwargs["analyzers"][0].odata_type
    assert document_store._vector_search_configuration.as_dict() == DEFAULT_VECTOR_SEARCH.as_dict()


@patch("haystack_integrations.document_stores.azure_ai_search.document_store.AzureAISearchDocumentStore")
def test_init_is_lazy(_mock_azure_search_client):
    AzureAISearchDocumentStore(azure_endpoint=Secret.from_token("test_endpoint"))
    _mock_azure_search_client.assert_not_called()


def test_init():
    document_store = AzureAISearchDocumentStore(
        api_key=Secret.from_token("fake-api-key"),
        azure_endpoint=Secret.from_token("fake_endpoint"),
        index_name="my_index",
        embedding_dimension=15,
        metadata_fields={"Title": str, "Pages": int},
    )

    assert document_store._index_name == "my_index"
    assert document_store._embedding_dimension == 15
    assert document_store._metadata_fields == {
        "Title": SimpleField(name="Title", type="Edm.String", filterable=True),
        "Pages": SimpleField(name="Pages", type="Edm.Int32", filterable=True),
    }
    assert document_store._vector_search_configuration == DEFAULT_VECTOR_SEARCH


def test_token_credential_takes_priority_over_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_AI_SEARCH_API_KEY", "test-api-key")
    monkeypatch.setenv("AZURE_AI_SEARCH_ENDPOINT", "test-endpoint")

    mock_token_credential = Mock(spec=TokenCredential)
    document_store = AzureAISearchDocumentStore(azure_token_credential=mock_token_credential)

    with patch(
        "haystack_integrations.document_stores.azure_ai_search.document_store.SearchIndexClient"
    ) as mock_index_client_cls:
        mock_index_client = Mock()
        mock_index_client.list_index_names.return_value = ["default"]
        mock_index_client.get_index.return_value = Mock(fields=[])
        mock_index_client.get_search_client.return_value = Mock()
        mock_index_client_cls.return_value = mock_index_client

        _ = document_store.client

        _, kwargs = mock_index_client_cls.call_args
        assert kwargs["credential"] is mock_token_credential


def _build_mock_document_store_with_schema(index_fields):
    store = AzureAISearchDocumentStore(
        api_key=Secret.from_token("fake-api-key"),
        azure_endpoint=Secret.from_token("fake-endpoint"),
        index_name="test-index",
    )
    search_client = Mock()
    index_client = Mock()
    index_client.list_index_names.return_value = ["test-index"]
    index_client.get_index.return_value = Mock(fields=index_fields)
    index_client.get_search_client.return_value = search_client
    store._index_client = index_client
    return store, search_client, index_client


def test_get_metadata_field_unique_values():
    index_fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True),
    ]
    document_store, search_client, _ = _build_mock_document_store_with_schema(index_fields)
    search_client.search.return_value = [
        {"category": "news"},
        {"category": "docs"},
        {"category": "api"},
        {"category": "news"},
    ]

    values, total_count = document_store.get_metadata_field_unique_values(
        metadata_field="meta.category", search_term="d", from_=0, size=10
    )

    assert values == ["docs"]
    assert total_count == 1


def test_query_sql_raises_not_implemented():
    document_store = AzureAISearchDocumentStore(
        api_key=Secret.from_token("fake-api-key"),
        azure_endpoint=Secret.from_token("fake-endpoint"),
        index_name="test-index",
    )

    with pytest.raises(NotImplementedError, match="does not support SQL queries"):
        document_store.query_sql("SELECT * FROM test-index")


@pytest.mark.parametrize(
    "metadata_fields, expected_error_match",
    [
        (
            {"Title": SearchField(name="mismatched", type="Edm.String", filterable=True)},
            "Name of SearchField",
        ),
        ({"Pages": object}, "Unsupported field type"),
    ],
)
def test_normalize_metadata_index_fields_raises(metadata_fields, expected_error_match):
    with pytest.raises(ValueError, match=expected_error_match):
        AzureAISearchDocumentStore._normalize_metadata_index_fields(metadata_fields)


def test_normalize_metadata_index_fields_skips_non_alpha_keys(caplog):
    with caplog.at_level(logging.WARNING):
        normalized = AzureAISearchDocumentStore._normalize_metadata_index_fields({"1invalid": str, "valid": int})
    assert "valid" in normalized
    assert "1invalid" not in normalized
    assert "Invalid key" in caplog.text


def test_normalize_metadata_index_fields_returns_empty_for_none():
    assert AzureAISearchDocumentStore._normalize_metadata_index_fields(None) == {}


@pytest.mark.parametrize(
    "method, kwargs, expected_match",
    [
        ("_bm25_retrieval", {"query": None}, "query must not be None"),
        ("_hybrid_retrieval", {"query": None, "query_embedding": [0.1]}, "query must not be None"),
        ("_hybrid_retrieval", {"query": "q", "query_embedding": []}, "query_embedding must be a non-empty"),
        ("_embedding_retrieval", {"query_embedding": []}, "query_embedding must be a non-empty"),
    ],
)
def test_internal_retrieval_validates_inputs(method, kwargs, expected_match):
    document_store = AzureAISearchDocumentStore(
        api_key=Secret.from_token("fake-api-key"),
        azure_endpoint=Secret.from_token("fake-endpoint"),
        index_name="test-index",
    )
    with pytest.raises(ValueError, match=expected_match):
        getattr(document_store, method)(**kwargs)


def test_collect_unique_values_combines_lists_and_scalars():
    docs = [
        {"tags": ["a", "b"]},
        {"tags": "c"},
        {"tags": None},
        {"tags": ["a", "d"]},
    ]
    assert AzureAISearchDocumentStore._collect_unique_values(docs, "tags") == {"a", "b", "c", "d"}


@pytest.mark.parametrize(
    "docs, expected",
    [
        ([], {"min": None, "max": None}),
        ([{"x": None}, {"x": [1, 2]}], {"min": None, "max": None}),
        ([{"x": 3}, {"x": 1}, {"x": 2}], {"min": 1, "max": 3}),
    ],
)
def test_get_min_max_from_documents(docs, expected):
    assert AzureAISearchDocumentStore._get_min_max_from_documents(docs, "x") == expected


@pytest.mark.parametrize(
    "field, expected_type",
    [
        (SimpleField(name="cat", type=SearchFieldDataType.String, filterable=True), "keyword"),
        (SearchableField(name="content", type=SearchFieldDataType.String), "text"),
        (SearchableField(name="title", type=SearchFieldDataType.String), "text"),
        (SimpleField(name="year", type=SearchFieldDataType.Int32, filterable=True), "long"),
        (SimpleField(name="rating", type=SearchFieldDataType.Double, filterable=True), "double"),
        (
            SearchField(
                name="tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
            ),
            "keyword",
        ),
        (SimpleField(name="when", type=SearchFieldDataType.DateTimeOffset, filterable=True), "date"),
    ],
)
def test_map_azure_field_type_variants(field, expected_type):
    assert AzureAISearchDocumentStore._map_azure_field_type(field) == expected_type


def test_map_azure_field_type_without_type_attribute():
    field = Mock(spec=[])
    field.name = "custom"
    assert AzureAISearchDocumentStore._map_azure_field_type(field) == "keyword"


def test_index_exists_raises_without_index_name():
    document_store = AzureAISearchDocumentStore(
        api_key=Secret.from_token("fake-api-key"),
        azure_endpoint=Secret.from_token("fake-endpoint"),
        index_name="test-index",
    )
    document_store._index_client = Mock()
    with pytest.raises(ValueError, match="Index name is required"):
        document_store._index_exists(None)


def test_get_raw_documents_by_id_skips_not_found(caplog):
    store, search_client, _ = _build_mock_document_store_with_schema(
        [SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True)]
    )
    search_client.get_document.side_effect = [
        {"id": "1", "content": "c1"},
        ResourceNotFoundError("not found"),
    ]
    with caplog.at_level(logging.WARNING):
        result = store._get_raw_documents_by_id(["1", "missing"])
    assert result == [{"id": "1", "content": "c1"}]
    assert "missing" in caplog.text


def _assert_documents_are_equal(received: list[Document], expected: list[Document]):
    """
    Assert that two lists of Documents are equal.

    This is used in every test, if a Document Store implementation has a different behaviour
    it should override this method. This can happen for example when the Document Store sets
    a score to returned Documents. Since we can't know what the score will be, we can't compare
    the Documents reliably.
    """
    sorted_received = sorted(received, key=lambda doc: doc.id)
    sorted_expected = sorted(expected, key=lambda doc: doc.id)
    assert len(sorted_received) == len(sorted_expected)

    for received_doc, expected_doc in zip(sorted_received, sorted_expected, strict=True):
        # Compare all attributes except score
        assert received_doc.id == expected_doc.id
        assert received_doc.content == expected_doc.content
        assert received_doc.embedding == expected_doc.embedding
        assert received_doc.meta == expected_doc.meta


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("AZURE_AI_SEARCH_ENDPOINT", None) and not os.environ.get("AZURE_AI_SEARCH_API_KEY", None),
    reason="Missing AZURE_AI_SEARCH_ENDPOINT or AZURE_AI_SEARCH_API_KEY.",
)
class TestDocumentStore(
    CountDocumentsTest,
    DeleteDocumentsTest,
    DeleteAllTest,
    DeleteByFilterTest,
    FilterableDocsFixtureMixin,
    WriteDocumentsTest,
    UpdateByFilterTest,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
):
    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        _assert_documents_are_equal(received, expected)

    def test_write_documents(self, document_store: AzureAISearchDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1

    # Parametrize the test with metadata fields
    @pytest.mark.parametrize(
        "document_store",
        [
            {"metadata_fields": {"author": str, "publication_year": int, "rating": float}},
        ],
        indirect=True,
    )
    def test_write_documents_with_meta(self, document_store: AzureAISearchDocumentStore):
        docs = [
            Document(
                id="1",
                meta={"author": "Tom", "publication_year": 2021, "rating": 4.5},
                content="This is a test document.",
            )
        ]
        document_store.write_documents(docs)
        doc = document_store.get_documents_by_id(["1"])
        assert doc[0] == docs[0]

    @pytest.mark.skip(reason="Azure AI search index overwrites duplicate documents by default")
    def test_write_documents_duplicate_fail(self, document_store: AzureAISearchDocumentStore): ...

    @pytest.mark.skip(reason="Azure AI search index overwrites duplicate documents by default")
    def test_write_documents_duplicate_skip(self, document_store: AzureAISearchDocumentStore): ...

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str}}],
        indirect=True,
    )
    def test_delete_by_filter(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with category metadata field."""
        DeleteByFilterTest.test_delete_by_filter(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str}}],
        indirect=True,
    )
    def test_delete_by_filter_no_matches(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with category metadata field."""
        DeleteByFilterTest.test_delete_by_filter_no_matches(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str, "year": int, "status": str}}],
        indirect=True,
    )
    def test_delete_by_filter_advanced_filters(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with category, year, status metadata fields."""
        DeleteByFilterTest.test_delete_by_filter_advanced_filters(document_store)

    # Metadata fields required by haystack UpdateByFilterTest filterable_docs (chapter, name, page, number, date, etc.)
    _FILTERABLE_DOCS_METADATA = {  # noqa: RUF012
        "name": str,
        "page": str,
        "chapter": str,
        "number": int,
        "date": str,
        "no_embedding": bool,
        "updated": bool,
        "extra_field": str,
    }

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": _FILTERABLE_DOCS_METADATA}],
        indirect=True,
    )
    def test_update_by_filter(self, document_store: AzureAISearchDocumentStore, filterable_docs):
        """Override to use a document_store with metadata fields for filterable_docs."""
        UpdateByFilterTest.test_update_by_filter(document_store, filterable_docs)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": _FILTERABLE_DOCS_METADATA}],
        indirect=True,
    )
    def test_update_by_filter_no_matches(self, document_store: AzureAISearchDocumentStore, filterable_docs):
        """Override to use a document_store with metadata fields for filterable_docs."""
        UpdateByFilterTest.test_update_by_filter_no_matches(document_store, filterable_docs)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": _FILTERABLE_DOCS_METADATA}],
        indirect=True,
    )
    def test_update_by_filter_multiple_fields(self, document_store: AzureAISearchDocumentStore, filterable_docs):
        """Override to use a document_store with metadata fields for filterable_docs."""
        UpdateByFilterTest.test_update_by_filter_multiple_fields(document_store, filterable_docs)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str, "year": int, "status": str, "featured": bool}}],
        indirect=True,
    )
    def test_update_by_filter_advanced_filters(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with category, year, status, featured metadata fields."""
        UpdateByFilterTest.test_update_by_filter_advanced_filters(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str, "status": str}}],
        indirect=True,
    )
    def test_update_by_filter_invalid_field(self, document_store: AzureAISearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A", "status": "draft"}),
        ]
        document_store.write_documents(docs)

        # Try to update a field that doesn't exist in the schema
        with pytest.raises(ValueError) as exc_info:
            document_store.update_by_filter(
                filters={"field": "meta.category", "operator": "==", "value": "A"},
                fields={"nonexistent_field": "value"},
            )
        assert "nonexistent_field" in str(exc_info.value)
        assert "not defined in index schema" in str(exc_info.value)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str, "status": str}}],
        indirect=True,
    )
    def test_count_documents_by_filter_simple(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with required metadata fields."""
        CountDocumentsByFilterTest.test_count_documents_by_filter_simple(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str, "status": str}}],
        indirect=True,
    )
    def test_count_documents_by_filter_compound(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with required metadata fields."""
        CountDocumentsByFilterTest.test_count_documents_by_filter_compound(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str}}],
        indirect=True,
    )
    def test_count_documents_by_filter_no_matches(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with required metadata fields."""
        CountDocumentsByFilterTest.test_count_documents_by_filter_no_matches(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str}}],
        indirect=True,
    )
    def test_count_documents_by_filter_empty_collection(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with required metadata fields."""
        CountDocumentsByFilterTest.test_count_documents_by_filter_empty_collection(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str, "status": str, "priority": int}}],
        indirect=True,
    )
    def test_count_unique_metadata_by_filter_all_documents(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with required metadata fields."""
        CountUniqueMetadataByFilterTest.test_count_unique_metadata_by_filter_all_documents(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str, "status": str, "priority": int}}],
        indirect=True,
    )
    def test_count_unique_metadata_by_filter_with_filter(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with required metadata fields."""
        CountUniqueMetadataByFilterTest.test_count_unique_metadata_by_filter_with_filter(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str, "year": int}}],
        indirect=True,
    )
    def test_count_unique_metadata_by_filter_with_multiple_filters(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with required metadata fields."""
        CountUniqueMetadataByFilterTest.test_count_unique_metadata_by_filter_with_multiple_filters(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str, "status": str, "priority": int, "rating": float}}],
        indirect=True,
    )
    def test_get_metadata_fields_info(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with required metadata fields."""
        GetMetadataFieldsInfoTest.test_get_metadata_fields_info(document_store)

    @pytest.mark.skip(reason="Azure AI Search returns index schema fields even on empty collections.")
    def test_get_metadata_fields_info_empty_collection(self, document_store: AzureAISearchDocumentStore): ...

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"priority": int}}],
        indirect=True,
    )
    def test_get_metadata_field_min_max_numeric(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with required metadata fields."""
        GetMetadataFieldMinMaxTest.test_get_metadata_field_min_max_numeric(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"rating": float}}],
        indirect=True,
    )
    def test_get_metadata_field_min_max_float(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with required metadata fields."""
        GetMetadataFieldMinMaxTest.test_get_metadata_field_min_max_float(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"priority": int}}],
        indirect=True,
    )
    def test_get_metadata_field_min_max_single_value(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with required metadata fields."""
        GetMetadataFieldMinMaxTest.test_get_metadata_field_min_max_single_value(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"priority": int}}],
        indirect=True,
    )
    def test_get_metadata_field_min_max_empty_collection(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with required metadata fields."""
        GetMetadataFieldMinMaxTest.test_get_metadata_field_min_max_empty_collection(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"priority": int, "age": int, "rating": float}}],
        indirect=True,
    )
    def test_get_metadata_field_min_max_meta_prefix(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with required metadata fields."""
        GetMetadataFieldMinMaxTest.test_get_metadata_field_min_max_meta_prefix(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str}}],
        indirect=True,
    )
    def test_get_metadata_field_unique_values_basic(self, document_store: AzureAISearchDocumentStore):
        """Override to use a document_store with required metadata fields."""
        GetMetadataFieldUniqueValuesTest.test_get_metadata_field_unique_values_basic(document_store)

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str, "status": str, "priority": int}}],
        indirect=True,
    )
    def test_count_documents_by_filter_integration(self, document_store: AzureAISearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "news", "status": "draft", "priority": 1}),
            Document(content="Doc 2", meta={"category": "news", "status": "published", "priority": 2}),
            Document(content="Doc 3", meta={"category": "docs", "status": "draft", "priority": 3}),
        ]
        document_store.write_documents(docs)

        count = document_store.count_documents_by_filter({"field": "meta.category", "operator": "==", "value": "news"})

        assert count == 2

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str, "status": str, "priority": int}}],
        indirect=True,
    )
    def test_count_unique_metadata_by_filter_integration(self, document_store: AzureAISearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "news", "status": "draft", "priority": 1}),
            Document(content="Doc 2", meta={"category": "news", "status": "published", "priority": 2}),
            Document(content="Doc 3", meta={"category": "docs", "status": "draft", "priority": 3}),
        ]
        document_store.write_documents(docs)

        counts = document_store.count_unique_metadata_by_filter(
            filters={"field": "meta.priority", "operator": ">=", "value": 1},
            metadata_fields=["meta.category", "status"],
        )

        assert counts == {"category": 2, "status": 2}

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str, "status": str, "priority": int}}],
        indirect=True,
    )
    def test_get_metadata_fields_info_integration(self, document_store: AzureAISearchDocumentStore):
        info = document_store.get_metadata_fields_info()

        assert info["content"] == {"type": "text"}
        assert info["category"] == {"type": "keyword"}
        assert info["status"] == {"type": "keyword"}
        assert info["priority"] == {"type": "long"}

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str, "status": str, "priority": int}}],
        indirect=True,
    )
    def test_get_metadata_field_min_max_integration(self, document_store: AzureAISearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "news", "status": "draft", "priority": 1}),
            Document(content="Doc 2", meta={"category": "news", "status": "published", "priority": 8}),
            Document(content="Doc 3", meta={"category": "docs", "status": "draft", "priority": 3}),
        ]
        document_store.write_documents(docs)

        result = document_store.get_metadata_field_min_max("meta.priority")

        assert result == {"min": 1, "max": 8}

    @pytest.mark.parametrize(
        "document_store",
        [{"metadata_fields": {"category": str, "status": str, "priority": int}}],
        indirect=True,
    )
    def test_get_metadata_field_unique_values_integration(self, document_store: AzureAISearchDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "news", "status": "draft", "priority": 1}),
            Document(content="Doc 2", meta={"category": "guides", "status": "published", "priority": 8}),
            Document(content="Doc 3", meta={"category": "docs", "status": "draft", "priority": 3}),
            Document(content="Doc 4", meta={"category": "news", "status": "archived", "priority": 10}),
        ]
        document_store.write_documents(docs)

        values, total_count = document_store.get_metadata_field_unique_values(
            metadata_field="meta.category",
            search_term="d",
            from_=0,
            size=10,
        )

        assert values == ["docs", "guides"]
        assert total_count == 2

    def test_query_sql_integration(self, document_store: AzureAISearchDocumentStore):
        with pytest.raises(NotImplementedError, match="does not support SQL queries"):
            document_store.query_sql("SELECT * FROM documents")


def _random_embeddings(n):
    return [round(random.random(), 7) for _ in range(n)]  # nosec: S311


TEST_EMBEDDING_1 = _random_embeddings(768)
TEST_EMBEDDING_2 = _random_embeddings(768)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("AZURE_AI_SEARCH_ENDPOINT", None) and not os.environ.get("AZURE_AI_SEARCH_API_KEY", None),
    reason="Missing AZURE_AI_SEARCH_ENDPOINT or AZURE_AI_SEARCH_API_KEY.",
)
@pytest.mark.parametrize(
    "document_store",
    [
        {"metadata_fields": {"name": str, "page": str, "chapter": str, "number": int, "date": datetime}},
    ],
    indirect=True,
)
class TestFilters(FilterDocumentsTest):
    # Overriding to change "date" to compatible ISO 8601 format
    @pytest.fixture
    def filterable_docs(self) -> list[Document]:
        """Fixture that returns a list of Documents that can be used to test filtering."""
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "100",
                        "chapter": "intro",
                        "number": 2,
                        "date": "1969-07-21T20:17:40Z",
                    },
                    embedding=_random_embeddings(768),
                )
            )
            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "123",
                        "chapter": "abstract",
                        "number": -2,
                        "date": "1972-12-11T19:54:58Z",
                    },
                    embedding=_random_embeddings(768),
                )
            )
            documents.append(
                Document(
                    content=f"A Foobar Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "page": "90",
                        "chapter": "conclusion",
                        "number": -10,
                        "date": "1989-11-09T17:53:00Z",
                    },
                    embedding=_random_embeddings(768),
                )
            )

            documents.append(
                Document(content=f"Doc {i} with zeros emb", meta={"name": "zeros_doc"}, embedding=TEST_EMBEDDING_1)
            )
            documents.append(
                Document(content=f"Doc {i} with ones emb", meta={"name": "ones_doc"}, embedding=TEST_EMBEDDING_2)
            )
        return documents

    # Overriding to compare the documents with the same order
    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        _assert_documents_are_equal(received, expected)

    # Azure search index supports UTC datetime in ISO 8601 format
    def test_comparison_greater_than_with_iso_date(self, document_store, filterable_docs):
        """Test filter_documents() with > comparator and datetime"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            {"field": "meta.date", "operator": ">", "value": "1972-12-11T19:54:58Z"}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("date") is not None
                and datetime.strptime(d.meta["date"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                > datetime.strptime("1972-12-11T19:54:58Z", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            ],
        )

    def test_comparison_greater_than_equal_with_iso_date(self, document_store, filterable_docs):
        """Test filter_documents() with >= comparator and datetime"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            {"field": "meta.date", "operator": ">=", "value": "1969-07-21T20:17:40Z"}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("date") is not None
                and datetime.strptime(d.meta["date"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                >= datetime.strptime("1969-07-21T20:17:40Z", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            ],
        )

    def test_comparison_less_than_with_iso_date(self, document_store, filterable_docs):
        """Test filter_documents() with < comparator and datetime"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            {"field": "meta.date", "operator": "<", "value": "1969-07-21T20:17:40Z"}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("date") is not None
                and datetime.strptime(d.meta["date"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                < datetime.strptime("1969-07-21T20:17:40Z", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            ],
        )

    def test_comparison_less_than_equal_with_iso_date(self, document_store, filterable_docs):
        """Test filter_documents() with <= comparator and datetime"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            {"field": "meta.date", "operator": "<=", "value": "1969-07-21T20:17:40Z"}
        )
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("date") is not None
                and datetime.strptime(d.meta["date"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                <= datetime.strptime("1969-07-21T20:17:40Z", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            ],
        )

    # Override as comparison operators with None/null raise errors
    def test_comparison_greater_than_with_none(self, document_store, filterable_docs):
        """Test filter_documents() with > comparator and None"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": ">", "value": None})

    def test_comparison_greater_than_equal_with_none(self, document_store, filterable_docs):
        """Test filter_documents() with >= comparator and None"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": ">=", "value": None})

    def test_comparison_less_than_with_none(self, document_store, filterable_docs):
        """Test filter_documents() with < comparator and None"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": "<", "value": None})

    def test_comparison_less_than_equal_with_none(self, document_store, filterable_docs):
        """Test filter_documents() with <= comparator and None"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"field": "meta.number", "operator": "<=", "value": None})

    # Override as Azure AI Search supports 'in' operator only for strings
    def test_comparison_in(self, document_store, filterable_docs):
        """Test filter_documents() with 'in' comparator"""
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents({"field": "meta.page", "operator": "in", "value": ["100", "123"]})
        assert len(result)
        expected = [d for d in filterable_docs if d.meta.get("page") is not None and d.meta["page"] in ["100", "123"]]
        self.assert_documents_are_equal(result, expected)

    @pytest.mark.skip(reason="Azure AI search index does not support not in operator")
    def test_comparison_not_in(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Azure AI search index does not support not in operator")
    def test_comparison_not_in_with_with_non_list(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Azure AI search index does not support not in operator")
    def test_comparison_not_in_with_with_non_list_iterable(self, document_store, filterable_docs): ...

    def test_missing_condition_operator_key(self, document_store, filterable_docs):
        """Test filter_documents() with missing operator key"""
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(
                filters={"conditions": [{"field": "meta.name", "operator": "eq", "value": "test"}]}
            )

    def test_nested_logical_filters(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        filters = {
            "operator": "OR",
            "conditions": [
                {"field": "meta.name", "operator": "==", "value": "name_0"},
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.number", "operator": "!=", "value": 0},
                        {"field": "meta.page", "operator": "==", "value": "123"},
                    ],
                },
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.chapter", "operator": "==", "value": "conclusion"},
                        {"field": "meta.page", "operator": "==", "value": "90"},
                    ],
                },
            ],
        }
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    # Ensure all required fields are present in doc.meta
                    ("name" in doc.meta and doc.meta.get("name") == "name_0")
                    or (
                        all(key in doc.meta for key in ["number", "page"])
                        and doc.meta.get("number") != 0
                        and doc.meta.get("page") == "123"
                    )
                    or (
                        all(key in doc.meta for key in ["page", "chapter"])
                        and doc.meta.get("chapter") == "conclusion"
                        and doc.meta.get("page") == "90"
                    )
                )
            ],
        )
