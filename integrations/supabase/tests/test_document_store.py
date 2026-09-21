# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountDocumentsTest,
    CountUniqueMetadataByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
    UpdateByFilterTest,
    WriteDocumentsTest,
)

from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore


@pytest.mark.integration
class TestDocumentStore(
    CountDocumentsTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    UpdateByFilterTest,
    WriteDocumentsTest,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
):
    def test_write_documents(self, document_store: SupabasePgvectorDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_get_metadata_field_unique_values_distinct_types(self, document_store: SupabasePgvectorDocumentStore):
        """
        Override: the base mixin test stores int, float, str and bool under the *same* metadata field
        name and expects all four back as distinct values. This store's ``meta`` column is JSONB, and
        PostgreSQL's JSONB equality treats a whole-number float (e.g. 1.0) and a numerically equal int
        (1) as the same value, so ``SELECT DISTINCT meta->'field'`` collapses them regardless of which
        other values share that field.

        This adapts the same intent - int, float, str and bool must come back as distinct, unmangled
        types via get_metadata_field_unique_values() - using one field per type instead of one shared
        field, which is what this store can actually support.

        The float value is a non-whole number (1.5, not 1.0): a whole-number float would still collapse
        with an int under PostgreSQL's JSONB numeric equality even in its own field, so a fractional
        value is used to sidestep that ambiguity entirely.
        """
        docs = [
            Document(content="Doc 1", meta={"priority_int": 1}),
            Document(content="Doc 2", meta={"priority_str": "1"}),
            Document(content="Doc 3", meta={"priority_float": 1.5}),
            Document(content="Doc 4", meta={"priority_bool": True}),
        ]
        document_store.write_documents(docs)

        int_values, int_count = document_store.get_metadata_field_unique_values(metadata_field="priority_int")
        str_values, str_count = document_store.get_metadata_field_unique_values(metadata_field="priority_str")
        float_values, float_count = document_store.get_metadata_field_unique_values(metadata_field="priority_float")
        bool_values, bool_count = document_store.get_metadata_field_unique_values(metadata_field="priority_bool")

        assert (int_count, str_count, float_count, bool_count) == (1, 1, 1, 1)
        assert int_values == [1] and type(int_values[0]) is int
        assert str_values == ["1"] and type(str_values[0]) is str
        assert float_values == [1.5] and type(float_values[0]) is float
        assert bool_values == [True] and type(bool_values[0]) is bool

    def test_write_blob(self, document_store: SupabasePgvectorDocumentStore):
        bytestream = ByteStream(b"test", meta={"meta_key": "meta_value"}, mime_type="mime_type")
        docs = [Document(id="1", blob=bytestream)]
        document_store.write_documents(docs)

        retrieved_docs = document_store.filter_documents()
        assert retrieved_docs == docs


@pytest.mark.integration
def test_delete_table_first_call(document_store):
    """
    Test that delete_table can be executed as the initial operation on the Document Store
    without triggering errors due to an uninitialized state.
    """
    document_store.delete_table()


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_init(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "some_connection_string")

    document_store = SupabasePgvectorDocumentStore(
        create_extension=True,
        schema_name="my_schema",
        table_name="my_table",
        language="spanish",
        embedding_dimension=512,
        vector_type="halfvec",
        vector_function="l2_distance",
        recreate_table=True,
        search_strategy="hnsw",
        hnsw_recreate_index_if_exists=True,
        hnsw_index_creation_kwargs={"m": 32, "ef_construction": 128},
        hnsw_index_name="my_hnsw_index",
        hnsw_ef_search=50,
        keyword_index_name="my_keyword_index",
    )

    assert document_store.create_extension
    assert document_store.schema_name == "my_schema"
    assert document_store.table_name == "my_table"
    assert document_store.language == "spanish"
    assert document_store.embedding_dimension == 512
    assert document_store.vector_type == "halfvec"
    assert document_store.vector_function == "l2_distance"
    assert document_store.recreate_table
    assert document_store.search_strategy == "hnsw"
    assert document_store.hnsw_recreate_index_if_exists
    assert document_store.hnsw_index_creation_kwargs == {"m": 32, "ef_construction": 128}
    assert document_store.hnsw_index_name == "my_hnsw_index"
    assert document_store.hnsw_ef_search == 50
    assert document_store.keyword_index_name == "my_keyword_index"


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_init_defaults(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "some_connection_string")

    document_store = SupabasePgvectorDocumentStore()

    assert not document_store.create_extension
    assert document_store.schema_name == "public"
    assert document_store.table_name == "haystack_documents"
    assert document_store.language == "english"
    assert document_store.embedding_dimension == 768
    assert document_store.vector_type == "vector"
    assert document_store.vector_function == "cosine_similarity"
    assert not document_store.recreate_table
    assert document_store.search_strategy == "exact_nearest_neighbor"


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_to_dict(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "some_connection_string")

    document_store = SupabasePgvectorDocumentStore(
        table_name="my_table",
        embedding_dimension=512,
        vector_type="halfvec",
        vector_function="l2_distance",
        recreate_table=True,
        search_strategy="hnsw",
        hnsw_recreate_index_if_exists=True,
        hnsw_index_creation_kwargs={"m": 32, "ef_construction": 128},
        hnsw_index_name="my_hnsw_index",
        hnsw_ef_search=50,
        keyword_index_name="my_keyword_index",
    )

    assert document_store.to_dict() == {
        "type": "haystack_integrations.document_stores.supabase.document_store.SupabasePgvectorDocumentStore",
        "init_parameters": {
            "connection_string": {"env_vars": ["SUPABASE_DB_URL"], "strict": True, "type": "env_var"},
            "create_extension": False,
            "table_name": "my_table",
            "schema_name": "public",
            "embedding_dimension": 512,
            "vector_type": "halfvec",
            "vector_function": "l2_distance",
            "recreate_table": True,
            "search_strategy": "hnsw",
            "hnsw_recreate_index_if_exists": True,
            "language": "english",
            "hnsw_index_creation_kwargs": {"m": 32, "ef_construction": 128},
            "hnsw_index_name": "my_hnsw_index",
            "hnsw_ef_search": 50,
            "keyword_index_name": "my_keyword_index",
        },
    }


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_from_dict(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_URL", "some_connection_string")

    data = {
        "type": "haystack_integrations.document_stores.supabase.document_store.SupabasePgvectorDocumentStore",
        "init_parameters": {
            "connection_string": {"env_vars": ["SUPABASE_DB_URL"], "strict": True, "type": "env_var"},
            "create_extension": False,
            "table_name": "my_table",
            "schema_name": "public",
            "embedding_dimension": 512,
            "vector_type": "halfvec",
            "vector_function": "l2_distance",
            "recreate_table": True,
            "search_strategy": "hnsw",
            "hnsw_recreate_index_if_exists": True,
            "language": "english",
            "hnsw_index_creation_kwargs": {"m": 32, "ef_construction": 128},
            "hnsw_index_name": "my_hnsw_index",
            "hnsw_ef_search": 50,
            "keyword_index_name": "my_keyword_index",
        },
    }

    document_store = SupabasePgvectorDocumentStore.from_dict(data)

    assert isinstance(document_store, SupabasePgvectorDocumentStore)
    assert not document_store.create_extension
    assert document_store.table_name == "my_table"
    assert document_store.schema_name == "public"
    assert document_store.embedding_dimension == 512
    assert document_store.vector_type == "halfvec"
    assert document_store.vector_function == "l2_distance"
    assert document_store.recreate_table
    assert document_store.search_strategy == "hnsw"
    assert document_store.hnsw_recreate_index_if_exists
    assert document_store.hnsw_index_creation_kwargs == {"m": 32, "ef_construction": 128}
    assert document_store.hnsw_index_name == "my_hnsw_index"
    assert document_store.hnsw_ef_search == 50
    assert document_store.keyword_index_name == "my_keyword_index"
