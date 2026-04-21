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
    def test_get_metadata_fields_info_empty_collection(self, document_store: SupabasePgvectorDocumentStore):
        """SupabasePgvectorDocumentStore always includes 'content' in fields info, even for empty stores."""
        assert document_store.count_documents() == 0

        fields_info = document_store.get_metadata_fields_info()
        assert fields_info == {"content": {"type": "text"}}

    def test_get_metadata_field_min_max_empty_collection(self, document_store: SupabasePgvectorDocumentStore):
        """SupabasePgvectorDocumentStore raises ValueError when the field doesn't exist in the store."""
        assert document_store.count_documents() == 0

        with pytest.raises(ValueError, match="not found in document store"):
            document_store.get_metadata_field_min_max("priority")

    def test_write_documents(self, document_store: SupabasePgvectorDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

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
