# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import numpy as np
import pytest
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest
from haystack.utils import Secret
from pandas import DataFrame

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


@pytest.mark.integration
class TestDocumentStore(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest):
    def test_write_documents(self, document_store: PgvectorDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_write_blob(self, document_store: PgvectorDocumentStore):
        bytestream = ByteStream(b"test", meta={"meta_key": "meta_value"}, mime_type="mime_type")
        docs = [Document(id="1", blob=bytestream)]
        document_store.write_documents(docs)

        retrieved_docs = document_store.filter_documents()
        assert retrieved_docs == docs

    def test_write_dataframe(self, document_store: PgvectorDocumentStore):
        dataframe = DataFrame({"col1": [1, 2], "col2": [3, 4]})
        docs = [Document(id="1", dataframe=dataframe)]

        document_store.write_documents(docs)

        retrieved_docs = document_store.filter_documents()
        assert retrieved_docs == docs

    def test_connection_check_and_recreation(self, document_store: PgvectorDocumentStore):
        original_connection = document_store.connection

        with patch.object(PgvectorDocumentStore, "_connection_is_valid", return_value=False):
            new_connection = document_store.connection

        # verify that a new connection is created
        assert new_connection is not original_connection
        assert document_store._connection == new_connection
        assert original_connection.closed

        assert document_store._cursor is not None
        assert document_store._dict_cursor is not None

        # test with new connection
        with patch.object(PgvectorDocumentStore, "_connection_is_valid", return_value=True):
            same_connection = document_store.connection
            assert same_connection is document_store._connection


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_init(monkeypatch):
    monkeypatch.setenv("PG_CONN_STR", "some_connection_string")

    document_store = PgvectorDocumentStore(
        schema_name="my_schema",
        table_name="my_table",
        embedding_dimension=512,
        vector_function="l2_distance",
        recreate_table=True,
        search_strategy="hnsw",
        hnsw_recreate_index_if_exists=True,
        hnsw_index_creation_kwargs={"m": 32, "ef_construction": 128},
        hnsw_index_name="my_hnsw_index",
        hnsw_ef_search=50,
        keyword_index_name="my_keyword_index",
    )

    assert document_store.schema_name == "my_schema"
    assert document_store.table_name == "my_table"
    assert document_store.embedding_dimension == 512
    assert document_store.vector_function == "l2_distance"
    assert document_store.recreate_table
    assert document_store.search_strategy == "hnsw"
    assert document_store.hnsw_recreate_index_if_exists
    assert document_store.hnsw_index_creation_kwargs == {"m": 32, "ef_construction": 128}
    assert document_store.hnsw_index_name == "my_hnsw_index"
    assert document_store.hnsw_ef_search == 50
    assert document_store.keyword_index_name == "my_keyword_index"


@pytest.mark.usefixtures("patches_for_unit_tests")
def test_to_dict(monkeypatch):
    monkeypatch.setenv("PG_CONN_STR", "some_connection_string")

    document_store = PgvectorDocumentStore(
        table_name="my_table",
        embedding_dimension=512,
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
        "type": "haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore",
        "init_parameters": {
            "connection_string": {"env_vars": ["PG_CONN_STR"], "strict": True, "type": "env_var"},
            "table_name": "my_table",
            "schema_name": "public",
            "embedding_dimension": 512,
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


def test_from_haystack_to_pg_documents():
    haystack_docs = [
        Document(
            id="1",
            content="This is a text",
            meta={"meta_key": "meta_value"},
            embedding=[0.1, 0.2, 0.3],
            score=0.5,
        ),
        Document(
            id="2",
            dataframe=DataFrame({"col1": [1, 2], "col2": [3, 4]}),
            meta={"meta_key": "meta_value"},
            embedding=[0.4, 0.5, 0.6],
            score=0.6,
        ),
        Document(
            id="3",
            blob=ByteStream(b"test", meta={"blob_meta_key": "blob_meta_value"}, mime_type="mime_type"),
            meta={"meta_key": "meta_value"},
            embedding=[0.7, 0.8, 0.9],
            score=0.7,
        ),
    ]

    with patch(
        "haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore.__init__"
    ) as mock_init:
        mock_init.return_value = None
        ds = PgvectorDocumentStore(connection_string="test")

    pg_docs = ds._from_haystack_to_pg_documents(haystack_docs)

    assert pg_docs[0]["id"] == "1"
    assert pg_docs[0]["content"] == "This is a text"
    assert pg_docs[0]["dataframe"] is None
    assert pg_docs[0]["blob_data"] is None
    assert pg_docs[0]["blob_meta"] is None
    assert pg_docs[0]["blob_mime_type"] is None
    assert pg_docs[0]["meta"].obj == {"meta_key": "meta_value"}
    assert pg_docs[0]["embedding"] == [0.1, 0.2, 0.3]
    assert "score" not in pg_docs[0]

    assert pg_docs[1]["id"] == "2"
    assert pg_docs[1]["content"] is None
    assert pg_docs[1]["dataframe"].obj == DataFrame({"col1": [1, 2], "col2": [3, 4]}).to_json()
    assert pg_docs[1]["blob_data"] is None
    assert pg_docs[1]["blob_meta"] is None
    assert pg_docs[1]["blob_mime_type"] is None
    assert pg_docs[1]["meta"].obj == {"meta_key": "meta_value"}
    assert pg_docs[1]["embedding"] == [0.4, 0.5, 0.6]
    assert "score" not in pg_docs[1]

    assert pg_docs[2]["id"] == "3"
    assert pg_docs[2]["content"] is None
    assert pg_docs[2]["dataframe"] is None
    assert pg_docs[2]["blob_data"] == b"test"
    assert pg_docs[2]["blob_meta"].obj == {"blob_meta_key": "blob_meta_value"}
    assert pg_docs[2]["blob_mime_type"] == "mime_type"
    assert pg_docs[2]["meta"].obj == {"meta_key": "meta_value"}
    assert pg_docs[2]["embedding"] == [0.7, 0.8, 0.9]
    assert "score" not in pg_docs[2]


def test_from_pg_to_haystack_documents():
    pg_docs = [
        {
            "id": "1",
            "content": "This is a text",
            "dataframe": None,
            "blob_data": None,
            "blob_meta": None,
            "blob_mime_type": None,
            "meta": {"meta_key": "meta_value"},
            "embedding": np.array([0.1, 0.2, 0.3]),
        },
        {
            "id": "2",
            "content": None,
            "dataframe": DataFrame({"col1": [1, 2], "col2": [3, 4]}).to_json(),
            "blob_data": None,
            "blob_meta": None,
            "blob_mime_type": None,
            "meta": {"meta_key": "meta_value"},
            "embedding": np.array([0.4, 0.5, 0.6]),
        },
        {
            "id": "3",
            "content": None,
            "dataframe": None,
            "blob_data": b"test",
            "blob_meta": {"blob_meta_key": "blob_meta_value"},
            "blob_mime_type": "mime_type",
            "meta": {"meta_key": "meta_value"},
            "embedding": np.array([0.7, 0.8, 0.9]),
        },
    ]

    ds = PgvectorDocumentStore(connection_string=Secret.from_token("test"))
    haystack_docs = ds._from_pg_to_haystack_documents(pg_docs)

    assert haystack_docs[0].id == "1"
    assert haystack_docs[0].content == "This is a text"
    assert haystack_docs[0].dataframe is None
    assert haystack_docs[0].blob is None
    assert haystack_docs[0].meta == {"meta_key": "meta_value"}
    assert haystack_docs[0].embedding == [0.1, 0.2, 0.3]
    assert haystack_docs[0].score is None

    assert haystack_docs[1].id == "2"
    assert haystack_docs[1].content is None
    assert haystack_docs[1].dataframe.equals(DataFrame({"col1": [1, 2], "col2": [3, 4]}))
    assert haystack_docs[1].blob is None
    assert haystack_docs[1].meta == {"meta_key": "meta_value"}
    assert haystack_docs[1].embedding == [0.4, 0.5, 0.6]
    assert haystack_docs[1].score is None

    assert haystack_docs[2].id == "3"
    assert haystack_docs[2].content is None
    assert haystack_docs[2].dataframe is None
    assert haystack_docs[2].blob.data == b"test"
    assert haystack_docs[2].blob.meta == {"blob_meta_key": "blob_meta_value"}
    assert haystack_docs[2].blob.mime_type == "mime_type"
    assert haystack_docs[2].meta == {"meta_key": "meta_value"}
    assert haystack_docs[2].embedding == [0.7, 0.8, 0.9]
    assert haystack_docs[2].score is None
