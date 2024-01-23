# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore
from pandas import DataFrame


class TestDocumentStore(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest):
    @pytest.fixture
    def document_store(self, request):
        connection_string = "postgresql://postgres:postgres@localhost:5432/postgres"
        table_name = f"haystack_{request.node.name}"
        embedding_dimension = 768
        vector_function = "cosine_distance"
        recreate_table = True
        search_strategy = "exact_nearest_neighbor"

        store = PgvectorDocumentStore(
            connection_string=connection_string,
            table_name=table_name,
            embedding_dimension=embedding_dimension,
            vector_function=vector_function,
            recreate_table=recreate_table,
            search_strategy=search_strategy,
        )
        yield store

        store.delete_table()

    def test_write_documents(self, document_store: PgvectorDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_write_blob(self, document_store: PgvectorDocumentStore):
        bytestream = ByteStream(b"test", meta={"meta_key": "meta_value"}, mime_type="mime_type")
        docs = [Document(id="1", blob=bytestream)]
        document_store.write_documents(docs)

        # TODO: update when filters are implemented
        retrieved_docs = document_store.filter_documents()
        assert retrieved_docs == docs

    def test_write_dataframe(self, document_store: PgvectorDocumentStore):
        dataframe = DataFrame({"col1": [1, 2], "col2": [3, 4]})
        docs = [Document(id="1", dataframe=dataframe)]

        document_store.write_documents(docs)

        # TODO: update when filters are implemented
        retrieved_docs = document_store.filter_documents()
        assert retrieved_docs == docs

    def test_init(self):
        document_store = PgvectorDocumentStore(
            connection_string="postgresql://postgres:postgres@localhost:5432/postgres",
            table_name="my_table",
            embedding_dimension=512,
            vector_function="l2_distance",
            recreate_table=True,
            search_strategy="hnsw",
            hnsw_recreate_index_if_exists=True,
            hnsw_index_creation_kwargs={"m": 32, "ef_construction": 128},
            hnsw_ef_search=50,
        )

        assert document_store.connection_string == "postgresql://postgres:postgres@localhost:5432/postgres"
        assert document_store.table_name == "my_table"
        assert document_store.embedding_dimension == 512
        assert document_store.vector_function == "l2_distance"
        assert document_store.recreate_table
        assert document_store.search_strategy == "hnsw"
        assert document_store.hnsw_recreate_index_if_exists
        assert document_store.hnsw_index_creation_kwargs == {"m": 32, "ef_construction": 128}
        assert document_store.hnsw_ef_search == 50

    def test_to_dict(self):
        document_store = PgvectorDocumentStore(
            connection_string="postgresql://postgres:postgres@localhost:5432/postgres",
            table_name="my_table",
            embedding_dimension=512,
            vector_function="l2_distance",
            recreate_table=True,
            search_strategy="hnsw",
            hnsw_recreate_index_if_exists=True,
            hnsw_index_creation_kwargs={"m": 32, "ef_construction": 128},
            hnsw_ef_search=50,
        )

        assert document_store.to_dict() == {
            "type": "haystack_integrations.document_stores.pgvector.document_store.PgvectorDocumentStore",
            "init_parameters": {
                "connection_string": "postgresql://postgres:postgres@localhost:5432/postgres",
                "table_name": "my_table",
                "embedding_dimension": 512,
                "vector_function": "l2_distance",
                "recreate_table": True,
                "search_strategy": "hnsw",
                "hnsw_recreate_index_if_exists": True,
                "hnsw_index_creation_kwargs": {"m": 32, "ef_construction": 128},
                "hnsw_ef_search": 50,
            },
        }
