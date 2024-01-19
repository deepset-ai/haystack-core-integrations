# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


class TestDocumentStore(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest):
    @pytest.fixture
    def document_store(self, request):
        connection_string = "postgresql://postgres:postgres@localhost:5432/postgres"
        table_name = f"haystack_{request.node.name}"
        embedding_dimension = 768
        embedding_similarity_function = "cosine_distance"
        recreate_table = True
        search_strategy = "exact_nearest_neighbor"

        store = PgvectorDocumentStore(
            connection_string=connection_string,
            table_name=table_name,
            embedding_dimension=embedding_dimension,
            embedding_similarity_function=embedding_similarity_function,
            recreate_table=recreate_table,
            search_strategy=search_strategy,
        )
        yield store

        store.delete_table()
