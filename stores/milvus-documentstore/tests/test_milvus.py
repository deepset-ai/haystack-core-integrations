# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import numpy as np
from milvus_documentstore.milvus import MilvusDocumentStore
from haystack.schema import Document
from haystack.testing import DocumentStoreBaseTestAbstract


datastore = MilvusDocumentStore(recreate_index=True, embedding_dim=768)


class TestMilvusDocumentStore(DocumentStoreBaseTestAbstract):
    @pytest.fixture
    def documents(self):
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "year": "2020",
                        "month": "01",
                        "numbers": [2, 4],
                    },
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "year": "2021",
                        "month": "02",
                        "numbers": [-2, -4],
                    },
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    content=f"Document {i}",
                    meta={"name": f"name_{i}", "month": "03"},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

        return documents

    @pytest.fixture
    def documents_not_list(self):
        documents = []
        for i in range(3):
            documents.append(
                Document(
                    content=f"A Foo Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "year": "2020",
                        "month": "01",
                        "numbers": 4,
                    },
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    content=f"A Bar Document {i}",
                    meta={
                        "name": f"name_{i}",
                        "year": "2021",
                        "month": "02",
                        "numbers": -4,
                    },
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

            documents.append(
                Document(
                    content=f"Document {i}",
                    meta={"name": f"name_{i}", "month": "03"},
                    embedding=np.random.rand(768).astype(np.float32),
                )
            )

        return documents

    @pytest.fixture
    def ds(self):
        datastore.delete_all_documents()
        datastore.delete_all_documents("custom_index")
        datastore.delete_all_labels()
        datastore.delete_all_labels("custom_index")

        return datastore

    @pytest.mark.integration
    def test_delete_index(self, ds, documents):
        """Contrary to other Document Stores, MilvusDocumentStore doesn't raise if the index is empty"""
        ds.write_documents(documents, index="custom_index")
        assert ds.get_document_count(index="custom_index") == len(documents)
        ds.delete_index(index="custom_index")
        assert ds.get_document_count(index="custom_index") == 0

    @pytest.mark.integration
    def test_get_embedding_count(self, ds, documents):
        ds.write_documents(documents)
        assert ds.get_embedding_count() == 9

    @pytest.mark.skip
    @pytest.mark.integration
    def test_comparison_filters(self, ds, documents):
        pass

    @pytest.mark.integration
    def test_comparison_filters_not_list(self, ds, documents_not_list):
        ds.write_documents(documents_not_list)

        result = ds.get_all_documents(filters={"numbers": {"$gt": 0.0}})
        print(result)
        assert len(result) == 3

        result = ds.get_all_documents(filters={"numbers": {"$gte": -4.0}})
        assert len(result) == 6

        result = ds.get_all_documents(filters={"numbers": {"$lt": 0.0}})
        assert len(result) == 3

        result = ds.get_all_documents(filters={"numbers": {"$lte": 4.0}})
        assert len(result) == 6

    @pytest.mark.skip
    @pytest.mark.integration
    def test_nested_condition_filters(self, ds, documents):
        pass

    @pytest.mark.integration
    def test_nested_condition_filters_not_list(self, ds, documents_not_list):
        ds.write_documents(documents_not_list)
        filters = {
            "$and": {
                "year": {"$lte": "2021", "$gte": "2020"},
                "$or": {"name": {"$in": ["name_0", "name_1"]}, "numbers": {"$lt": 5.0}},
            }
        }
        result = ds.get_all_documents(filters=filters)
        assert len(result) == 6

        filters_simplified = {
            "year": {"$lte": "2021", "$gte": "2020"},
            "$or": {"name": {"$in": ["name_0", "name_2"]}, "numbers": {"$lt": 5.0}},
        }
        result = ds.get_all_documents(filters=filters_simplified)
        assert len(result) == 6

        filters = {
            "$and": {
                "year": {"$lte": "2021", "$gte": "2020"},
                "$or": {
                    "name": {"$in": ["name_0", "name_1"]},
                    "$and": {"numbers": {"$lt": 5.0}, "$not": {"month": {"$eq": "01"}}},
                },
            }
        }
        result = ds.get_all_documents(filters=filters)
        assert len(result) == 5

        filters_simplified = {
            "year": {"$lte": "2021", "$gte": "2020"},
            "$or": {
                "name": ["name_0", "name_1"],
                "$and": {"numbers": {"$lt": 5.0}, "$not": {"month": {"$eq": "01"}}},
            },
        }
        result = ds.get_all_documents(filters=filters_simplified)
        assert len(result) == 5

    @pytest.mark.skip
    @pytest.mark.integration
    def test_nested_condition_not_filters(self, ds, documents):
        pass

    @pytest.mark.integration
    def test_nested_condition_filters_not_list(self, ds, documents_not_list):
        ds.write_documents(documents_not_list)
        filters = {
            "$and": {
                "year": {"$lte": "2021", "$gte": "2020"},
                "$or": {"name": {"$in": ["name_0", "name_1"]}, "numbers": {"$lt": 5.0}},
            }
        }
        result = ds.get_all_documents(filters=filters)
        assert len(result) == 6

        filters_simplified = {
            "year": {"$lte": "2021", "$gte": "2020"},
            "$or": {"name": {"$in": ["name_0", "name_2"]}, "numbers": {"$lt": 5.0}},
        }
        result = ds.get_all_documents(filters=filters_simplified)
        assert len(result) == 6

        filters = {
            "$and": {
                "year": {"$lte": "2021", "$gte": "2020"},
                "$or": {
                    "name": {"$in": ["name_0", "name_1"]},
                    "$and": {"numbers": {"$lt": 5.0}, "$not": {"month": {"$eq": "01"}}},
                },
            }
        }
        result = ds.get_all_documents(filters=filters)
        assert len(result) == 5

        filters_simplified = {
            "year": {"$lte": "2021", "$gte": "2020"},
            "$or": {
                "name": ["name_0", "name_1"],
                "$and": {"numbers": {"$lt": 5.0}, "$not": {"month": {"$eq": "01"}}},
            },
        }
        result = ds.get_all_documents(filters=filters_simplified)
        assert len(result) == 5

    @pytest.mark.skip
    @pytest.mark.integration
    def test_custom_embedding_field(self, ds, documents):
        pass
