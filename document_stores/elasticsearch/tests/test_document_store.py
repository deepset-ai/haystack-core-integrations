# SPDX-FileCopyrightText: 2023-present Silvano Cerza <silvanocerza@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List
from unittest.mock import patch

import pytest
from haystack.preview.dataclasses.document import Document
from haystack.preview.document_stores.errors import DuplicateDocumentError
from haystack.preview.document_stores.protocols import DuplicatePolicy
from haystack.preview.testing.document_store import DocumentStoreBaseTests
from haystack.preview.errors import FilterError
import pandas as pd
import numpy as np

from elasticsearch_haystack.document_store import ElasticsearchDocumentStore


class TestDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def docstore(self, request):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        hosts = ["http://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = ElasticsearchDocumentStore(hosts=hosts, index=index)
        yield store
        store._client.options(ignore_status=[400, 404]).indices.delete(index=index)

    @patch("elasticsearch_haystack.document_store.Elasticsearch")
    def test_to_dict(self, _mock_elasticsearch_client):
        document_store = ElasticsearchDocumentStore(hosts="some hosts")
        res = document_store.to_dict()
        assert res == {
            "type": "ElasticsearchDocumentStore",
            "init_parameters": {
                "hosts": "some hosts",
                "index": "default",
            },
        }

    @patch("elasticsearch_haystack.document_store.Elasticsearch")
    def test_from_dict(self, _mock_elasticsearch_client):
        data = {
            "type": "ElasticsearchDocumentStore",
            "init_parameters": {
                "hosts": "some hosts",
                "index": "default",
            },
        }
        document_store = ElasticsearchDocumentStore.from_dict(data)
        assert document_store._hosts == "some hosts"
        assert document_store._index == "default"

    def test_bm25_retrieval(self, docstore: ElasticsearchDocumentStore):
        docstore.write_documents(
            [
                Document(content="Haskell is a functional programming language"),
                Document(content="Lisp is a functional programming language"),
                Document(content="Exilir is a functional programming language"),
                Document(content="F# is a functional programming language"),
                Document(content="C# is a functional programming language"),
                Document(content="C++ is an object oriented programming language"),
                Document(content="Dart is an object oriented programming language"),
                Document(content="Go is an object oriented programming language"),
                Document(content="Python is a object oriented programming language"),
                Document(content="Ruby is a object oriented programming language"),
                Document(content="PHP is a object oriented programming language"),
            ]
        )

        res = docstore._bm25_retrieval("functional", top_k=3)
        assert len(res) == 3
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content

    def test_write_duplicate_fail(self, docstore: ElasticsearchDocumentStore):
        """
        Verify `DuplicateDocumentError` is raised when trying to write duplicate files.

        `DocumentStoreBaseTests` declares this test but we override it since we return
        a different error message that it expects.
        """
        doc = Document(content="test doc")
        docstore.write_documents([doc])
        with pytest.raises(DuplicateDocumentError):
            docstore.write_documents(documents=[doc], policy=DuplicatePolicy.FAIL)
        assert docstore.filter_documents(filters={"id": doc.id}) == [doc]

    def test_delete_not_empty(self, docstore: ElasticsearchDocumentStore):
        """
        Verifies delete properly deletes specified document.

        `DocumentStoreBaseTests` declares this test but we override it since we
        want `delete_documents` to be idempotent.
        """
        doc = Document(text="test doc")
        docstore.write_documents([doc])

        docstore.delete_documents([doc.id])

        res = docstore.filter_documents(filters={"id": doc.id})
        assert res == []

    def test_delete_empty(self, docstore: ElasticsearchDocumentStore):
        """
        Verifies delete doesn't raises when trying to delete a non-existing document.

        `DocumentStoreBaseTests` declares this test but we override it since we
        want `delete_documents` to be idempotent.
        """
        docstore.delete_documents(["test"])

    def test_delete_not_empty_nonexisting(self, docstore: ElasticsearchDocumentStore):
        """
        Verifies delete properly deletes specified document in DocumentStore containing
        multiple documents.

        `DocumentStoreBaseTests` declares this test but we override it since we
        want `delete_documents` to be idempotent.
        """
        doc = Document(text="test doc")
        docstore.write_documents([doc])

        docstore.delete_documents(["non_existing"])

        assert docstore.filter_documents(filters={"id": doc.id}) == [doc]

    ####

    def test_count_empty(self, docstore: ElasticsearchDocumentStore):
        assert docstore.count_documents() == 0

    def test_count_not_empty(self, docstore: ElasticsearchDocumentStore):
        docstore.write_documents(
            [Document(content="test doc 1"), Document(content="test doc 2"), Document(content="test doc 3")]
        )
        assert docstore.count_documents() == 3

    def test_no_filter_empty(self, docstore: ElasticsearchDocumentStore):
        assert docstore.filter_documents() == []
        assert docstore.filter_documents(filters={}) == []

    def test_no_filter_not_empty(self, docstore: ElasticsearchDocumentStore):
        docs = [Document(content="test doc")]
        docstore.write_documents(docs)
        assert docstore.filter_documents() == docs
        assert docstore.filter_documents(filters={}) == docs

    def test_filter_simple_metadata_value(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": "100"})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"])

    def test_filter_simple_list_single_element(
        self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]
    ):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": ["100"]})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"])

    def test_filter_document_content(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"content": "A Foo Document 1"})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.content == "A Foo Document 1"])

    def test_filter_document_dataframe(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"dataframe": pd.DataFrame([1])})
        assert self.contains_same_docs(
            result,
            [doc for doc in filterable_docs if doc.dataframe is not None and doc.dataframe.equals(pd.DataFrame([1]))],
        )

    def test_filter_simple_list_one_value(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": ["100"]})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.meta.get("page") in ["100"]])

    def test_filter_simple_list(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": ["100", "123"]})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]]
        )

    def test_incorrect_filter_name(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"non_existing_meta_field": ["whatever"]})
        assert len(result) == 0

    def test_incorrect_filter_type(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters="something odd")  # type: ignore

    def test_incorrect_filter_value(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": ["nope"]})
        assert len(result) == 0

    def test_incorrect_filter_nesting(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"number": {"page": "100"}})

    def test_deeper_incorrect_filter_nesting(
        self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]
    ):
        docstore.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"number": {"page": {"chapter": "intro"}}})

    def test_eq_filter_explicit(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$eq": "100"}})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"])

    def test_eq_filter_implicit(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": "100"})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"])

    def test_eq_filter_table(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"dataframe": pd.DataFrame([1])})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if isinstance(doc.dataframe, pd.DataFrame) and doc.dataframe.equals(pd.DataFrame([1]))
            ],
        )

    def test_eq_filter_embedding(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        embedding = [0.0] * 768
        result = docstore.filter_documents(filters={"embedding": embedding})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if embedding == doc.embedding])

    def test_in_filter_explicit(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$in": ["100", "123", "n.a."]}})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]]
        )

    def test_in_filter_implicit(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": ["100", "123", "n.a."]})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]]
        )

    @pytest.mark.skip(reason="Not supported")
    def test_in_filter_table(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Not supported")
    def test_in_filter_embedding(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        pass

    def test_ne_filter(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$ne": "100"}})
        assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.meta.get("page") != "100"])

    def test_ne_filter_table(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"dataframe": {"$ne": pd.DataFrame([1])}})
        assert self.contains_same_docs(
            result,
            [doc for doc in filterable_docs if doc.dataframe is None or not doc.dataframe.equals(pd.DataFrame([1]))],
        )

    def test_ne_filter_embedding(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        embedding = [0.0] * 768
        result = docstore.filter_documents(filters={"embedding": {"$ne": embedding}})
        assert self.contains_same_docs(
            result,
            [doc for doc in filterable_docs if doc.embedding is None or not embedding == doc.embedding],
        )

    @pytest.mark.skip(reason="Not supported")
    def test_nin_filter_table(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Not supported")
    def test_nin_filter_embedding(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        pass

    def test_nin_filter(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$nin": ["100", "123", "n.a."]}})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if doc.meta.get("page") not in ["100", "123"]]
        )

    def test_gt_filter(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$gt": 0.0}})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if "number" in doc.meta and doc.meta["number"] > 0]
        )

    def test_gt_filter_non_numeric(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$gt": "100"}})
        assert self.contains_same_docs(
            result, [d for d in filterable_docs if "page" in d.meta and d.meta["page"] > "100"]
        )

    def test_gt_filter_table(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"dataframe": {"$gt": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})
        assert result == []

    def test_gt_filter_embedding(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        embedding_zeros = np.zeros([768, 1]).astype(np.float32)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"embedding": {"$gt": embedding_zeros}})

    def test_gte_filter(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$gte": -2}})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if "number" in doc.meta and doc.meta["number"] >= -2]
        )

    def test_gte_filter_non_numeric(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$gte": "100"}})
        assert self.contains_same_docs(
            result, [d for d in filterable_docs if "page" in d.meta and d.meta["page"] >= "100"]
        )

    def test_gte_filter_table(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"dataframe": {"$gte": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})
        assert result == []

    def test_gte_filter_embedding(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        embedding_zeros = np.zeros([768, 1]).astype(np.float32)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"embedding": {"$gte": embedding_zeros}})

    def test_lt_filter(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lt": 0.0}})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if "number" in doc.meta and doc.meta["number"] < 0]
        )

    def test_lt_filter_non_numeric(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$lt": "100"}})
        assert result == []

    def test_lt_filter_table(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"dataframe": {"$lt": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})
        assert self.contains_same_docs(result, [d for d in filterable_docs if d.dataframe is not None])

    def test_lt_filter_embedding(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        embedding_ones = np.ones([768, 1]).astype(np.float32)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"embedding": {"$lt": embedding_ones}})

    def test_lte_filter(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lte": 2.0}})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if "number" in doc.meta and doc.meta["number"] <= 2.0]
        )

    def test_lte_filter_non_numeric(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$lte": "100"}})
        assert self.contains_same_docs(
            result, [d for d in filterable_docs if "page" in d.meta and d.meta["page"] <= "100"]
        )

    def test_lte_filter_table(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"dataframe": {"$lte": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})
        assert self.contains_same_docs(result, [d for d in filterable_docs if d.dataframe is not None])

    def test_lte_filter_embedding(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        embedding_ones = np.ones([768, 1]).astype(np.float32)
        with pytest.raises(FilterError):
            docstore.filter_documents(filters={"embedding": {"$lte": embedding_ones}})

    def test_filter_simple_implicit_and_with_multi_key_dict(
        self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]
    ):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lte": 2.0, "$gte": 0.0}})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.meta and doc.meta["number"] >= 0.0 and doc.meta["number"] <= 2.0
            ],
        )

    def test_filter_simple_explicit_and_with_multikey_dict(
        self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]
    ):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$and": {"$gte": 0, "$lte": 2}}})
        assert self.contains_same_docs(
            result, [doc for doc in filterable_docs if "number" in doc.meta and 0 <= doc.meta["number"] <= 2]
        )

    def test_filter_simple_explicit_and_with_list(
        self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]
    ):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$and": [{"$lte": 2}, {"$gte": 0}]}})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.meta and doc.meta["number"] <= 2.0 and doc.meta["number"] >= 0.0
            ],
        )

    def test_filter_simple_implicit_and(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"number": {"$lte": 2.0, "$gte": 0}})
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.meta and doc.meta["number"] <= 2.0 and doc.meta["number"] >= 0.0
            ],
        )

    def test_filter_nested_explicit_and(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        filters = {"$and": {"number": {"$and": {"$lte": 2, "$gte": 0}}, "name": {"$in": ["name_0", "name_1"]}}}
        result = docstore.filter_documents(filters=filters)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    "number" in doc.meta
                    and doc.meta["number"] >= 0
                    and doc.meta["number"] <= 2
                    and doc.meta["name"] in ["name_0", "name_1"]
                )
            ],
        )

    def test_filter_nested_implicit_and(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        filters_simplified = {"number": {"$lte": 2, "$gte": 0}, "name": ["name_0", "name_1"]}
        result = docstore.filter_documents(filters=filters_simplified)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    "number" in doc.meta
                    and doc.meta["number"] <= 2
                    and doc.meta["number"] >= 0
                    and doc.meta.get("name") in ["name_0", "name_1"]
                )
            ],
        )

    def test_filter_simple_or(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        filters = {"$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}}}
        result = docstore.filter_documents(filters=filters)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (("number" in doc.meta and doc.meta["number"] < 1) or doc.meta.get("name") in ["name_0", "name_1"])
            ],
        )

    def test_filter_nested_or(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        filters = {"$or": {"name": {"$or": [{"$eq": "name_0"}, {"$eq": "name_1"}]}, "number": {"$lt": 1.0}}}
        result = docstore.filter_documents(filters=filters)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (doc.meta.get("name") in ["name_0", "name_1"] or ("number" in doc.meta and doc.meta["number"] < 1))
            ],
        )

    def test_filter_nested_and_or_explicit(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        filters_simplified = {
            "$and": {"page": {"$eq": "123"}, "$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}}}
        }
        result = docstore.filter_documents(filters=filters_simplified)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    doc.meta.get("page") in ["123"]
                    and (
                        doc.meta.get("name") in ["name_0", "name_1"]
                        or ("number" in doc.meta and doc.meta["number"] < 1)
                    )
                )
            ],
        )

    def test_filter_nested_and_or_implicit(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        filters_simplified = {
            "page": {"$eq": "123"},
            "$or": {"name": {"$in": ["name_0", "name_1"]}, "number": {"$lt": 1.0}},
        }
        result = docstore.filter_documents(filters=filters_simplified)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    doc.meta.get("page") in ["123"]
                    and (
                        doc.meta.get("name") in ["name_0", "name_1"]
                        or ("number" in doc.meta and doc.meta["number"] < 1)
                    )
                )
            ],
        )

    def test_filter_nested_or_and(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        filters_simplified = {
            "$or": {
                "number": {"$lt": 1},
                "$and": {"name": {"$in": ["name_0", "name_1"]}, "$not": {"chapter": {"$eq": "intro"}}},
            }
        }
        result = docstore.filter_documents(filters=filters_simplified)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    ("number" in doc.meta and doc.meta["number"] < 1)
                    or (
                        doc.meta.get("name") in ["name_0", "name_1"]
                        and ("chapter" in doc.meta and doc.meta["chapter"] != "intro")
                    )
                )
            ],
        )

    def test_filter_nested_multiple_identical_operators_same_level(
        self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]
    ):
        docstore.write_documents(filterable_docs)
        filters = {
            "$or": [
                {"$and": {"name": {"$in": ["name_0", "name_1"]}, "page": "100"}},
                {"$and": {"chapter": {"$in": ["intro", "abstract"]}, "page": "123"}},
            ]
        }
        result = docstore.filter_documents(filters=filters)
        assert self.contains_same_docs(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    (doc.meta.get("name") in ["name_0", "name_1"] and doc.meta.get("page") == "100")
                    or (doc.meta.get("chapter") in ["intro", "abstract"] and doc.meta.get("page") == "123")
                )
            ],
        )

    def test_write(self, docstore: ElasticsearchDocumentStore):
        doc = Document(content="test doc")
        docstore.write_documents([doc])
        assert docstore.filter_documents(filters={"id": doc.id}) == [doc]

    def test_write_duplicate_skip(self, docstore: ElasticsearchDocumentStore):
        doc = Document(content="test doc")
        docstore.write_documents([doc])
        docstore.write_documents(documents=[doc], policy=DuplicatePolicy.SKIP)
        assert docstore.filter_documents(filters={"id": doc.id}) == [doc]

    def test_write_duplicate_overwrite(self, docstore: ElasticsearchDocumentStore):
        doc1 = Document(content="test doc 1")
        doc2 = Document(content="test doc 2")
        object.__setattr__(doc2, "id", doc1.id)  # Make two docs with different content but same ID

        docstore.write_documents([doc2])
        assert docstore.filter_documents(filters={"id": doc1.id}) == [doc2]
        docstore.write_documents(documents=[doc1], policy=DuplicatePolicy.OVERWRITE)
        assert docstore.filter_documents(filters={"id": doc1.id}) == [doc1]

    def test_write_not_docs(self, docstore: ElasticsearchDocumentStore):
        with pytest.raises(ValueError):
            docstore.write_documents(["not a document for sure"])  # type: ignore

    def test_write_not_list(self, docstore: ElasticsearchDocumentStore):
        with pytest.raises(ValueError):
            docstore.write_documents("not a list actually")  # type: ignore

    # The tests below are filters not supported by ElasticsearchDocumentStore
    # def test_in_filter_table(self):
    #     pass

    # def test_in_filter_embedding(self):
    #     pass

    # def test_ne_filter_table(self):
    #     pass

    # def test_ne_filter_embedding(self):
    #     pass

    # def test_nin_filter_table(self):
    #     pass

    # def test_nin_filter_embedding(self):
    #     pass

    # def test_gt_filter_non_numeric(self):
    #     pass

    # def test_gt_filter_table(self):
    #     pass

    # def test_gt_filter_embedding(self):
    #     pass

    # def test_gte_filter_non_numeric(self):
    #     pass

    # def test_gte_filter_table(self):
    #     pass

    # def test_gte_filter_embedding(self):
    #     pass

    # def test_lt_filter_non_numeric(self):
    #     pass

    # def test_lt_filter_table(self):
    #     pass

    # def test_lt_filter_embedding(self):
    #     pass

    # def test_lte_filter_non_numeric(self):
    #     pass

    # def test_lte_filter_table(self):
    #     pass

    # def test_lte_filter_embedding(self):
    #     pass

    # def test_nin_filter(self, docstore, filterable_docs):
    #     docstore.write_documents(filterable_docs)
    #     result = docstore.filter_documents(filters={"page": {"$nin": ["100", "123", "n.a."]}})
    #     expected = [doc for doc in filterable_docs if doc.meta.get("page") not in ["100", "123"]]
    #     assert self.contains_same_docs(result, expected)

    # def test_filter_document_text(self, docstore, filterable_docs):
    #     docstore.write_documents(filterable_docs)
    #     result = docstore.filter_documents(filters={"text": "A Foo Document 1"})
    #     assert self.contains_same_docs(result, [doc for doc in filterable_docs if doc.text == "A Foo Document 1"])

    # def test_filter_document_dataframe(self, docstore, filterable_docs):
    #     docstore.write_documents(filterable_docs)
    #     result = docstore.filter_documents(filters={"dataframe": pd.DataFrame([1])})
    #     expected = (
    #         [doc for doc in filterable_docs if doc.dataframe is not None and doc.dataframe.equals(pd.DataFrame([1]))],
    #     )
    #     assert self.contains_same_docs(result, expected)

    # def test_eq_filter_table(self, docstore, filterable_docs):
    #     docstore.write_documents(filterable_docs)
    #     result = docstore.filter_documents(filters={"dataframe": pd.DataFrame([1])})
    #     assert self.contains_same_docs(
    #         result,
    #         [
    #             doc
    #             for doc in filterable_docs
    #             if isinstance(doc.dataframe, pd.DataFrame) and doc.dataframe.equals(pd.DataFrame([1]))
    #         ],
    #     )

    # def test_ne_filter(self, docstore, filterable_docs):
    #     docstore.write_documents(filterable_docs)
    #     result = docstore.filter_documents(filters={"page": {"$ne": "100"}})
    #     assert self.contains_same_docs(
    #         result,
    #         [doc for doc in filterable_docs if doc.meta.get("page") != "100"],
    #     )

    # def test_nin_filter_table(self, docstore, filterable_docs):
    #     docstore.write_documents(filterable_docs)
    #     result = docstore.filter_documents(filters={"dataframe": {"$nin": [pd.DataFrame([1]), pd.DataFrame([0])]}})
    #     assert self.contains_same_docs(
    #         result,
    #         [
    #             doc
    #             for doc in filterable_docs
    #             if not isinstance(doc.dataframe, pd.DataFrame)
    #             or (not doc.dataframe.equals(pd.DataFrame([1])) and not doc.dataframe.equals(pd.DataFrame([0])))
    #         ],
    #     )

    # def test_filter_nested_or(self, docstore, filterable_docs):
    #     docstore.write_documents(filterable_docs)
    #     filters = {
    #         "$or": {
    #             "name": {"$or": [{"$eq": "name_0"}, {"$eq": "name_1"}]},
    #             "number": {"$lt": 1.0},
    #         }
    #     }
    #     result = docstore.filter_documents(filters=filters)
    #     assert self.contains_same_docs(
    #         result,
    #         [
    #             doc
    #             for doc in filterable_docs
    #             if (doc.meta.get("name") in ["name_0", "name_1"] or ("number" in doc.meta and doc.meta["number"] < 1))
    #         ],
    #     )

    # def test_filter_nested_or_and(self, docstore, filterable_docs):
    #     docstore.write_documents(filterable_docs)
    #     filters_simplified = {
    #         "$or": {
    #             "number": {"$lt": 1},
    #             "$and": {
    #                 "name": {"$in": ["name_0", "name_1"]},
    #                 "$not": {"chapter": {"$eq": "intro"}},
    #             },
    #         }
    #     }
    #     result = docstore.filter_documents(filters=filters_simplified)
    #     assert self.contains_same_docs(
    #         result,
    #         [
    #             doc
    #             for doc in filterable_docs
    #             if (
    #                 ("number" in doc.meta and doc.meta["number"] < 1)
    #                 or (
    #                     doc.meta.get("name") in ["name_0", "name_1"]
    #                     and ("chapter" in doc.meta and doc.meta["chapter"] != "intro")
    #                 )
    #             )
    #         ],
    #     )
