# SPDX-FileCopyrightText: 2023-present Silvano Cerza <silvanocerza@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List
from unittest.mock import patch

import pandas as pd
import pytest
from haystack.preview.dataclasses.document import Document
from haystack.preview.document_stores.errors import DuplicateDocumentError
from haystack.preview.document_stores.protocols import DuplicatePolicy
from haystack.preview.testing.document_store import DocumentStoreBaseTests

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

    @pytest.mark.skip(reason="Not supported")
    def test_in_filter_table(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Not supported")
    def test_in_filter_embedding(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        pass

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

    def test_lt_filter_non_numeric(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"page": {"$lt": "100"}})
        assert result == []

    def test_lt_filter_table(self, docstore: ElasticsearchDocumentStore, filterable_docs: List[Document]):
        docstore.write_documents(filterable_docs)
        result = docstore.filter_documents(filters={"dataframe": {"$lt": pd.DataFrame([[1, 2, 3], [-1, -2, -3]])}})
        assert self.contains_same_docs(result, [d for d in filterable_docs if d.dataframe is not None])

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
