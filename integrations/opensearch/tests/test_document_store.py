# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import random
from typing import List
from unittest.mock import patch

import pytest
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
from opensearchpy.exceptions import RequestError


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_to_dict(_mock_opensearch_client):
    document_store = OpenSearchDocumentStore(hosts="some hosts")
    res = document_store.to_dict()
    assert res == {
        "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
        "init_parameters": {
            "hosts": "some hosts",
            "index": "default",
        },
    }


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_from_dict(_mock_opensearch_client):
    data = {
        "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
        "init_parameters": {
            "hosts": "some hosts",
            "index": "default",
        },
    }
    document_store = OpenSearchDocumentStore.from_dict(data)
    assert document_store._hosts == "some hosts"
    assert document_store._index == "default"


@patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
def test_init_is_lazy(_mock_opensearch_client):
    OpenSearchDocumentStore(hosts="testhost")
    _mock_opensearch_client.assert_not_called()


@pytest.mark.integration
class TestDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def document_store(self, request):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        hosts = ["https://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = OpenSearchDocumentStore(
            hosts=hosts,
            index=index,
            http_auth=("admin", "admin"),
            verify_certs=False,
            embedding_dim=768,
            method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
        )
        yield store
        store.client.indices.delete(index=index, params={"ignore": [400, 404]})

    @pytest.fixture
    def document_store_embedding_dim_4(self, request):
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        hosts = ["https://localhost:9200"]
        # Use a different index for each test so we can run them in parallel
        index = f"{request.node.name}"

        store = OpenSearchDocumentStore(
            hosts=hosts,
            index=index,
            http_auth=("admin", "admin"),
            verify_certs=False,
            embedding_dim=4,
            method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
        )
        yield store
        store.client.indices.delete(index=index, params={"ignore": [400, 404]})

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        """
        The OpenSearchDocumentStore.filter_documents() method returns a Documents with their score set.
        We don't want to compare the score, so we set it to None before comparing the documents.
        """
        received_meta = []
        for doc in received:
            r = {
                "number": doc.meta.get("number"),
                "name": doc.meta.get("name"),
            }
            received_meta.append(r)

        expected_meta = []
        for doc in expected:
            r = {
                "number": doc.meta.get("number"),
                "name": doc.meta.get("name"),
            }
            expected_meta.append(r)
        for doc in received:
            doc.score = None

        super().assert_documents_are_equal(received, expected)

    def test_write_documents(self, document_store: OpenSearchDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_bm25_retrieval(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
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

        res = document_store._bm25_retrieval("functional", top_k=3)
        assert len(res) == 3
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content

    def test_bm25_retrieval_pagination(self, document_store: OpenSearchDocumentStore):
        """
        Test that handling of pagination works as expected, when the matching documents are > 10.
        """
        document_store.write_documents(
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
                Document(content="Java is an object oriented programming language"),
                Document(content="Javascript is a programming language"),
                Document(content="Typescript is a programming language"),
                Document(content="C is a programming language"),
            ]
        )

        res = document_store._bm25_retrieval("programming", top_k=11)
        assert len(res) == 11
        assert all("programming" in doc.content for doc in res)

    def test_bm25_retrieval_all_terms_must_match(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
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

        res = document_store._bm25_retrieval("functional Haskell", top_k=3, all_terms_must_match=True)
        assert len(res) == 1
        assert "Haskell is a functional programming language" in res[0].content

    def test_bm25_retrieval_all_terms_must_match_false(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
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

        res = document_store._bm25_retrieval("functional Haskell", top_k=10, all_terms_must_match=False)
        assert len(res) == 5
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content
        assert "functional" in res[3].content
        assert "functional" in res[4].content

    def test_bm25_retrieval_with_fuzziness(self, document_store: OpenSearchDocumentStore):
        document_store.write_documents(
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

        query_with_typo = "functinal"
        # Query without fuzziness to search for the exact match
        res = document_store._bm25_retrieval(query_with_typo, top_k=3, fuzziness="0")
        # Nothing is found as the query contains a typo
        assert res == []

        # Query with fuzziness with the same query
        res = document_store._bm25_retrieval(query_with_typo, top_k=3, fuzziness="1")
        assert len(res) == 3
        assert "functional" in res[0].content
        assert "functional" in res[1].content
        assert "functional" in res[2].content

    def test_embedding_retrieval(self, document_store_embedding_dim_4: OpenSearchDocumentStore):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(content="Not very similar document", embedding=[0.0, 0.8, 0.3, 0.9]),
        ]
        document_store_embedding_dim_4.write_documents(docs)
        results = document_store_embedding_dim_4._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=2, filters={}
        )
        assert len(results) == 2
        assert results[0].content == "Most similar document"
        assert results[1].content == "2nd best document"

    def test_embedding_retrieval_with_filters(self, document_store_embedding_dim_4: OpenSearchDocumentStore):
        docs = [
            Document(content="Most similar document", embedding=[1.0, 1.0, 1.0, 1.0]),
            Document(content="2nd best document", embedding=[0.8, 0.8, 0.8, 1.0]),
            Document(
                content="Not very similar document with meta field",
                embedding=[0.0, 0.8, 0.3, 0.9],
                meta={"meta_field": "custom_value"},
            ),
        ]
        document_store_embedding_dim_4.write_documents(docs)

        filters = {"field": "meta_field", "operator": "==", "value": "custom_value"}
        # we set top_k=3, to make the test pass as we are not sure whether efficient filtering is supported for nmslib
        # TODO: remove top_k=3, when efficient filtering is supported for nmslib
        results = document_store_embedding_dim_4._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=3, filters=filters
        )
        assert len(results) == 1
        assert results[0].content == "Not very similar document with meta field"

    def test_embedding_retrieval_pagination(self, document_store_embedding_dim_4: OpenSearchDocumentStore):
        """
        Test that handling of pagination works as expected, when the matching documents are > 10.
        """

        docs = [
            Document(content=f"Document {i}", embedding=[random.random() for _ in range(4)])  # noqa: S311
            for i in range(20)
        ]

        document_store_embedding_dim_4.write_documents(docs)
        results = document_store_embedding_dim_4._embedding_retrieval(
            query_embedding=[0.1, 0.1, 0.1, 0.1], top_k=11, filters={}
        )
        assert len(results) == 11

    def test_embedding_retrieval_query_documents_different_embedding_sizes(
        self, document_store_embedding_dim_4: OpenSearchDocumentStore
    ):
        """
        Test that the retrieval fails if the query embedding and the documents have different embedding sizes.
        """
        docs = [Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4])]
        document_store_embedding_dim_4.write_documents(docs)

        with pytest.raises(RequestError):
            document_store_embedding_dim_4._embedding_retrieval(query_embedding=[0.1, 0.1])

    def test_write_documents_different_embedding_sizes_fail(
        self, document_store_embedding_dim_4: OpenSearchDocumentStore
    ):
        """
        Test that write_documents fails if the documents have different embedding sizes.
        """
        docs = [
            Document(content="Hello world", embedding=[0.1, 0.2, 0.3, 0.4]),
            Document(content="Hello world", embedding=[0.1, 0.2]),
        ]

        with pytest.raises(DocumentStoreError):
            document_store_embedding_dim_4.write_documents(docs)

    @patch("haystack_integrations.document_stores.opensearch.document_store.bulk")
    def test_write_documents_with_badly_formatted_bulk_errors(self, mock_bulk, document_store):
        error = {"some_key": "some_value"}
        mock_bulk.return_value = ([], [error])

        with pytest.raises(DocumentStoreError) as e:
            document_store.write_documents([Document(content="Hello world")])
            e.match(f"{error}")
