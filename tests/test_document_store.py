# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import List

import pytest
from haystack import Document
from haystack.document_stores import DuplicatePolicy, MissingDocumentError
from haystack.testing.document_store import DocumentStoreBaseTests

from astra_store.document_store import AstraDocumentStore


class TestDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def document_store(self) -> AstraDocumentStore:
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        astra_id = os.getenv("ASTRA_DB_ID", "")
        astra_region = os.getenv("ASTRA_DB_REGION", "us-east1")

        astra_application_token = os.getenv(
            "ASTRA_DB_APPLICATION_TOKEN",
            "",
        )

        keyspace_name = "astra_haystack_test"
        collection_name = "test_collection_new"

        astra_store = AstraDocumentStore(
            astra_id=astra_id,
            astra_region=astra_region,
            astra_application_token=astra_application_token,
            astra_keyspace=keyspace_name,
            astra_collection=collection_name,
            duplicates_policy=DuplicatePolicy.OVERWRITE,
            embedding_dim=768,
            model_name="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
        )
        return astra_store

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self, document_store: AstraDocumentStore):
        """
        Cleaning up document store
        """
        document_store.delete_documents(delete_all=True)
        assert document_store.count_documents() == 0

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        """
        Assert that two lists of Documents are equal.
        This is used in every test, if a Document Store implementation has a different behaviour
        it should override this method.

        This can happen for example when the Document Store sets a score to returned Documents.
        Since we can't know what the score will be, we can't compare the Documents reliably.
        """
        import operator

        received.sort(key=operator.attrgetter('content'))
        assert received == expected

    def test_delete_documents_non_existing_document(self, document_store: AstraDocumentStore):
        """
        Test delete_documents() doesn't delete any Document when called with non existing id.
        """
        doc = Document(content="test doc")
        document_store.write_documents([doc])
        assert document_store.count_documents() == 1

        with pytest.raises(MissingDocumentError):
            document_store.delete_documents(["non_existing_id"])

        # No Document has been deleted
        assert document_store.count_documents() == 1

    # @pytest.mark.skip(reason="Unsupported filter operator not in.")
    # def test_comparison_not_in(self, document_store, filterable_docs):
    #     pass
    #
    # @pytest.mark.skip(reason="Unsupported filter operator not in.")
    # def test_comparison_not_in_with_with_non_list(self, document_store, filterable_docs):
    #     pass
    #
    # @pytest.mark.skip(reason="Unsupported filter operator not in.")
    # def test_comparison_not_in_with_with_non_list_iterable(self, document_store, filterable_docs):
    #     pass

    # @pytest.mark.skip(reason="Unsupported filter operator $gt.")
    # def test_comparison_greater_than_with_iso_date(self, document_store, filterable_docs):
    #     pass

    # @pytest.mark.skip(reason="Unsupported filter operator $gt.")
    # def test_comparison_greater_than_with_string(self, document_store, filterable_docs):
    #     pass

    # @pytest.mark.skip(reason="Unsupported filter operator $gt.")
    # def test_comparison_greater_than_with_dataframe(self, document_store, filterable_docs):
    #     pass

    # @pytest.mark.skip(reason="Unsupported filter operator $gt.")
    # def test_comparison_greater_than_with_list(self, document_store, filterable_docs):
    #     pass

    # @pytest.mark.skip(reason="Unsupported filter operator $gt.")
    # def test_comparison_greater_than_with_none(self, document_store, filterable_docs):
    #     pass

    # @pytest.mark.skip(reason="Unsupported filter operator $gte.")
    # def test_comparison_greater_than_equal(self, document_store, filterable_docs):
    #     pass
