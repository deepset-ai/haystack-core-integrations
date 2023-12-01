# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import List

import pytest
from haystack.preview import Document
from haystack.preview.document_stores import DuplicatePolicy
from haystack.preview.testing.document_store import DocumentStoreBaseTests

from astra_store.document_store import AstraDocumentStore


class TestDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def docstore(self) -> AstraDocumentStore:
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        astra_id = os.getenv("ASTRA_DB_ID", "")
        astra_region = os.getenv("ASTRA_DB_REGION", "us-east1")

        astra_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")
        keyspace_name = "test"
        collection_name = "test_collection"

        astra_store = AstraDocumentStore(
            astra_id=astra_id,
            astra_region=astra_region,
            astra_application_token=astra_application_token,
            astra_keyspace=keyspace_name,
            astra_collection=collection_name,
            duplicates_policy=DuplicatePolicy.OVERWRITE,
            embedding_dim=384,
        )
        return astra_store

    @pytest.mark.unit
    def test_delete_all(self, docstore: AstraDocumentStore):
        """
        Cleaning up document store
        """
        docstore.delete_documents(delete_all=True)
        assert docstore.count_documents() == 0

    @pytest.mark.skip(reason="Unsupported filter operator $lte")
    @pytest.mark.unit
    def test_lte_filter_non_numeric(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $ne.")
    @pytest.mark.unit
    def test_ne_filter(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gte.")
    @pytest.mark.unit
    def test_gte_filter(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gte.")
    @pytest.mark.unit
    def test_gte_filter_non_numeric(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gte.")
    @pytest.mark.unit
    def test_gte_filter_table(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gte.")
    @pytest.mark.unit
    def test_gte_filter_embedding(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gt.")
    @pytest.mark.unit
    def test_gt_filter(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gt.")
    @pytest.mark.unit
    def test_gt_filter_non_numeric(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gt.")
    @pytest.mark.unit
    def test_gt_filter_table(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gt.")
    @pytest.mark.unit
    def test_gt_filter_embedding(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lt.")
    @pytest.mark.unit
    def test_lt_filter(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lt.")
    @pytest.mark.unit
    def test_lt_filter_non_numeric(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lt.")
    @pytest.mark.unit
    def test_lt_filter_table(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lt.")
    @pytest.mark.unit
    def test_lt_filter_embedding(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lte.")
    @pytest.mark.unit
    def test_lte_filter(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lte.")
    @pytest.mark.unit
    def test_lte_filter_non_numeric(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lte.")
    @pytest.mark.unit
    def test_lte_filter_table(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lte.")
    @pytest.mark.unit
    def test_lte_filter_embedding(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $ne.")
    @pytest.mark.unit
    def test_ne_filter_table(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $ne.")
    @pytest.mark.unit
    def test_ne_filter_embedding(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $nin.")
    @pytest.mark.unit
    def test_nin_filter(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $nin.")
    @pytest.mark.unit
    def test_nin_filter_table(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $nin.")
    @pytest.mark.unit
    def test_nin_filter_embedding(self, docstore: AstraDocumentStore, filterable_docs: List[Document]):
        pass
