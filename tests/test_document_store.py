# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import List

import pytest
from haystack.dataclasses.document import Document
from haystack.document_stores.protocol import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests
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
    def run_before_and_after_tests(self, docstore: AstraDocumentStore):
        """
        Cleaning up document store
        """
        docstore.delete_documents(delete_all=True)
        assert docstore.count_documents() == 0

