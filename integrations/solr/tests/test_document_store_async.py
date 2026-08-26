# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store_async import (
    CountDocumentsAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    DeleteDocumentsAsyncTest,
    FilterDocumentsAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
    UpdateByFilterAsyncTest,
    WriteDocumentsAsyncTest,
)

from .test_document_store_common import SolrDocumentStoreTestMixin


@pytest.mark.integration
class TestSolrDocumentStoreAsync(
    SolrDocumentStoreTestMixin,
    WriteDocumentsAsyncTest,
    CountDocumentsAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    FilterDocumentsAsyncTest,
    DeleteDocumentsAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    UpdateByFilterAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
):
    """The shared async document store suites, run against a real Solr core."""

    @pytest.fixture
    def document_store(self, document_store):
        # Pull in the conftest fixture, overriding the plain one the base classes declare.
        return document_store

    async def test_write_documents_async(self, document_store) -> None:
        """`DuplicatePolicy.NONE` resolves to `FAIL`, so writing the same id twice raises."""
        documents = [Document(id="1", content="test doc")]
        assert await document_store.write_documents_async(documents) == 1
        with pytest.raises(DuplicateDocumentError):
            await document_store.write_documents_async(documents, DuplicatePolicy.NONE)


@pytest.mark.integration
class TestSolrAsyncSpecificBehaviour:
    async def test_sync_and_async_see_the_same_data(self, document_store):
        """The two clients are independent, but they address the same core."""
        document_store.write_documents([Document(id="1", content="written sync")], DuplicatePolicy.OVERWRITE)
        assert await document_store.count_documents_async() == 1

        await document_store.write_documents_async(
            [Document(id="2", content="written async")], DuplicatePolicy.OVERWRITE
        )
        assert document_store.count_documents() == 2

    async def test_bm25_retrieval_async(self, document_store):
        await document_store.write_documents_async(
            [
                Document(id="1", content="Apache Solr is a search platform"),
                Document(id="2", content="Completely unrelated text about cooking"),
            ],
            DuplicatePolicy.OVERWRITE,
        )
        results = await document_store._bm25_retrieval_async("Apache Solr search", top_k=10)
        assert results[0].id == "1"

    async def test_embedding_retrieval_async(self, document_store):
        await document_store.write_documents_async(
            [
                Document(id="1", content="a", meta={"group": "x"}, embedding=[1.0] + [0.0] * 767),
                Document(id="2", content="b", meta={"group": "y"}, embedding=[1.0] + [0.0] * 767),
            ],
            DuplicatePolicy.OVERWRITE,
        )
        results = await document_store._embedding_retrieval_async(
            [1.0] + [0.0] * 767,
            filters={"field": "meta.group", "operator": "==", "value": "x"},
            top_k=10,
        )
        assert [document.id for document in results] == ["1"]

    async def test_metadata_types_survive_an_async_round_trip(self, document_store):
        document = Document(id="1", content="x", meta={"page": "100", "number": -2, "flag": False})
        await document_store.write_documents_async([document], DuplicatePolicy.OVERWRITE)
        restored = (await document_store.filter_documents_async())[0]
        assert restored.meta == document.meta

    async def test_reconnects_after_close(self, document_store):
        await document_store.write_documents_async([Document(id="1", content="x")], DuplicatePolicy.OVERWRITE)
        await document_store.close_async()
        assert await document_store.count_documents_async() == 1
