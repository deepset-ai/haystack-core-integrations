# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests exercised against a live Dakera server.

These are skipped unless ``DAKERA_URL`` is set. Start a server with the
`dakera-deploy` docker-compose stack (server + MinIO) before running them:

    hatch run test:integration

The ``TestDakeraDocumentStore`` class reuses Haystack's ``CountDocumentsTest``,
``DeleteDocumentsTest`` and ``WriteDocumentsTest`` mixins. Dakera upserts vectors by id,
so the duplicate-policy cases (``FAIL``/``SKIP``) are skipped — the same approach the
Pinecone integration takes for an overwrite-by-id vector store.
"""

import os

import pytest
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteDocumentsTest,
    WriteDocumentsTest,
)

from haystack_integrations.components.retrievers.dakera import DakeraEmbeddingRetriever

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("DAKERA_URL"), reason="DAKERA_URL not set"),
]


def _docs():
    return [
        Document(id="a", content="alpha", embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], meta={"lang": "en"}),
        Document(id="b", content="beta", embedding=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], meta={"lang": "fr"}),
        Document(id="c", content="gamma", embedding=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], meta={"lang": "en"}),
    ]


class TestDakeraDocumentStore(CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest):
    """Standard Haystack document-store behaviour against a live Dakera server.

    The ``document_store`` fixture is provided by ``conftest.py``.
    """

    def test_write_documents(self, document_store):
        docs = [Document(id="1", embedding=[0.1] * 8)]
        assert document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE) == 1

    @pytest.mark.skip(reason="Dakera upserts by id (OVERWRITE only)")
    def test_write_documents_duplicate_fail(self, document_store): ...

    @pytest.mark.skip(reason="Dakera upserts by id (OVERWRITE only)")
    def test_write_documents_duplicate_skip(self, document_store): ...

    @pytest.mark.skip(reason="Dakera creates a namespace lazily; deleting from an empty store is a no-op")
    def test_delete_documents_empty_document_store(self, document_store): ...


def test_embedding_retrieval(document_store):
    document_store.write_documents(_docs(), policy=DuplicatePolicy.OVERWRITE)
    retriever = DakeraEmbeddingRetriever(document_store=document_store, top_k=1)
    result = retriever.run(query_embedding=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert result["documents"][0].id == "a"
    assert result["documents"][0].content == "alpha"


def test_filter_documents(document_store):
    document_store.write_documents(_docs(), policy=DuplicatePolicy.OVERWRITE)
    filters = {"field": "meta.lang", "operator": "==", "value": "en"}
    docs = document_store.filter_documents(filters=filters)
    assert {doc.id for doc in docs} == {"a", "c"}


def test_user_metadata_named_content_survives_round_trip(document_store):
    # Regression guard: a user's own `content` metadata must not be clobbered by the
    # Document's content (stored under the reserved `_dakera_content` key).
    doc = Document(id="x", content="body text", embedding=[0.5] * 8, meta={"content": "user meta"})
    document_store.write_documents([doc], policy=DuplicatePolicy.OVERWRITE)
    retriever = DakeraEmbeddingRetriever(document_store=document_store, top_k=1)
    retrieved = retriever.run(query_embedding=[0.5] * 8)["documents"][0]
    assert retrieved.content == "body text"
    assert retrieved.meta.get("content") == "user meta"
