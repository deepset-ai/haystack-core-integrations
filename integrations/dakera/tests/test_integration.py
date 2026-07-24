# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests exercised against a live Dakera server.

These are skipped unless ``DAKERA_URL`` is set. Start a server with the
`dakera-deploy` docker-compose stack (server + MinIO) before running them:

    hatch run test:integration
"""

import os

import pytest
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

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


def test_write_and_count(document_store):
    written = document_store.write_documents(_docs(), policy=DuplicatePolicy.OVERWRITE)
    assert written == 3
    assert document_store.count_documents() == 3


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


def test_delete_documents(document_store):
    document_store.write_documents(_docs(), policy=DuplicatePolicy.OVERWRITE)
    document_store.delete_documents(["a"])
    assert document_store.count_documents() == 2
