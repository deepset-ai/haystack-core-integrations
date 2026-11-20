# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from uuid import uuid4

import pytest

from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


@pytest.fixture()
def test_files_path():
    return Path(__file__).parent / "test_files"


@pytest.fixture
def multivector_document_store():
    store = WeaviateDocumentStore(
        url="http://localhost:8080",
        collection_settings={
            "class": f"Multivector{uuid4().hex}",
            "vectorConfig": {
                "default": {"vectorizer": {"none": {}}, "vectorIndexType": "hnsw"},
                "multi_vector": {
                    "vectorizer": {"none": {}},
                    "vectorIndexType": "hnsw",
                    "vectorIndexConfig": {"multivector": {"enabled": True}},
                },
            },
        },
    )
    try:
        yield store
    finally:
        store.client.collections.delete(store._collection_settings["class"])
        store.close()
