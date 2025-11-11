import numpy as np
import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from haystack.dataclasses import Document
from haystack.testing.document_store import (
    TEST_EMBEDDING_1,
    TEST_EMBEDDING_2,
    _random_embeddings,
)


class _TestEmbeddingFunction(EmbeddingFunction):
    """
    Chroma lets you provide custom functions to compute embeddings,
    we use this feature to provide a fake algorithm returning random
    vectors in unit tests.
    """

    def __call__(self, input: Documents) -> Embeddings:  # noqa - chroma will inspect the signature, it must match
        # embed the documents somehow
        return [np.random.default_rng().uniform(-1, 1, 768).tolist()]


@pytest.fixture
def embedding_function():
    return _TestEmbeddingFunction()


@pytest.fixture
def filterable_docs() -> list[Document]:
    """
    This fixture has been copied from haystack/testing/document_store.py and modified to
    remove the documents that don't have textual content, as Chroma does not support writing them.
    """
    documents = []
    for i in range(3):
        documents.append(
            Document(
                content=f"A Foo Document {i}",
                meta={
                    "name": f"name_{i}",
                    "page": "100",
                    "chapter": "intro",
                    "number": 2,
                    "date": "1969-07-21T20:17:40",
                },
                embedding=_random_embeddings(768),
            )
        )
        documents.append(
            Document(
                content=f"A Bar Document {i}",
                meta={
                    "name": f"name_{i}",
                    "page": "123",
                    "chapter": "abstract",
                    "number": -2,
                    "date": "1972-12-11T19:54:58",
                },
                embedding=_random_embeddings(768),
            )
        )
        documents.append(
            Document(
                content=f"A Foobar Document {i}",
                meta={
                    "name": f"name_{i}",
                    "page": "90",
                    "chapter": "conclusion",
                    "number": -10,
                    "date": "1989-11-09T17:53:00",
                },
                embedding=_random_embeddings(768),
            )
        )
        documents.append(
            Document(
                content=f"Document {i} without embedding",
                meta={
                    "name": f"name_{i}",
                    "no_embedding": True,
                    "chapter": "conclusion",
                },
            )
        )
        documents.append(
            Document(
                content=f"Doc {i} with zeros emb",
                meta={"name": "zeros_doc"},
                embedding=TEST_EMBEDDING_1,
            )
        )
        documents.append(
            Document(
                content=f"Doc {i} with ones emb",
                meta={"name": "ones_doc"},
                embedding=TEST_EMBEDDING_2,
            )
        )
    return documents
