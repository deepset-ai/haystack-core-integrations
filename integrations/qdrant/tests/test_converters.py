from types import SimpleNamespace

import numpy as np
import pytest
from haystack.dataclasses import Document, SparseEmbedding
from qdrant_client.http import models as rest

from haystack_integrations.document_stores.qdrant.converters import (
    DENSE_VECTORS_NAME,
    SPARSE_VECTORS_NAME,
    convert_haystack_documents_to_qdrant_points,
    convert_id,
    convert_qdrant_point_to_haystack_document,
)


def test_convert_id_is_deterministic():
    first_id = convert_id("test-id")
    second_id = convert_id("test-id")
    assert first_id == second_id


def test_point_to_document_reverts_proper_structure_from_record_with_sparse():
    point = rest.Record(
        id="c7c62e8e-02b9-4ec6-9f88-46bd97b628b7",
        payload={
            "id": "my-id",
            "id_hash_keys": ["content"],
            "content": "Lorem ipsum",
            "content_type": "text",
            "meta": {
                "test_field": 1,
            },
        },
        vector={
            "text-dense": [1.0, 0.0, 0.0, 0.0],
            "text-sparse": {"indices": [7, 1024, 367], "values": [0.1, 0.98, 0.33]},
        },
    )
    document = convert_qdrant_point_to_haystack_document(point, use_sparse_embeddings=True)
    assert "my-id" == document.id
    assert "Lorem ipsum" == document.content
    assert "text" == document.content_type
    assert {"indices": [7, 1024, 367], "values": [0.1, 0.98, 0.33]} == document.sparse_embedding.to_dict()
    assert {"test_field": 1} == document.meta
    assert 0.0 == np.sum(np.array([1.0, 0.0, 0.0, 0.0]) - document.embedding)


def test_point_to_document_reverts_proper_structure_from_record_without_sparse():
    point = rest.Record(
        id="c7c62e8e-02b9-4ec6-9f88-46bd97b628b7",
        payload={
            "id": "my-id",
            "id_hash_keys": ["content"],
            "content": "Lorem ipsum",
            "content_type": "text",
            "meta": {
                "test_field": 1,
            },
        },
        vector=[1.0, 0.0, 0.0, 0.0],
    )
    document = convert_qdrant_point_to_haystack_document(point, use_sparse_embeddings=False)
    assert "my-id" == document.id
    assert "Lorem ipsum" == document.content
    assert "text" == document.content_type
    assert document.sparse_embedding is None
    assert {"test_field": 1} == document.meta
    assert 0.0 == np.sum(np.array([1.0, 0.0, 0.0, 0.0]) - document.embedding)


def test_point_to_document_with_sparse_enabled_but_vector_none():
    point = rest.Record(
        id="c7c62e8e-02b9-4ec6-9f88-46bd97b628b7",
        payload={"id": "my-id", "content": "Lorem"},
        vector=None,
    )
    document = convert_qdrant_point_to_haystack_document(point, use_sparse_embeddings=True)
    assert document.embedding is None
    assert document.sparse_embedding is None


def test_point_to_document_preserves_score_from_scored_point():
    point = rest.ScoredPoint(
        id="c7c62e8e-02b9-4ec6-9f88-46bd97b628b7",
        payload={"id": "my-id", "content": "Lorem"},
        vector=[0.1, 0.2],
        score=0.75,
        version=0,
    )
    document = convert_qdrant_point_to_haystack_document(point, use_sparse_embeddings=False)
    assert document.score == 0.75


def test_convert_haystack_documents_to_qdrant_points_without_sparse():
    doc = Document(content="hello", embedding=[0.1, 0.2, 0.3])
    points = convert_haystack_documents_to_qdrant_points([doc], use_sparse_embeddings=False)
    assert len(points) == 1
    assert points[0].vector == [0.1, 0.2, 0.3]
    assert points[0].payload["content"] == "hello"
    assert "embedding" not in points[0].payload


def test_convert_haystack_documents_to_qdrant_points_without_sparse_without_embedding():
    doc = Document(content="hello")
    points = convert_haystack_documents_to_qdrant_points([doc], use_sparse_embeddings=False)
    assert points[0].vector == {}


def test_convert_haystack_documents_to_qdrant_points_with_sparse():
    sparse = SparseEmbedding(indices=[0, 5], values=[0.1, 0.7])
    doc = Document(content="hello", embedding=[0.1, 0.2], sparse_embedding=sparse)
    points = convert_haystack_documents_to_qdrant_points([doc], use_sparse_embeddings=True)
    assert points[0].vector[DENSE_VECTORS_NAME] == [0.1, 0.2]
    assert isinstance(points[0].vector[SPARSE_VECTORS_NAME], rest.SparseVector)
    assert points[0].vector[SPARSE_VECTORS_NAME].indices == [0, 5]
    assert points[0].vector[SPARSE_VECTORS_NAME].values == [0.1, 0.7]


def test_convert_haystack_documents_to_qdrant_points_with_sparse_only_dense():
    doc = Document(content="hello", embedding=[0.1, 0.2])
    points = convert_haystack_documents_to_qdrant_points([doc], use_sparse_embeddings=True)
    assert points[0].vector == {DENSE_VECTORS_NAME: [0.1, 0.2]}


def test_convert_haystack_documents_to_qdrant_points_with_sparse_no_vectors():
    doc = Document(content="hello")
    points = convert_haystack_documents_to_qdrant_points([doc], use_sparse_embeddings=True)
    assert points[0].vector == {}


@pytest.mark.parametrize(
    "vector",
    [
        {DENSE_VECTORS_NAME: [0.1, 0.2]},
        {DENSE_VECTORS_NAME: [0.1, 0.2], SPARSE_VECTORS_NAME: {"indices": [0], "values": [0.5]}},
    ],
    ids=["no_sparse_key", "sparse_value_not_sparse_vector_instance"],
)
def test_point_to_document_sparse_vector_edge_cases(vector):
    point = SimpleNamespace(id="x", payload={"id": "x", "content": "x"}, vector=vector)
    document = convert_qdrant_point_to_haystack_document(point, use_sparse_embeddings=True)
    assert document.embedding == [0.1, 0.2]
    assert document.sparse_embedding is None
