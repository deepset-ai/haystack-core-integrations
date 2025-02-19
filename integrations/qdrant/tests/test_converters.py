import numpy as np
from haystack import Document
from pandas import DataFrame
from qdrant_client.http import models as rest

from haystack_integrations.document_stores.qdrant.converters import (
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


def test_point_to_document_skips_dataframe():

    point = rest.Record(
        id="c7c62e8e-02b9-4ec6-9f88-46bd97b628b7",
        payload={
            "id": "my-id",
            "content": "Lorem ipsum",
            "content_type": "text",
            "meta": {
                "test_field": 1,
            },
            "dataframe": {"a": [1, 2, 3]},
        },
        vector=[1.0, 0.0, 0.0, 0.0],
    )
    document = convert_qdrant_point_to_haystack_document(point, use_sparse_embeddings=False)
    assert "my-id" == document.id
    assert "Lorem ipsum" == document.content
    assert "text" == document.content_type
    assert {"test_field": 1} == document.meta
    assert 0.0 == np.sum(np.array([1.0, 0.0, 0.0, 0.0]) - document.embedding)
    assert not hasattr(document, "dataframe") or document.dataframe is None


def test_documents_to_points_skips_dataframe():
    doc = Document(
        id="my-id",
        content="Lorem ipsum",
        embedding=[1.0, 0.0, 0.0, 0.0],
    )

    doc.dataframe = DataFrame([[1, 2], [3, 4]])

    points = convert_haystack_documents_to_qdrant_points([doc], use_sparse_embeddings=False)
    assert len(points) == 1

    assert points[0].payload["content"] == "Lorem ipsum"
    assert points[0].vector == [1.0, 0.0, 0.0, 0.0]
    assert "dataframe" not in points[0].payload
