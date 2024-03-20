import numpy as np
import pytest
from haystack_integrations.document_stores.qdrant.converters import HaystackToQdrant, QdrantToHaystack
from qdrant_client.http import models as rest

CONTENT_FIELD = "content"
NAME_FIELD = "name"
EMBEDDING_FIELD = "vector"
SPARSE_EMBEDDING_FIELD = "sparse-vector"


@pytest.fixture
def haystack_to_qdrant() -> HaystackToQdrant:
    return HaystackToQdrant()


@pytest.fixture
def qdrant_to_haystack() -> QdrantToHaystack:
    return QdrantToHaystack(
        content_field=CONTENT_FIELD,
        name_field=NAME_FIELD,
        embedding_field=EMBEDDING_FIELD,
        sparse_embedding_field=SPARSE_EMBEDDING_FIELD
    )


def test_convert_id_is_deterministic(haystack_to_qdrant: HaystackToQdrant):
    first_id = haystack_to_qdrant.convert_id("test-id")
    second_id = haystack_to_qdrant.convert_id("test-id")
    assert first_id == second_id


def test_point_to_document_reverts_proper_structure_from_record(
    qdrant_to_haystack: QdrantToHaystack,
):
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
    document = qdrant_to_haystack.point_to_document(point)
    assert "my-id" == document.id
    assert "Lorem ipsum" == document.content
    assert "text" == document.content_type
    assert {"indices": [7, 1024, 367], "values": [0.1, 0.98, 0.33]} == document.sparse_embedding.to_dict()
    assert {"test_field": 1} == document.meta
    assert 0.0 == np.sum(np.array([1.0, 0.0, 0.0, 0.0]) - document.embedding)
