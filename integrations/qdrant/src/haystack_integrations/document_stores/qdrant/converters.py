import uuid

from haystack import logging
from haystack.dataclasses import Document
from qdrant_client.http import models as rest

logger = logging.getLogger(__name__)

DENSE_VECTORS_NAME = "text-dense"
SPARSE_VECTORS_NAME = "text-sparse"
_MULTI_VECTOR_NAME = "multi-vector"


UUID_NAMESPACE = uuid.UUID("3896d314-1e95-4a3a-b45a-945f9f0b541d")


def convert_haystack_documents_to_qdrant_points(
    documents: list[Document],
    *,
    use_sparse_embeddings: bool,
    use_multi_vector_embeddings: bool = False,
) -> list[rest.PointStruct]:
    """Convert a list of Haystack Document objects to Qdrant PointStruct objects."""
    points = []
    for document in documents:
        payload = document.to_dict(flatten=False)
        multi_vector = payload.pop("multi_vector_embedding", None)
        if multi_vector is not None and not use_multi_vector_embeddings:
            msg = "Set multi_vector_embedding_dim on the Document Store before writing multivector embeddings."
            raise ValueError(msg)

        if use_sparse_embeddings or use_multi_vector_embeddings:
            vector = {}

            dense_vector = payload.pop("embedding", None)
            if dense_vector is not None:
                vector[DENSE_VECTORS_NAME] = dense_vector

            if use_sparse_embeddings:
                sparse_vector = payload.pop("sparse_embedding", None)
                if sparse_vector is not None:
                    vector[SPARSE_VECTORS_NAME] = rest.SparseVector(**sparse_vector)
            if multi_vector is not None:
                vector[_MULTI_VECTOR_NAME] = multi_vector

        else:
            vector = payload.pop("embedding") or {}
        _id = convert_id(document.id)

        point = rest.PointStruct(
            payload=payload,
            vector=vector,
            id=_id,
        )
        points.append(point)
    return points


def convert_id(_id: str) -> str:
    """
    Converts any string into a UUID-like format in a deterministic way.

    Qdrant does not accept any string as an id, so an internal id has to be
    generated for each point. This is a deterministic way of doing so.
    """
    return uuid.uuid5(UUID_NAMESPACE, _id).hex


QdrantPoint = rest.ScoredPoint | rest.Record


def convert_qdrant_point_to_haystack_document(point: QdrantPoint, use_sparse_embeddings: bool) -> Document:
    """Convert a Qdrant ScoredPoint or Record to a Haystack Document object."""
    payload = dict(point.payload or {})
    payload["score"] = point.score if hasattr(point, "score") else None

    if not use_sparse_embeddings and not isinstance(point.vector, dict):
        payload["embedding"] = point.vector if hasattr(point, "vector") else None
    elif hasattr(point, "vector") and point.vector is not None and isinstance(point.vector, dict):
        payload["embedding"] = point.vector.get(DENSE_VECTORS_NAME)
        payload["multi_vector_embedding"] = point.vector.get(_MULTI_VECTOR_NAME)

        if SPARSE_VECTORS_NAME in point.vector:
            sparse_vector = point.vector[SPARSE_VECTORS_NAME]
            if isinstance(sparse_vector, rest.SparseVector):
                sparse_vector_dict = {
                    "indices": sparse_vector.indices,
                    "values": sparse_vector.values,
                }
                payload["sparse_embedding"] = sparse_vector_dict

    return Document.from_dict(payload)
