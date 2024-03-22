import logging
import uuid
from typing import List, Union

from haystack.dataclasses import Document
from qdrant_client.http import models as rest

logger = logging.getLogger(__name__)

DENSE_VECTORS_NAME = "text-dense"
SPARSE_VECTORS_NAME = "text-sparse"


class HaystackToQdrant:
    """A converter from Haystack to Qdrant types."""

    UUID_NAMESPACE = uuid.UUID("3896d314-1e95-4a3a-b45a-945f9f0b541d")

    def documents_to_batch(
        self,
        documents: List[Document],
        *,
        embedding_field: str,
        sparse_embedding_field: str,
    ) -> List[rest.PointStruct]:
        points = []
        for document in documents:
            payload = document.to_dict(flatten=False)
            vector = {}
            if embedding_field in payload and payload[embedding_field] is not None:
                dense_vector = payload.pop(embedding_field) or []
                vector[DENSE_VECTORS_NAME] = dense_vector
            if (
                sparse_embedding_field in payload
                and payload[sparse_embedding_field] is not None
                and payload[sparse_embedding_field] != ""
            ):
                sparse_vector = payload.pop(sparse_embedding_field, {"indices": [], "values": []})
                sparse_vector_instance = rest.SparseVector(**sparse_vector)
                vector[SPARSE_VECTORS_NAME] = sparse_vector_instance
            _id = self.convert_id(payload.get("id"))

            point = rest.PointStruct(
                payload=payload,
                vector=vector,
                id=_id,
            )
            points.append(point)
        return points

    def convert_id(self, _id: str) -> str:
        """
        Converts any string into a UUID-like format in a deterministic way.

        Qdrant does not accept any string as an id, so an internal id has to be
        generated for each point. This is a deterministic way of doing so.
        """
        return uuid.uuid5(self.UUID_NAMESPACE, _id).hex


QdrantPoint = Union[rest.ScoredPoint, rest.Record]


class QdrantToHaystack:
    def __init__(self, content_field: str, name_field: str, embedding_field: str, sparse_embedding_field: str):
        self.content_field = content_field
        self.name_field = name_field
        self.embedding_field = embedding_field
        self.sparse_embedding_field = sparse_embedding_field

    def point_to_document(self, point: QdrantPoint) -> Document:
        payload = {**point.payload}
        if hasattr(point, "vector") and point.vector is not None and DENSE_VECTORS_NAME in point.vector:
            payload["embedding"] = point.vector[DENSE_VECTORS_NAME]
        else:
            payload["embedding"] = None
        payload["score"] = point.score if hasattr(point, "score") else None
        if hasattr(point, "vector") and point.vector is not None and SPARSE_VECTORS_NAME in point.vector:
            parse_vector_dict = {
                "indices": point.vector[SPARSE_VECTORS_NAME].indices,
                "values": point.vector[SPARSE_VECTORS_NAME].values,
            }
            payload["sparse_embedding"] = parse_vector_dict
        else:
            payload["sparse_embedding"] = None
        return Document.from_dict(payload)
