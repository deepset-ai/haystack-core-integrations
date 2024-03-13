import uuid
from typing import List, Union

from haystack.dataclasses import Document
from qdrant_client.http import models as rest


class HaystackToQdrant:
    """A converter from Haystack to Qdrant types."""

    UUID_NAMESPACE = uuid.UUID("3896d314-1e95-4a3a-b45a-945f9f0b541d")

    def documents_to_batch(
        self,
        documents: List[Document],
        *,
        embedding_field: str,
    ) -> List[rest.PointStruct]:
        points = []
        for document in documents:
            payload = document.to_dict(flatten=False)
            # TODO: vector should be built not only from embedding_field but also from the field containing sparse embeddings
            # TODO: Because with sparse vectors, the vector is now a dict
            vector = payload.pop(embedding_field) or {}
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
    def __init__(self, content_field: str, name_field: str, embedding_field: str):
        self.content_field = content_field
        self.name_field = name_field
        self.embedding_field = embedding_field

    def point_to_document(self, point: QdrantPoint) -> Document:
        payload = {**point.payload}
        # TODO: rework the converters part because now it's a mess with the new sparse embedding feature
        # TODO: With dense+sparse embedding, vector is now a dict ?
        # TODO: Unnamed dense vector are accessed with "" key ?
        payload["embedding"] = point.vector[""] if hasattr(point, "vector") else None
        payload["score"] = point.score if hasattr(point, "score") else None
        # TODO: Because haystack document don't have sparse embedding field (only dense) in their dataclass, put it in meta ?
        payload["meta"]["sparse-embedding"] = point.vector["text-sparse"] if hasattr(point, "vector") else None
        return Document.from_dict(payload)
