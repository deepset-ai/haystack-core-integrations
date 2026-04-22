# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from dataclasses import replace
from typing import Any, Literal

import boto3
from botocore.exceptions import ClientError
from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils.filters import document_matches_filter

from .filters import _normalize_filters, _validate_filters

logger = logging.getLogger(__name__)

# S3 Vectors allows up to 500 vectors per put_vectors call
_WRITE_BATCH_SIZE = 500

# S3 Vectors allows up to 100 keys per get_vectors call, 500 per delete_vectors call
_GET_BATCH_SIZE = 100
_DELETE_BATCH_SIZE = 500

# Maximum number of results from query_vectors
_MAX_TOP_K = 100

# S3 Vectors metadata limits
_MAX_TOTAL_METADATA_BYTES = 40 * 1024  # 40 KB total

# Reserved metadata keys used to store Haystack Document fields
_CONTENT_KEY = "_content"
_BLOB_DATA_KEY = "_blob_data"
_BLOB_META_KEY = "_blob_meta"
_BLOB_MIME_TYPE_KEY = "_blob_mime_type"

_RESERVED_META_KEYS = {_CONTENT_KEY, _BLOB_DATA_KEY, _BLOB_META_KEY, _BLOB_MIME_TYPE_KEY}

# These keys are stored but should not be used in query filters.
# They are configured as nonFilterableMetadataKeys on the index.
_NON_FILTERABLE_KEYS = [_CONTENT_KEY, _BLOB_DATA_KEY, _BLOB_META_KEY, _BLOB_MIME_TYPE_KEY]


class S3VectorsDocumentStore:
    """
    A Document Store using [Amazon S3 Vectors](https://aws.amazon.com/s3/features/vectors/).

    Amazon S3 Vectors provides serverless vector storage and similarity search within Amazon S3.
    This document store stores Haystack `Document` objects as vectors with associated metadata
    in an S3 vector bucket and index.

    **Service limits:**

    - Maximum `top_k`: 100 results per query
    - Maximum vector dimension: 4,096
    - Metadata per vector: 40 KB total, 2 KB filterable
    - All documents must have embeddings (`float32` only)
    - Distance metrics: `cosine` or `euclidean` (set at index creation, immutable)
    - `filter_documents()` is client-side — prefer `S3VectorsEmbeddingRetriever` with filters

    Usage example:
    ```python
    from haystack_integrations.document_stores.amazon_s3_vectors import S3VectorsDocumentStore

    document_store = S3VectorsDocumentStore(
        vector_bucket_name="my-vectors",
        index_name="my-index",
        dimension=768,
    )
    ```
    """

    def __init__(
        self,
        *,
        vector_bucket_name: str,
        index_name: str,
        dimension: int,
        distance_metric: Literal["cosine", "euclidean"] = "cosine",
        region_name: str | None = None,
        aws_access_key_id: Secret | None = None,
        aws_secret_access_key: Secret | None = None,
        aws_session_token: Secret | None = None,
        create_bucket_and_index: bool = True,
        non_filterable_metadata_keys: list[str] | None = None,
    ) -> None:
        """
        Create an S3VectorsDocumentStore instance.

        :param vector_bucket_name: Name of the S3 vector bucket.
        :param index_name: Name of the vector index within the bucket.
        :param dimension: Dimensionality of the embeddings (e.g. 768, 1536).
        :param distance_metric: Distance metric for similarity search: `"cosine"` or `"euclidean"`.
        :param region_name: AWS region. If not provided, uses the default from the environment/config.
        :param aws_access_key_id: AWS access key ID. If not provided, uses the default credential chain.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token for temporary credentials.
        :param create_bucket_and_index: Whether to automatically create the vector bucket and index
            if they do not exist. Defaults to `True`.
        :param non_filterable_metadata_keys: Additional metadata keys to mark as non-filterable
            on the index (beyond the internal keys used for Document content/blob storage).
        """
        self.vector_bucket_name = vector_bucket_name
        self.index_name = index_name
        self.dimension = dimension
        self.distance_metric = distance_metric
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.create_bucket_and_index = create_bucket_and_index
        self.non_filterable_metadata_keys = non_filterable_metadata_keys or []

        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily create and return the boto3 s3vectors client."""
        if self._client is not None:
            return self._client

        kwargs: dict[str, Any] = {"service_name": "s3vectors"}
        if self.region_name:
            kwargs["region_name"] = self.region_name
        if self.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.aws_access_key_id.resolve_value()
        if self.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self.aws_secret_access_key.resolve_value()
        if self.aws_session_token:
            kwargs["aws_session_token"] = self.aws_session_token.resolve_value()

        self._client = boto3.client(**kwargs)

        if self.create_bucket_and_index:
            self._ensure_bucket_and_index()

        return self._client

    def _ensure_bucket_and_index(self) -> None:
        """Create the vector bucket and index if they don't already exist."""
        client = self._client

        # Ensure bucket exists
        try:
            client.get_vector_bucket(vectorBucketName=self.vector_bucket_name)
            logger.info("Using existing vector bucket '{bucket}'.", bucket=self.vector_bucket_name)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NotFoundException":
                logger.info("Creating vector bucket '{bucket}'.", bucket=self.vector_bucket_name)
                client.create_vector_bucket(vectorBucketName=self.vector_bucket_name)
            else:
                raise

        # Ensure index exists
        all_non_filterable = list(set(_NON_FILTERABLE_KEYS + self.non_filterable_metadata_keys))
        try:
            client.get_index(vectorBucketName=self.vector_bucket_name, indexName=self.index_name)
            logger.info(
                "Using existing index '{index}' in bucket '{bucket}'. "
                "`dimension`, `distance_metric`, and `non_filterable_metadata_keys` will be ignored.",
                index=self.index_name,
                bucket=self.vector_bucket_name,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "NotFoundException":
                logger.info(
                    "Creating index '{index}' in bucket '{bucket}' (dimension={dim}, metric={metric}).",
                    index=self.index_name,
                    bucket=self.vector_bucket_name,
                    dim=self.dimension,
                    metric=self.distance_metric,
                )
                client.create_index(
                    vectorBucketName=self.vector_bucket_name,
                    indexName=self.index_name,
                    dataType="float32",
                    dimension=self.dimension,
                    distanceMetric=self.distance_metric,
                    metadataConfiguration={"nonFilterableMetadataKeys": all_non_filterable},
                )
            else:
                raise

    def to_dict(self) -> dict[str, Any]:
        """Serialize this document store to a dictionary."""
        return default_to_dict(
            self,
            vector_bucket_name=self.vector_bucket_name,
            index_name=self.index_name,
            dimension=self.dimension,
            distance_metric=self.distance_metric,
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id.to_dict() if self.aws_access_key_id else None,
            aws_secret_access_key=self.aws_secret_access_key.to_dict() if self.aws_secret_access_key else None,
            aws_session_token=self.aws_session_token.to_dict() if self.aws_session_token else None,
            create_bucket_and_index=self.create_bucket_and_index,
            non_filterable_metadata_keys=self.non_filterable_metadata_keys,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "S3VectorsDocumentStore":
        """Deserialize a document store from a dictionary."""
        deserialize_secrets_inplace(
            data["init_parameters"],
            keys=["aws_access_key_id", "aws_secret_access_key", "aws_session_token"],
        )
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Return the number of documents in the document store.

        .. note::

            S3 Vectors does not provide a dedicated count API. This method lists all vector keys
            via pagination, which can be slow for large indexes.
        """
        client = self._get_client()
        count = 0
        next_token = None
        while True:
            kwargs: dict[str, Any] = {
                "vectorBucketName": self.vector_bucket_name,
                "indexName": self.index_name,
            }
            if next_token:
                kwargs["nextToken"] = next_token
            response = client.list_vectors(**kwargs)
            count += len(response.get("vectors", []))
            next_token = response.get("nextToken")
            if not next_token:
                break
        return count

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE) -> int:
        """
        Write Documents to the S3 Vectors index.

        All documents must have an embedding set. S3 Vectors `put_vectors` is an upsert operation
        by default, so `DuplicatePolicy.OVERWRITE` is the natural behavior.
        `DuplicatePolicy.SKIP` will check for existing documents first (slower).
        `DuplicatePolicy.NONE` will raise an error if a document already exists.

        Metadata per vector is limited to 40 KB total (2 KB filterable).

        :param documents: A list of Documents to write. Each document must have an embedding.
        :param policy: The duplicate policy. Defaults to `DuplicatePolicy.OVERWRITE`.
        :returns: The number of documents written.
        """
        if len(documents) == 0:
            return 0

        client = self._get_client()
        written = 0

        for i in range(0, len(documents), _WRITE_BATCH_SIZE):
            batch = documents[i : i + _WRITE_BATCH_SIZE]

            # Validate embeddings upfront
            for doc in batch:
                if doc.embedding is None:
                    msg = f"Document '{doc.id}' has no embedding. S3VectorsDocumentStore requires embeddings."
                    raise DocumentStoreError(msg)

            # Batch-check for existing documents when needed
            existing_ids: set[str] = set()
            if policy in (DuplicatePolicy.SKIP, DuplicatePolicy.NONE):
                batch_ids = [doc.id for doc in batch]
                for j in range(0, len(batch_ids), _GET_BATCH_SIZE):
                    id_chunk = batch_ids[j : j + _GET_BATCH_SIZE]
                    response = client.get_vectors(
                        vectorBucketName=self.vector_bucket_name,
                        indexName=self.index_name,
                        keys=id_chunk,
                    )
                    for v in response.get("vectors", []):
                        existing_ids.add(v["key"])

                if policy == DuplicatePolicy.NONE and existing_ids:
                    msg = (
                        f"Document(s) {sorted(existing_ids)} already exist in the document store. "
                        "Use DuplicatePolicy.OVERWRITE or DuplicatePolicy.SKIP."
                    )
                    raise DocumentStoreError(msg)

            vectors_to_write = []
            for doc in batch:
                if policy == DuplicatePolicy.SKIP and doc.id in existing_ids:
                    continue
                vectors_to_write.append(self._document_to_s3_vector(doc))

            if vectors_to_write:
                client.put_vectors(
                    vectorBucketName=self.vector_bucket_name,
                    indexName=self.index_name,
                    vectors=vectors_to_write,
                )
                written += len(vectors_to_write)

        return written

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Return documents matching the provided filters.

        .. warning::

            S3 Vectors only supports metadata filtering during vector similarity queries, not as a
            standalone operation. This method lists all vectors and applies filters client-side,
            which can be very slow for large indexes. For filtered retrieval, prefer using
            `S3VectorsEmbeddingRetriever` with filters instead.

        :param filters: Haystack-format filters to apply.
        :returns: A list of matching Documents.
        """
        if filters:
            logger.warning(
                "S3 Vectors does not support standalone filtered listing. "
                "filter_documents() will fetch ALL vectors and apply filters client-side, "
                "which can be very slow for large indexes. "
                "Prefer using S3VectorsEmbeddingRetriever with filters for efficient filtered retrieval."
            )

        client = self._get_client()

        # list_vectors supports returnData and returnMetadata directly,
        # so we can read documents in a single paginated pass without
        # a separate get_vectors round-trip.
        documents: list[Document] = []
        next_token = None
        while True:
            kwargs: dict[str, Any] = {
                "vectorBucketName": self.vector_bucket_name,
                "indexName": self.index_name,
                "returnData": True,
                "returnMetadata": True,
            }
            if next_token:
                kwargs["nextToken"] = next_token
            response = client.list_vectors(**kwargs)
            for v in response.get("vectors", []):
                documents.append(self._s3_vector_to_document(v))
            next_token = response.get("nextToken")
            if not next_token:
                break

        if filters:
            _validate_filters(filters)
            documents = [doc for doc in documents if document_matches_filter(filters=filters, document=doc)]

        return documents

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Delete documents by their IDs.

        :param document_ids: List of document IDs to delete.
        """
        if not document_ids:
            return

        client = self._get_client()
        for i in range(0, len(document_ids), _DELETE_BATCH_SIZE):
            batch = document_ids[i : i + _DELETE_BATCH_SIZE]
            client.delete_vectors(
                vectorBucketName=self.vector_bucket_name,
                indexName=self.index_name,
                keys=batch,
            )

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """
        Retrieve documents most similar to the query embedding.

        This method is not part of the public interface.
        Use `S3VectorsEmbeddingRetriever` instead.

        :param query_embedding: The query embedding vector.
        :param filters: Optional Haystack-format metadata filters.
        :param top_k: Maximum number of results to return. S3 Vectors caps this at 100.
        :returns: List of Documents sorted by similarity. Returned documents will not contain
            embeddings (S3 Vectors `query_vectors` does not return vector data).
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        _validate_filters(filters)
        s3_filter = _normalize_filters(filters) if filters else None

        if top_k > _MAX_TOP_K:
            logger.warning(
                "Requested top_k={top_k} exceeds S3 Vectors maximum of {max_k}. Results will be capped.",
                top_k=top_k,
                max_k=_MAX_TOP_K,
            )

        client = self._get_client()

        query_kwargs: dict[str, Any] = {
            "vectorBucketName": self.vector_bucket_name,
            "indexName": self.index_name,
            "topK": min(top_k, _MAX_TOP_K),  # S3 Vectors caps at 100
            "queryVector": {"float32": query_embedding},
            "returnMetadata": True,
            "returnDistance": True,
        }
        if s3_filter:
            query_kwargs["filter"] = s3_filter

        result = client.query_vectors(**query_kwargs)

        # Convert distance to a score.
        # For cosine: S3 Vectors returns cosine *distance* (0 = identical, 2 = opposite).
        # Haystack convention is higher score = more similar, so we convert: score = 1 - distance.
        # For euclidean: we negate the distance so that closer vectors score higher.
        distance_metric = result.get("distanceMetric", self.distance_metric)

        documents = []
        for v in result.get("vectors", []):
            doc = self._s3_vector_to_document(v)

            # Compute score from distance
            score = None
            raw_distance = v.get("distance")
            if raw_distance is not None:
                if distance_metric == "cosine":
                    score = 1.0 - raw_distance
                else:
                    # euclidean: negate so higher = more similar
                    score = -raw_distance

            # query_vectors does not return vector data; attach score
            documents.append(replace(doc, embedding=None, score=score))

        return documents

    @staticmethod
    def _document_to_s3_vector(doc: Document) -> dict[str, Any]:
        """Convert a Haystack Document to an S3 Vectors vector entry."""
        metadata: dict[str, Any] = {}

        # Store content as non-filterable metadata
        if doc.content is not None:
            metadata[_CONTENT_KEY] = doc.content

        # Store blob fields
        if doc.blob is not None:
            metadata[_BLOB_DATA_KEY] = base64.b64encode(doc.blob.data).decode("ascii")
            if doc.blob.meta:
                metadata[_BLOB_META_KEY] = doc.blob.meta
            if doc.blob.mime_type:
                metadata[_BLOB_MIME_TYPE_KEY] = doc.blob.mime_type

        # Store user metadata
        if doc.meta:
            for key, value in doc.meta.items():
                if key in _RESERVED_META_KEYS:
                    logger.warning(
                        "Metadata key '{key}' is reserved; user value will be ignored.",
                        key=key,
                    )
                    continue
                metadata[key] = value

        # Warn if metadata is likely too large
        try:
            meta_size = len(json.dumps(metadata).encode("utf-8"))
            if meta_size > _MAX_TOTAL_METADATA_BYTES:
                logger.warning(
                    "Document '{doc_id}' has ~{size} bytes of metadata, exceeding the S3 Vectors "
                    "limit of {limit} bytes. The put_vectors call may fail.",
                    doc_id=doc.id,
                    size=meta_size,
                    limit=_MAX_TOTAL_METADATA_BYTES,
                )
        except (TypeError, ValueError):
            pass  # Best-effort size check

        return {
            "key": doc.id,
            "data": {"float32": doc.embedding},
            "metadata": metadata,
        }

    @staticmethod
    def _s3_vector_to_document(vector: dict[str, Any]) -> Document:
        """Convert an S3 Vectors vector response to a Haystack Document."""
        metadata = dict(vector.get("metadata", {}))
        content = metadata.pop(_CONTENT_KEY, None)
        blob_data = metadata.pop(_BLOB_DATA_KEY, None)
        blob_meta = metadata.pop(_BLOB_META_KEY, None)
        blob_mime_type = metadata.pop(_BLOB_MIME_TYPE_KEY, None)

        blob = None
        if blob_data is not None:
            blob = ByteStream(
                data=base64.b64decode(blob_data) if isinstance(blob_data, str) else blob_data,
                meta=blob_meta or {},
                mime_type=blob_mime_type,
            )

        embedding = None
        data = vector.get("data", {})
        if "float32" in data:
            embedding = data["float32"]

        return Document(
            id=vector["key"],
            content=content,
            meta=metadata,
            embedding=embedding,
            blob=blob,
        )
