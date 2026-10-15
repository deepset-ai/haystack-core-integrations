# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import math
from collections.abc import Iterator
from dataclasses import replace
from typing import Any, Literal

import boto3
from botocore.exceptions import ClientError
from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils.filters import document_matches_filter

from .filters import _normalize_filters, _validate_filters

logger = logging.getLogger(__name__)

# S3 Vectors allows up to 500 vectors per put_vectors call
_WRITE_BATCH_SIZE = 500

# S3 Vectors allows up to 500 keys per delete_vectors call
_DELETE_BATCH_SIZE = 500

# Maximum top_k of a query_vectors request. Results come back in pages of up to 100.
_MAX_TOP_K = 10_000

# S3 Vectors metadata limits
_MAX_TOTAL_METADATA_BYTES = 40 * 1024  # 40 KB total

# Reserved metadata key used to store the Document content
_CONTENT_KEY = "_content"

_RESERVED_META_KEYS = {_CONTENT_KEY}

# These keys are stored but should not be used in query filters.
# They are configured as nonFilterableMetadataKeys on the index.
_NON_FILTERABLE_KEYS = [_CONTENT_KEY]


class S3VectorsDocumentStore:
    """
    A Document Store using [Amazon S3 Vectors](https://aws.amazon.com/s3/features/vectors/).

    Amazon S3 Vectors provides serverless vector storage and similarity search within Amazon S3.
    This document store stores Haystack `Document` objects as vectors with associated metadata
    in an S3 vector bucket and index.

    Service limits:

    - Maximum `top_k`: 10,000 results per query. Search is approximate, so a `top_k` close to the number of
      stored or matching documents can return fewer results
    - Maximum vector dimension: 4,096
    - Metadata per vector: 40 KB total, 2 KB filterable
    - A metadata value must be a string, a boolean, a finite number, or a non-empty array of either
      all strings or all non-boolean numbers. Anything else (`None`, nested objects, empty or mixed
      arrays) is dropped with a warning rather than failing the write
    - Every record needs a vector (`float32` only), so documents written without an embedding are
      stored with a placeholder vector (stripped again on read) and are only meaningfully
      retrievable via `filter_documents()`
    - Only `DuplicatePolicy.OVERWRITE` is supported, since `put_vectors` is an upsert
    - Distance metrics: `cosine` or `euclidean` (set at index creation, immutable)
    - `filter_documents()` filters locally, so prefer `S3VectorsEmbeddingRetriever` with filters
    - `Document.blob` is not stored

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
            on the index (beyond the internal key used for Document content).
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

        # S3 Vectors requires a vector on every record, and a cosine index rejects zero-norm
        # vectors, so documents written without an embedding get this non-zero placeholder.
        self._dummy_vector = [-10.0] * dimension

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

    def _iter_vectors(self, *, return_data: bool = False, return_metadata: bool = False) -> Iterator[dict[str, Any]]:
        """
        Yield every vector in the index, paginating through `list_vectors`.

        :param return_data: Whether to ask S3 Vectors for the vector data.
        :param return_metadata: Whether to ask S3 Vectors for the vector metadata.
        """
        client = self._get_client()
        next_token = None
        while True:
            kwargs: dict[str, Any] = {
                "vectorBucketName": self.vector_bucket_name,
                "indexName": self.index_name,
            }
            if return_data:
                kwargs["returnData"] = True
            if return_metadata:
                kwargs["returnMetadata"] = True
            if next_token:
                kwargs["nextToken"] = next_token
            response = client.list_vectors(**kwargs)
            yield from response.get("vectors", [])
            next_token = response.get("nextToken")
            if not next_token:
                break

    def count_documents(self) -> int:
        """
        Return the number of documents in the document store.

        Note: S3 Vectors does not provide a dedicated count API. This method lists all vector keys
        via pagination, which can be slow for large indexes.
        """
        return sum(1 for _ in self._iter_vectors())

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE) -> int:
        """
        Write Documents to the S3 Vectors index.

        `S3VectorsDocumentStore` only supports `DuplicatePolicy.OVERWRITE`: `put_vectors` is an
        upsert, and S3 Vectors offers no cheap existence check (only `get_vectors` on batches of
        100 keys, or a full index scan), so enforcing any other policy would cost an extra read
        pass on every write. Other policies are ignored with a warning.

        Documents without an embedding are stored with a placeholder vector, since S3 Vectors
        requires a vector on every record. `filter_documents()` strips the placeholder, so they read
        back with `embedding=None`, but they are meaningless as similarity-search results.

        Metadata per vector is limited to 40 KB total (2 KB filterable).

        :param documents: A list of Documents to write.
        :param policy: The duplicate policy. Only `DuplicatePolicy.OVERWRITE` is supported.
        :returns: The number of documents written.
        :raises ValueError: If `documents` is not a list of `Document` instances.
        """
        # Validate input type up front, before any writes happen.
        if not isinstance(documents, list) or any(not isinstance(d, Document) for d in documents):
            msg = "Please provide a list of Documents."
            raise ValueError(msg)

        if len(documents) == 0:
            return 0

        # DuplicatePolicy.NONE means "the store decides", so it needs no warning.
        if policy not in (DuplicatePolicy.NONE, DuplicatePolicy.OVERWRITE):
            logger.warning(
                "S3VectorsDocumentStore only supports DuplicatePolicy.OVERWRITE because put_vectors is an "
                "upsert. Ignoring policy={policy} and overwriting.",
                policy=str(policy),
            )

        # Report every embedding-less document in one pass, before any put_vectors call, so the
        # warning describes the whole write rather than arriving batch by batch.
        missing_embedding_ids = [doc.id for doc in documents if doc.embedding is None]
        if missing_embedding_ids:
            logger.warning(
                "Document(s) {ids} have no embedding and will be stored with a placeholder vector. "
                "They are retrievable with filter_documents() but will pollute similarity search results.",
                ids=missing_embedding_ids,
            )

        client = self._get_client()
        written = 0

        for i in range(0, len(documents), _WRITE_BATCH_SIZE):
            batch = documents[i : i + _WRITE_BATCH_SIZE]
            vectors_to_write = [
                self._document_to_s3_vector(doc, fallback_embedding=self._dummy_vector) for doc in batch
            ]
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

        Note: S3 Vectors only supports metadata filtering during vector similarity queries, not as a
        standalone operation. This method lists all vectors and filters them locally with Haystack,
        which can be slow for large indexes. `S3VectorsEmbeddingRetriever` filters through S3
        Vectors instead, and the two can match different documents for the same filter, especially
        on array metadata: S3 Vectors matches an array field when any element matches, while
        Haystack compares the whole array.

        :param filters: Haystack-format filters to apply.
        :returns: A list of matching Documents.
        """
        _validate_filters(filters)

        if filters:
            logger.warning(
                "S3 Vectors does not support standalone filtered listing. "
                "filter_documents() will fetch ALL vectors and apply filters client-side, "
                "which can be very slow for large indexes. "
                "Prefer using S3VectorsEmbeddingRetriever with filters for efficient filtered retrieval."
            )

        # list_vectors supports returnData and returnMetadata directly,
        # so we can read documents in a single paginated pass without
        # a separate get_vectors round-trip.
        documents = [
            self._s3_vector_to_document(v, placeholder_embedding=self._dummy_vector)
            for v in self._iter_vectors(return_data=True, return_metadata=True)
        ]

        if filters:
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

    def delete_all_documents(self) -> None:
        """
        Delete all documents from the index.

        Note: S3 Vectors has no truncate operation, so every key is listed and then deleted in
        batches, which can be slow for large indexes.
        """
        self.delete_documents([v["key"] for v in self._iter_vectors()])

    def close(self) -> None:
        """
        Close the underlying boto3 client and release its connection pool.

        The store can be used again afterwards: the next call lazily creates a new client.
        """
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                # Closing is best-effort; a failure here must not mask the caller's own work.
                logger.debug("Ignoring error while closing the S3 Vectors client.")
            self._client = None

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
        :param top_k: Maximum number of results to return. S3 Vectors caps this at 10,000.
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
            "topK": min(top_k, _MAX_TOP_K),
            "queryVector": {"float32": query_embedding},
            "returnMetadata": True,
            "returnDistance": True,
        }
        if s3_filter:
            query_kwargs["filter"] = s3_filter

        # query_vectors returns up to 100 results per page.
        vectors: list[dict[str, Any]] = []
        while True:
            result = client.query_vectors(**query_kwargs)
            vectors.extend(result.get("vectors", []))
            next_token = result.get("nextToken")
            if not next_token:
                break
            query_kwargs["nextToken"] = next_token

        # Convert distance to a score.
        # For cosine: S3 Vectors returns cosine *distance* (0 = identical, 2 = opposite).
        # Haystack convention is higher score = more similar, so we convert: score = 1 - distance.
        # For euclidean: we negate the distance so that closer vectors score higher.
        distance_metric = result.get("distanceMetric", self.distance_metric)

        documents = []
        for v in vectors:
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
    def _document_to_s3_vector(doc: Document, *, fallback_embedding: list[float] | None = None) -> dict[str, Any]:
        """
        Convert a Haystack Document to an S3 Vectors vector entry.

        :param doc: The Document to convert.
        :param fallback_embedding: Vector to send when `doc` has no embedding. S3 Vectors requires
            a vector on every record.
        """
        metadata: dict[str, Any] = {}

        # Store content as non-filterable metadata
        if doc.content is not None:
            metadata[_CONTENT_KEY] = doc.content

        if doc.blob is not None:
            logger.warning(
                "Document '{doc_id}' has a blob, which S3VectorsDocumentStore does not store.", doc_id=doc.id
            )

        # Store user metadata
        dropped: list[str] = []
        if doc.meta:
            for key, value in doc.meta.items():
                if key in _RESERVED_META_KEYS:
                    logger.warning(
                        "Metadata key '{key}' is reserved; user value will be ignored.",
                        key=key,
                    )
                    continue
                if not S3VectorsDocumentStore._is_supported_metadata_value(value):
                    dropped.append(key)
                    continue
                # A tuple is an array as far as S3 Vectors is concerned, but botocore's document-type
                # validator only accepts `list`.
                metadata[key] = list(value) if isinstance(value, tuple) else value

        if dropped:
            logger.warning(
                "Document '{doc_id}': metadata key(s) {keys} have values S3 Vectors cannot store and "
                "were dropped. Metadata values must be a string, a boolean, a finite number, or a "
                "non-empty array of either all strings or all non-boolean numbers.",
                doc_id=doc.id,
                keys=sorted(dropped),
            )

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
            "data": {"float32": doc.embedding if doc.embedding is not None else fallback_embedding},
            "metadata": metadata,
        }

    @staticmethod
    def _is_supported_metadata_value(value: Any) -> bool:
        """
        Return whether S3 Vectors can store `value` as a metadata value.

        The accepted set was established against the live service: a string (empty is fine), a
        boolean, a finite number (ints of any magnitude), or a non-empty array whose elements are
        *either* all strings or all non-boolean numbers. Everything else is rejected with a
        `ValidationException` — `None`, dicts, empty arrays, nested arrays, arrays mixing strings and
        numbers, arrays containing booleans, and NaN/infinity.

        This matters in practice: `DocumentSplitter` with `split_overlap` writes a `_split_overlap`
        list of dicts, so without this check a plain splitter-to-store pipeline would fail.
        """
        # bool first: it is a subclass of int, and unlike numbers it is allowed only outside arrays.
        if isinstance(value, (bool, str)):
            return True
        if isinstance(value, int):
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, (list, tuple)):
            if not value:
                return False
            if all(isinstance(item, str) for item in value):
                return True
            return all(
                not isinstance(item, bool) and isinstance(item, (int, float)) and math.isfinite(item) for item in value
            )
        return False

    @staticmethod
    def _s3_vector_to_document(vector: dict[str, Any], *, placeholder_embedding: list[float] | None = None) -> Document:
        """
        Convert an S3 Vectors vector response to a Haystack Document.

        :param vector: The vector as returned by `list_vectors` or `get_vectors`.
        :param placeholder_embedding: Vector that stands for "no embedding". A stored vector equal to
            it reads back as `embedding=None`, so a document written without an embedding round-trips
            as one. See `write_documents`.
        """
        metadata = dict(vector.get("metadata", {}))
        content = metadata.pop(_CONTENT_KEY, None)

        embedding = None
        data = vector.get("data", {})
        if "float32" in data:
            embedding = data["float32"]
            # A placeholder vector means the document was written without an embedding; don't hand
            # the caller a vector it never supplied.
            if placeholder_embedding is not None and embedding == placeholder_embedding:
                embedding = None

        return Document(
            id=vector["key"],
            content=content,
            meta=metadata,
            embedding=embedding,
        )
