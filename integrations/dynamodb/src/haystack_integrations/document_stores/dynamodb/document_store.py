# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import time
from collections.abc import Iterator
from typing import Any

import boto3
from botocore.exceptions import ClientError
from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils.filters import document_matches_filter

logger = logging.getLogger(__name__)

# DynamoDB `SearchVectors` returns a `Score` whose meaning depends on the index's distance
# function. For a COSINE index the score is a *distance*: per the AWS docs it ranges from 0
# (identical) to 2 (opposite), and DynamoDB returns the k *smallest* scores — lower means more
# similar. Haystack's `Document.score` uses the opposite convention (higher = more relevant), so
# the distance is converted to a similarity below. Verified on real AWS: an identical vector
# scored 0.0 and an orthogonal one scored 1.0.
_COSINE_MAX_DISTANCE = 2.0

# Hard, non-adjustable service quota on `TopK` per `SearchVectors` request; see "Vector indexes" in
# https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/ServiceQuotas.html
SEARCH_VECTORS_MAX_TOP_K = 100
_VECTOR_ATTRIBUTE = "embedding"
# `id` is referenced through a placeholder in expressions so it can never clash with DynamoDB reserved words.
_ID_PLACEHOLDER = {"#id": "id"}


class DynamoDBDocumentStore:
    """
    A Haystack DocumentStore backed by Amazon DynamoDB native vector search.

    Uses the `SearchVectors` API (GA 2026-08-05). Documents are stored as items in a
    DynamoDB table with a vector index, and retrieved via cosine similarity search.

    Example usage:

    ```python
    from haystack_integrations.document_stores.dynamodb import DynamoDBDocumentStore

    store = DynamoDBDocumentStore(
        table_name="haystack-documents",
        index_name="haystack-vector-index",
        embedding_dimension=768,
        region_name="us-east-1",
    )
    ```
    """

    def __init__(
        self,
        *,
        table_name: str = "haystack_documents",
        index_name: str = "haystack_vector_index",
        embedding_dimension: int = 768,
        region_name: str | None = None,
        aws_access_key_id: Secret = Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False),
        aws_secret_access_key: Secret = Secret.from_env_var("AWS_SECRET_ACCESS_KEY", strict=False),
        aws_session_token: Secret = Secret.from_env_var("AWS_SESSION_TOKEN", strict=False),
        create_table_if_not_exists: bool = True,
        similarity_function: str = "cosine",
    ) -> None:
        """
        Creates a new DynamoDBDocumentStore instance.

        :param table_name: Name of the DynamoDB table to store documents in. Created if it
            does not exist and `create_table_if_not_exists` is `True`.
        :param index_name: Name of the vector index on the table.
        :param embedding_dimension: Dimensionality of document embeddings.
        :param region_name: AWS region. Defaults to the boto3 session's configured region.
        :param aws_access_key_id: AWS access key as a `Secret`. Defaults to `AWS_ACCESS_KEY_ID`
            env var, falling back to the default boto3 credential chain if not set.
        :param aws_secret_access_key: AWS secret key as a `Secret`. Defaults to
            `AWS_SECRET_ACCESS_KEY` env var.
        :param aws_session_token: AWS session token as a `Secret`, for temporary credentials.
            Defaults to `AWS_SESSION_TOKEN` env var.
        :param create_table_if_not_exists: If `True`, create the table and vector index on
            first use if they don't already exist.
        :param similarity_function: Vector similarity function. This integration currently supports
            only `"cosine"`. DynamoDB itself also offers `DOT_PRODUCT` and `EUCLIDEAN` indexes, but
            their score conversion is not implemented yet.
        :raises ValueError: If `similarity_function` is not `"cosine"`.
        """
        if similarity_function != "cosine":
            msg = (
                f"This integration currently supports only 'cosine', got {similarity_function!r}. "
                "DynamoDB also offers DOT_PRODUCT and EUCLIDEAN vector indexes, but they are not wired up yet."
            )
            raise ValueError(msg)

        self.table_name = table_name
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        self.region_name = region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.create_table_if_not_exists = create_table_if_not_exists
        self.similarity_function = similarity_function
        # Vector-index readiness polling (see `_wait_for_vector_index_ready`). With the index
        # declared inline on CreateTable it is queryable in ~20s, so poll frequently; the timeout
        # stays generous to tolerate a slower region or a pre-existing table still backfilling.
        self.index_ready_timeout = 900.0
        self.index_ready_poll_interval = 5.0
        self._client: Any | None = None
        self._table_ready = False

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        kwargs: dict[str, Any] = {}
        if self.region_name:
            kwargs["region_name"] = self.region_name
        access_key = self.aws_access_key_id.resolve_value()
        secret_key = self.aws_secret_access_key.resolve_value()
        session_token = self.aws_session_token.resolve_value()
        if access_key:
            kwargs["aws_access_key_id"] = access_key
        if secret_key:
            kwargs["aws_secret_access_key"] = secret_key
        if session_token:
            kwargs["aws_session_token"] = session_token
        self._client = boto3.client("dynamodb", **kwargs)
        return self._client

    def _ensure_table(self) -> None:
        if self._table_ready:
            return
        client = self._get_client()
        description = self._describe_table(client)
        if description is None:
            if not self.create_table_if_not_exists:
                msg = f"Table '{self.table_name}' does not exist and create_table_if_not_exists is False."
                raise ValueError(msg)
            self._create_table(client)
            client.get_waiter("table_exists").wait(TableName=self.table_name)
            description = client.describe_table(TableName=self.table_name)["Table"]
        self._validate_table(description)
        self._wait_for_vector_index_ready(client, description)
        self._table_ready = True

    def _describe_table(self, client: Any) -> dict[str, Any] | None:
        """Returns the `DescribeTable` payload for this store's table, or `None` if it does not exist."""
        try:
            return client.describe_table(TableName=self.table_name)["Table"]
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
            return None

    def _create_table(self, client: Any) -> None:
        logger.info(
            "Creating DynamoDB table '{table}' with vector index '{index}'",
            table=self.table_name,
            index=self.index_name,
        )
        try:
            client.create_table(**self._create_table_params())
        except ClientError as e:
            # Another process created the table between our DescribeTable and CreateTable; the
            # caller waits for it to become active just like for a table we created ourselves.
            if e.response["Error"]["Code"] != "ResourceInUseException":
                raise

    def _create_table_params(self) -> dict[str, Any]:
        # Declare the vector index inline on `CreateTable` rather than adding it afterwards with
        # `UpdateTable(VectorIndexUpdates=...)`. Adding an index to an existing table triggers a
        # *backfill* that keeps it in `IndexStatus=CREATING`/`Backfilling=True`, and therefore
        # unqueryable, for many minutes (measured: >6 min even for an empty table). An index
        # created with the table has nothing to backfill and is queryable in ~20s.
        # Vector indexes are a distinct index type from GSIs/LSIs, hence `VectorIndexes` here.
        return {
            "TableName": self.table_name,
            "AttributeDefinitions": [{"AttributeName": "id", "AttributeType": "S"}],
            "KeySchema": [{"AttributeName": "id", "KeyType": "HASH"}],
            "BillingMode": "PAY_PER_REQUEST",
            "VectorIndexes": [
                {
                    "IndexName": self.index_name,
                    "VectorAttribute": {"AttributeName": _VECTOR_ATTRIBUTE},
                    "Dimensions": self.embedding_dimension,
                    "DistanceFunction": "COSINE",
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
        }

    def _find_vector_index(self, description: dict[str, Any]) -> dict[str, Any] | None:
        indexes = description.get("VectorIndexes") or []
        return next((idx for idx in indexes if idx.get("IndexName") == self.index_name), None)

    def _validate_table(self, description: dict[str, Any]) -> None:
        """
        Checks that an existing table matches this store's configuration.

        Without this check a mismatched table only fails later, on the first `SearchVectors` or
        `PutItem` call, with an opaque service error.

        :param description: The `Table` payload returned by `DescribeTable`.
        :raises ValueError: If the key schema or the vector index is incompatible with this store.
        """
        key_schema = description.get("KeySchema", [])
        if [(k.get("AttributeName"), k.get("KeyType")) for k in key_schema] != [("id", "HASH")]:
            msg = (
                f"Table '{self.table_name}' must have a single partition key named 'id' and no sort key, "
                f"found key schema {key_schema}."
            )
            raise ValueError(msg)

        index = self._find_vector_index(description)
        if index is None:
            existing = [idx.get("IndexName") for idx in description.get("VectorIndexes") or []]
            msg = (
                f"Table '{self.table_name}' has no vector index named '{self.index_name}' "
                f"(existing vector indexes: {existing}). Create one with DistanceFunction=COSINE, "
                f"Dimensions={self.embedding_dimension} on attribute '{_VECTOR_ATTRIBUTE}', "
                "or point the store at a different table."
            )
            raise ValueError(msg)

        problems = []
        if index.get("Dimensions") != self.embedding_dimension:
            problems.append(f"Dimensions={index.get('Dimensions')} (expected {self.embedding_dimension})")
        if index.get("DistanceFunction") != "COSINE":
            problems.append(f"DistanceFunction={index.get('DistanceFunction')} (expected COSINE)")
        vector_attribute = (index.get("VectorAttribute") or {}).get("AttributeName")
        if vector_attribute != _VECTOR_ATTRIBUTE:
            problems.append(f"VectorAttribute={vector_attribute!r} (expected {_VECTOR_ATTRIBUTE!r})")
        if problems:
            msg = (
                f"Vector index '{self.index_name}' on table '{self.table_name}' does not match this store's "
                f"configuration: {', '.join(problems)}."
            )
            raise ValueError(msg)

    def _index_readiness(self, description: dict[str, Any]) -> tuple[bool, str, Any]:
        """Returns `(ready, status, backfilling)` for this store's vector index from a `DescribeTable` payload."""
        index = self._find_vector_index(description)
        if index is None:
            return False, "not reported", "n/a"
        status = index.get("IndexStatus", "unknown")
        # `Backfilling` is absent entirely for an index that never had to backfill.
        backfilling = index.get("Backfilling", False)
        return status == "ACTIVE" and not backfilling, status, backfilling

    def _wait_for_vector_index_ready(self, client: Any, description: dict[str, Any]) -> None:
        """
        Blocks until the vector index is queryable.

        A vector index has its own lifecycle that is *not* captured by the table's status or the
        ``table_exists`` waiter, and there is no dedicated boto3 waiter for it, so we poll
        ``DescribeTable`` until the index reports ``IndexStatus=ACTIVE`` and is not backfilling.
        ``SearchVectors`` may still return ``ResourceNotFoundException`` for a few seconds after
        that while the vector-search endpoint catches up.

        :param client: The DynamoDB client to poll with.
        :param description: The most recent `Table` payload from `DescribeTable`.
        :raises TimeoutError: If the index does not become queryable within `index_ready_timeout`.
        """
        deadline = time.monotonic() + self.index_ready_timeout
        while True:
            ready, status, backfilling = self._index_readiness(description)
            if ready:
                return
            if time.monotonic() >= deadline:
                msg = (
                    f"Vector index '{self.index_name}' on table '{self.table_name}' did not become "
                    f"queryable within {self.index_ready_timeout}s "
                    f"(last status: {status}, backfilling: {backfilling})."
                )
                raise TimeoutError(msg)
            time.sleep(self.index_ready_poll_interval)
            description = client.describe_table(TableName=self.table_name)["Table"]

    @staticmethod
    def _sanitize_metadata_value(value: Any) -> Any:
        """
        Recursively coerces a metadata value into a DynamoDB-attribute-safe shape.

        DynamoDB's item API rejects raw Python objects it doesn't natively map (e.g. `UUID`);
        this mirrors the same coercion needed for S3 Vectors metadata, applied here defensively
        even though DynamoDB item attributes are more permissive than S3 Vectors' flat model.
        """
        if isinstance(value, dict):
            return {k: DynamoDBDocumentStore._sanitize_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [DynamoDBDocumentStore._sanitize_metadata_value(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _doc_to_item(self, doc: Document) -> dict[str, Any]:
        d = doc.to_dict(flatten=False)
        doc_id = d.pop("id")
        embedding = d.pop("embedding", None)
        # DynamoDB has no native nested-document metadata concept comparable to S3 Vectors'
        # flat-only constraint, but we still JSON-serialize the payload for a stable,
        # order-independent round trip and to keep parity with the sibling S3 Vectors adapter.
        payload = self._sanitize_metadata_value(d)
        item: dict[str, Any] = {
            "id": doc_id,
            "payload": json.dumps(payload),
        }
        if embedding is not None:
            item[_VECTOR_ATTRIBUTE] = embedding
        return item

    @staticmethod
    def _item_to_doc(item: dict[str, Any]) -> Document:
        payload = json.loads(item["payload"]) if "payload" in item else {}
        payload["id"] = item["id"]
        if _VECTOR_ATTRIBUTE in item:
            payload["embedding"] = item[_VECTOR_ATTRIBUTE]
        return Document.from_dict(payload)

    def count_documents(self) -> int:
        """
        Returns the number of documents in the store.

        Uses a consistent `Scan` with `Select="COUNT"` rather than `describe_table`'s
        `ItemCount`, which is only updated roughly every six hours by DynamoDB and would
        fail the base test contract's expectation that a count reflects a just-completed
        write immediately.

        :returns: Exact document count.
        """
        self._ensure_table()
        client = self._get_client()
        total = 0
        paginator = client.get_paginator("scan")
        for page in paginator.paginate(TableName=self.table_name, Select="COUNT", ConsistentRead=True):
            total += page.get("Count", 0)
        return total

    def _scan_documents(self, client: Any, filters: dict[str, Any] | None = None) -> Iterator[Document]:
        """
        Yields every stored document matching `filters` via a consistent full-table `Scan`.

        A `Scan` is eventually consistent by default, which surfaced as read-after-write races
        right after `write_documents` during real-AWS validation, hence `ConsistentRead=True`.

        :param client: The DynamoDB client to scan with.
        :param filters: Haystack metadata filters applied client-side; `None` or `{}` matches everything.
        """
        paginator = client.get_paginator("scan")
        for page in paginator.paginate(TableName=self.table_name, ConsistentRead=True):
            for raw_item in page.get("Items", []):
                doc = self._item_to_doc(_from_dynamodb_item(raw_item))
                if not filters or document_matches_filter(filters, doc):
                    yield doc

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns documents matching the provided filters.

        DynamoDB's `SearchVectors`/`Query` filter expressions can only reference attributes
        declared in the index's `SearchSchema` at index-creation time. Since Haystack's metadata
        filters are arbitrary and not known at index-creation time, filtering here is applied
        client-side after a consistent full-table scan, so the cost grows with the table size.

        :param filters: Haystack metadata filters. If `None`, all documents are returned.
        :returns: List of matching `Document` objects.
        """
        self._ensure_table()
        return list(self._scan_documents(self._get_client(), filters))

    @staticmethod
    def _validate_documents(documents: list[Document]) -> None:
        if not isinstance(documents, list) or any(not isinstance(doc, Document) for doc in documents):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

    def _put_item_kwargs(self, doc: Document, policy: DuplicatePolicy) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"TableName": self.table_name, "Item": _to_dynamodb_item(self._doc_to_item(doc))}
        if policy != DuplicatePolicy.OVERWRITE:
            # A conditional write makes FAIL and SKIP atomic and saves a separate GetItem per document.
            kwargs["ConditionExpression"] = "attribute_not_exists(#id)"
            kwargs["ExpressionAttributeNames"] = _ID_PLACEHOLDER
        return kwargs

    @staticmethod
    def _handle_put_error(error: ClientError, doc: Document, policy: DuplicatePolicy) -> None:
        """
        Re-raises `error` unless it is the conditional-check failure that signals a duplicate.

        For a duplicate, raises `DuplicateDocumentError` under `FAIL` and returns under `SKIP`.
        """
        if error.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise error
        if policy == DuplicatePolicy.FAIL:
            msg = f"Document with id '{doc.id}' already exists."
            raise DuplicateDocumentError(msg) from error

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents to the store.

        Documents are written one by one. With `FAIL`, documents preceding the first duplicate
        stay written.

        :param documents: Documents to write.
        :param policy: How to handle duplicates: `OVERWRITE`, `SKIP`, or `FAIL`. `NONE` (the
            default) behaves like `FAIL`.
        :raises ValueError: If `documents` contains non-`Document` objects.
        :raises DuplicateDocumentError: If a duplicate is found and policy is `FAIL`.
        :returns: Number of documents written.
        """
        self._validate_documents(documents)
        if not documents:
            return 0
        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        self._ensure_table()
        client = self._get_client()
        written = 0
        for doc in documents:
            try:
                client.put_item(**self._put_item_kwargs(doc, policy))
            except ClientError as e:
                self._handle_put_error(e, doc, policy)
                continue
            written += 1
        return written

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes documents by their IDs.

        :param document_ids: List of document IDs to delete.
        """
        if not document_ids:
            return
        self._ensure_table()
        client = self._get_client()
        for doc_id in document_ids:
            client.delete_item(TableName=self.table_name, Key={"id": {"S": doc_id}})

    def delete_all_documents(self) -> None:
        """
        Deletes all documents in the store.

        Items are deleted one by one after a consistent scan; the table and its vector index are kept.
        """
        self._ensure_table()
        client = self._get_client()
        paginator = client.get_paginator("scan")
        pages = paginator.paginate(
            TableName=self.table_name,
            ConsistentRead=True,
            ProjectionExpression="#id",
            ExpressionAttributeNames=_ID_PLACEHOLDER,
        )
        for page in pages:
            for raw_item in page.get("Items", []):
                client.delete_item(TableName=self.table_name, Key={"id": raw_item["id"]})

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Deletes all documents matching the filters.

        :param filters: Haystack metadata filters selecting the documents to delete. Must not be
            empty; use `delete_all_documents` to clear the store.
        :returns: The number of documents deleted.
        :raises ValueError: If `filters` is empty.
        """
        if not filters:
            msg = "filters must not be empty; use delete_all_documents() to delete every document."
            raise ValueError(msg)
        self._ensure_table()
        client = self._get_client()
        deleted = 0
        for doc in self._scan_documents(client, filters):
            client.delete_item(TableName=self.table_name, Key={"id": {"S": doc.id}})
            deleted += 1
        return deleted

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Merges `meta` into the metadata of all documents matching the filters.

        Existing metadata keys not present in `meta` are kept; matching keys are overwritten.

        :param filters: Haystack metadata filters selecting the documents to update. Must not be empty.
        :param meta: The metadata fields to set on each matching document.
        :returns: The number of documents updated.
        :raises ValueError: If `filters` is empty.
        """
        if not filters:
            msg = "filters must not be empty when updating documents by filter."
            raise ValueError(msg)
        self._ensure_table()
        client = self._get_client()
        updated = 0
        for doc in self._scan_documents(client, filters):
            updated_doc = dataclasses.replace(doc, meta={**doc.meta, **meta})
            client.put_item(**self._put_item_kwargs(updated_doc, DuplicatePolicy.OVERWRITE))
            updated += 1
        return updated

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieves documents most similar to the query embedding using cosine similarity.

        Uses DynamoDB's native `SearchVectors` API. This method is used internally by
        `DynamoDBEmbeddingRetriever`. Metadata filters are applied client-side after the
        vector search returns, for the same `SearchSchema` constraint documented on
        `filter_documents`. When filters are set, the maximum number of candidates DynamoDB
        allows (`SEARCH_VECTORS_MAX_TOP_K`, currently 100) is fetched and filtered down to
        `top_k`. Matches ranked below those candidates are not reachable, so a selective filter
        can return fewer than `top_k` documents even when more matching documents exist.

        :param query_embedding: The query vector.
        :param top_k: Number of top results to return, between 1 and `SEARCH_VECTORS_MAX_TOP_K`.
        :param filters: Optional metadata filters, applied client-side.
        :returns: List of `Document` objects ordered most-similar-first, with `score` set to a
            similarity in ``[0, 1]`` (converted from DynamoDB's cosine distance).
        :raises ValueError: If `query_embedding` is empty or `top_k` is outside the allowed range.
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)
        if not 1 <= top_k <= SEARCH_VECTORS_MAX_TOP_K:
            msg = f"top_k must be between 1 and {SEARCH_VECTORS_MAX_TOP_K} (DynamoDB SearchVectors limit), got {top_k}."
            raise ValueError(msg)

        self._ensure_table()
        client = self._get_client()

        fetch_k = SEARCH_VECTORS_MAX_TOP_K if filters else top_k
        response = client.search_vectors(
            TableName=self.table_name,
            IndexName=self.index_name,
            SearchVector=[{"N": str(v)} for v in query_embedding],
            TopK=fetch_k,
        )

        docs = []
        for match in response.get("SearchResults", []):
            item = _from_dynamodb_item(match["Item"])
            doc = self._item_to_doc(item)
            doc = dataclasses.replace(doc, score=self._distance_to_similarity(match.get("Score")))
            if not filters or document_matches_filter(filters, doc):
                docs.append(doc)
            if len(docs) >= top_k:
                break
        return docs

    @staticmethod
    def _distance_to_similarity(score: float | None) -> float | None:
        """
        Converts a DynamoDB COSINE distance into a Haystack similarity score.

        DynamoDB returns a cosine *distance* in ``[0, 2]`` where 0 is identical, while Haystack's
        `Document.score` convention is higher-is-more-relevant. Mapping to ``1 - distance / 2``
        yields ``1.0`` for an identical vector and ``0.0`` for an opposite one, preserving
        DynamoDB's ordering while matching Haystack's semantics.

        :param score: The raw `Score` returned by `SearchVectors`, if any.
        :returns: The corresponding similarity in ``[0, 1]``, or `None` if no score was returned.
        """
        if score is None:
            return None
        return 1.0 - (float(score) / _COSINE_MAX_DISTANCE)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            table_name=self.table_name,
            index_name=self.index_name,
            embedding_dimension=self.embedding_dimension,
            region_name=self.region_name,
            aws_access_key_id=self.aws_access_key_id.to_dict(),
            aws_secret_access_key=self.aws_secret_access_key.to_dict(),
            aws_session_token=self.aws_session_token.to_dict(),
            create_table_if_not_exists=self.create_table_if_not_exists,
            similarity_function=self.similarity_function,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DynamoDBDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        deserialize_secrets_inplace(
            data["init_parameters"],
            ["aws_access_key_id", "aws_secret_access_key", "aws_session_token"],
        )
        return default_from_dict(cls, data)


def _to_dynamodb_item(item: dict[str, Any]) -> dict[str, Any]:
    """Converts a plain-Python item dict into DynamoDB's typed attribute-value format."""
    out: dict[str, Any] = {}
    for key, value in item.items():
        if key == _VECTOR_ATTRIBUTE and isinstance(value, list):
            out[key] = {"L": [{"N": str(v)} for v in value]}
        elif isinstance(value, str):
            out[key] = {"S": value}
        elif isinstance(value, bool):
            out[key] = {"BOOL": value}
        elif isinstance(value, (int, float)):
            out[key] = {"N": str(value)}
    return out


def _from_dynamodb_item(item: dict[str, Any]) -> dict[str, Any]:
    """Converts DynamoDB's typed attribute-value format back into a plain-Python dict."""
    out: dict[str, Any] = {}
    for key, value in item.items():
        if "S" in value:
            out[key] = value["S"]
        elif "N" in value:
            out[key] = float(value["N"]) if "." in value["N"] else int(value["N"])
        elif "BOOL" in value:
            out[key] = value["BOOL"]
        elif "L" in value and key == _VECTOR_ATTRIBUTE:
            out[key] = [float(v["N"]) for v in value["L"]]
    return out
