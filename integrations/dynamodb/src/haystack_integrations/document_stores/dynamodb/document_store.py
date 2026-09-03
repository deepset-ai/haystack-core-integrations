# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
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

# DynamoDB SearchVectors returns cosine similarity, not distance: higher is more similar.
# This is the opposite convention from services like S3 Vectors, whose cosine "distance"
# is 1 - similarity (lower is more similar). Verified against the boto3 SearchVectors
# reference; treat as authoritative unless a live call proves otherwise.
_DEFAULT_TOP_K_CAP = 10000


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
        :param similarity_function: Vector similarity function. Only `"cosine"` is currently
            supported by DynamoDB's `SearchVectors` API.
        :raises ValueError: If `similarity_function` is not `"cosine"`.
        """
        if similarity_function != "cosine":
            msg = f"Only 'cosine' is supported by DynamoDB SearchVectors, got {similarity_function!r}."
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
        try:
            client.describe_table(TableName=self.table_name)
            self._table_ready = True
            return
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
            if not self.create_table_if_not_exists:
                msg = f"Table '{self.table_name}' does not exist and create_table_if_not_exists is False."
                raise ValueError(msg) from e

        client.create_table(
            TableName=self.table_name,
            AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
            KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
            BillingMode="PAY_PER_REQUEST",
        )
        client.get_waiter("table_exists").wait(TableName=self.table_name)

        # Vector indexes are a distinct index type from GSIs/LSIs, created via the
        # `VectorIndexUpdates` parameter on `UpdateTable` (or `VectorIndexes` on
        # `CreateTable`) — NOT via `GlobalSecondaryIndexUpdates`. Verified against the
        # real API reference (CreateVectorIndexAction/VectorAttributeDefinition) after
        # a naive GSI-shaped attempt failed real-AWS validation with a ParamValidationError.
        client.update_table(
            TableName=self.table_name,
            VectorIndexUpdates=[
                {
                    "Create": {
                        "IndexName": self.index_name,
                        "VectorAttribute": {"AttributeName": "embedding"},
                        "Dimensions": self.embedding_dimension,
                        "DistanceFunction": "COSINE",
                        "Projection": {"ProjectionType": "ALL"},
                    }
                }
            ],
        )
        client.get_waiter("table_exists").wait(TableName=self.table_name)
        self._table_ready = True

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
            item["embedding"] = embedding
        return item

    @staticmethod
    def _item_to_doc(item: dict[str, Any]) -> Document:
        payload = json.loads(item["payload"]) if "payload" in item else {}
        payload["id"] = item["id"]
        if "embedding" in item:
            payload["embedding"] = item["embedding"]
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

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns documents matching the provided filters.

        DynamoDB's `SearchVectors`/`Query` filter expressions can only reference attributes
        declared in the index's `SearchSchema` at index-creation time (the same constraint
        that broke a naive mock-only implementation of this pattern on a sibling project).
        Since Haystack's metadata filters are arbitrary and not known at index-creation time,
        filtering here is applied client-side after a full table scan.

        Uses `ConsistentRead=True`: a `Scan` is eventually consistent by default, which
        surfaced as real test failures immediately after `write_documents` in real-AWS
        validation — a plain-read-after-write race, not a filter-logic bug.

        :param filters: Haystack metadata filters. If `None`, all documents are returned.
        :returns: List of matching `Document` objects.
        """
        self._ensure_table()
        client = self._get_client()
        docs: list[Document] = []
        paginator = client.get_paginator("scan")
        for page in paginator.paginate(TableName=self.table_name, ConsistentRead=True):
            for raw_item in page.get("Items", []):
                item = _from_dynamodb_item(raw_item)
                doc = self._item_to_doc(item)
                if filters is None or document_matches_filter(filters, doc):
                    docs.append(doc)
        return docs

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents to the store.

        :param documents: Documents to write.
        :param policy: How to handle duplicates — `OVERWRITE`, `SKIP`, or `FAIL` (default).
        :raises ValueError: If `documents` contains non-`Document` objects.
        :raises DuplicateDocumentError: If a duplicate is found and policy is `FAIL`.
        :returns: Number of documents written.
        """
        if not documents:
            return 0
        if not isinstance(documents[0], Document):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        self._ensure_table()
        client = self._get_client()

        existing_ids: set[str] = set()
        if policy in (DuplicatePolicy.FAIL, DuplicatePolicy.SKIP):
            for doc in documents:
                response = client.get_item(
                    TableName=self.table_name, Key={"id": {"S": doc.id}}, ConsistentRead=True
                )
                if "Item" in response:
                    existing_ids.add(doc.id)

        written = 0
        for doc in documents:
            if doc.id in existing_ids:
                if policy == DuplicatePolicy.FAIL:
                    msg = f"Document with id '{doc.id}' already exists."
                    raise DuplicateDocumentError(msg)
                continue  # SKIP
            item = self._doc_to_item(doc)
            client.put_item(TableName=self.table_name, Item=_to_dynamodb_item(item))
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
        `filter_documents`. To avoid dropping matches that fall outside `top_k` post-filter,
        results are over-fetched (capped at DynamoDB's documented `SearchVectors` limit of
        10,000, not the 100-item per-page result limit which is a separate, unrelated cap).

        :param query_embedding: The query vector.
        :param top_k: Number of top results to return.
        :param filters: Optional metadata filters, applied client-side.
        :returns: List of `Document` objects sorted by descending similarity score.
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        self._ensure_table()
        client = self._get_client()

        fetch_k = min(top_k * 10, _DEFAULT_TOP_K_CAP) if filters else top_k
        response = client.search_vectors(
            TableName=self.table_name,
            IndexName=self.index_name,
            QueryVector=query_embedding,
            TopK=fetch_k,
            ReturnMetadata=True,
        )

        docs = []
        for match in response.get("Vectors", []):
            item = _from_dynamodb_item(match["Item"])
            doc = self._item_to_doc(item)
            doc = dataclasses.replace(doc, score=match.get("Distance"))
            if filters is None or document_matches_filter(filters, doc):
                docs.append(doc)
            if len(docs) >= top_k:
                break
        return docs

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
        if key == "embedding" and isinstance(value, list):
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
        elif "L" in value and key == "embedding":
            out[key] = [float(v["N"]) for v in value["L"]]
    return out
