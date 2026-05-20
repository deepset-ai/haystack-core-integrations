# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from haystack import default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack.utils.filters import document_matches_filter

from vespa.application import Vespa

from .errors import VespaDocumentStoreConfigError, VespaDocumentStoreError
from .filters import _normalize_filters

logger = logging.getLogger(__name__)

DEFAULT_QUERY_LIMIT = 400
DEFAULT_BULK_BATCH_SIZE = 100
HTTP_NOT_FOUND = 404
DEFAULT_BM25_RANKING = "bm25"
DEFAULT_SEMANTIC_RANKING = "semantic"
VESPA_CLOUD_SECRET_TOKEN_ENV = "VESPA_CLOUD_SECRET_TOKEN"


def _filters_need_python_fallback(filters: dict[str, Any]) -> bool:
    if "conditions" in filters:
        conditions = filters["conditions"]
        if not isinstance(conditions, list):
            return False
        return any(isinstance(condition, dict) and _filters_need_python_fallback(condition) for condition in conditions)

    value = filters.get("value")
    if value is None:
        return True
    if filters.get("operator") in {">", ">=", "<", "<="} and isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return False
        return True
    return False


class VespaDocumentStore:
    """
    Document store backed by an existing [Vespa](https://vespa.ai/) application.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        port: int = 8080,
        cert: Secret | None = None,
        key: Secret | None = None,
        vespa_cloud_secret_token: Secret | None = None,
        additional_headers: dict[str, str] | None = None,
        content_cluster_name: str = "content",
        schema: str = "doc",
        namespace: str | None = None,
        groupname: str | None = None,
        content_field: str = "content",
        embedding_field: str = "embedding",
        id_field: str = "id",
        metadata_fields: list[str] | None = None,
        query_limit: int = DEFAULT_QUERY_LIMIT,
    ) -> None:
        """
        Create a new Vespa document store.

        :param url: Vespa endpoint base URL. If omitted, the `VESPA_URL` environment variable is used.
        :param port: Vespa HTTP port.
        :param cert: Secret resolving to the data plane certificate file path for mTLS authentication.
        :param key: Secret resolving to the data plane key file path for mTLS authentication.
        :param vespa_cloud_secret_token: Vespa Cloud data plane secret token for token authentication.
            If omitted, the `VESPA_CLOUD_SECRET_TOKEN` environment variable is used when set, matching pyvespa.
        :param additional_headers: Additional headers to send to the Vespa application.
        :param content_cluster_name: Vespa content cluster name.
        :param schema: Vespa schema name to read from and write to.
        :param namespace: Vespa namespace. Defaults to the schema name when omitted.
        :param groupname: Optional Vespa group name.
        :param content_field: Vespa field containing the document text.
        :param embedding_field: Vespa field containing the dense embedding.
        :param id_field: Optional Vespa field containing the document id in query responses.
            Vespa document IDs are always written via `data_id`. If this field is missing in the
            schema or summaries, the integration falls back to parsing the Vespa document path.
        :param metadata_fields: Optional allowlist of metadata fields to feed and return.
        :param query_limit: Maximum number of documents returned by bulk queries. Defaults to 400 to
            stay within Vespa's common query hit limit unless explicitly overridden.
        """
        self.url = url or os.environ.get("VESPA_URL", "")
        self.port = port
        self.cert = cert
        self.key = key
        self.vespa_cloud_secret_token = vespa_cloud_secret_token or Secret.from_env_var(
            VESPA_CLOUD_SECRET_TOKEN_ENV, strict=False
        )
        self.additional_headers = additional_headers
        self.content_cluster_name = content_cluster_name
        self.schema = schema
        self.namespace = namespace or schema
        self.groupname = groupname
        self.content_field = content_field
        self.embedding_field = embedding_field
        self.id_field = id_field
        self.metadata_fields = metadata_fields
        self.query_limit = query_limit
        self._app: Any | None = None

    @property
    def app(self) -> Any:
        """
        Return the underlying `pyvespa` `Vespa` HTTP client.

        It is built from this store's `url`, `port`, and authentication settings
        (`cert`, `key`, `vespa_cloud_secret_token`, `additional_headers`) so mTLS, bearer token,
        and custom headers from the constructor (or environment) are applied.
        """
        if self._app is None:
            if not self.url:
                msg = "A Vespa URL is required to initialize the document store"
                raise VespaDocumentStoreConfigError(msg)
            self._app = Vespa(
                url=self.url,
                port=self.port,
                cert=self.cert.resolve_value() if self.cert else None,
                key=self.key.resolve_value() if self.key else None,
                vespa_cloud_secret_token=self.vespa_cloud_secret_token.resolve_value(),
                additional_headers=self.additional_headers,
            )
        return self._app

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the document store to a dictionary.

        Uses the same init-parameter names as :meth:`__init__` and `default_to_dict` so nested serialization stays
        aligned with Haystack's default component serialization.

        :returns: Serialized document store data.
        """
        return default_to_dict(
            self,
            url=self.url,
            port=self.port,
            cert=self.cert,
            key=self.key,
            vespa_cloud_secret_token=self.vespa_cloud_secret_token,
            additional_headers=self.additional_headers,
            content_cluster_name=self.content_cluster_name,
            schema=self.schema,
            namespace=self.namespace,
            groupname=self.groupname,
            content_field=self.content_field,
            embedding_field=self.embedding_field,
            id_field=self.id_field,
            metadata_fields=self.metadata_fields,
            query_limit=self.query_limit,
        )

    def count_documents(self) -> int:
        """
        Return the total number of documents in Vespa.

        :returns: Document count.
        """
        response = self._query(yql=self._build_yql(where="true", limit=0), hits=0)
        return int(response.get("root", {}).get("fields", {}).get("totalCount", 0))

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Return the number of documents matching the provided filters.

        :param filters: Haystack metadata filters.
        :returns: Count of matching documents.
        """
        where = _normalize_filters(filters, content_field=self.content_field)
        response = self._query(yql=self._build_yql(where=where, limit=0), hits=0)
        return int(response.get("root", {}).get("fields", {}).get("totalCount", 0))

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Write documents to Vespa.

        :param documents: Documents to store.
        :param policy: Duplicate handling policy.
        :returns: Number of documents written.
        """
        if (
            not isinstance(documents, Iterable)
            or isinstance(documents, (str, bytes))
            or any(not isinstance(doc, Document) for doc in documents)
        ):
            msg = "Please provide a list of Documents."
            raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        written = 0
        for document in documents:
            if policy == DuplicatePolicy.FAIL and self._document_exists(document.id):
                msg = f"Document with id '{document.id}' already exists."
                raise DuplicateDocumentError(msg)
            if policy == DuplicatePolicy.SKIP and self._document_exists(document.id):
                continue

            response = self.app.feed_data_point(
                schema=self.schema,
                namespace=self.namespace,
                groupname=self.groupname,
                data_id=document.id,
                fields=self._document_to_vespa_fields(document),
            )
            self._ensure_success(response, f"Failed to write document '{document.id}' to Vespa")
            written += 1
        return written

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Delete documents by id.

        :param document_ids: Document ids to delete.
        """
        for document_id in document_ids:
            response = self.app.delete_data(
                schema=self.schema,
                namespace=self.namespace,
                groupname=self.groupname,
                data_id=document_id,
            )
            status_code = getattr(response, "status_code", 200)
            if status_code not in {200, 202, 204, 404}:
                self._ensure_success(response, f"Failed to delete document '{document_id}' from Vespa")

    def delete_all_documents(self) -> None:
        """
        Delete all documents for this store's schema, namespace, and content cluster.

        Implemented with pyvespa `Vespa.delete_all_docs` (Document V1 bulk delete).
        """
        self.app.delete_all_docs(
            content_cluster_name=self.content_cluster_name,
            schema=self.schema,
            namespace=self.namespace,
        )

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Delete all documents matching the provided filters.

        :param filters: Haystack metadata filters.
        :returns: Number of deleted documents.
        """
        documents = self._collect_matching_documents(filters=filters)
        self.delete_documents([document.id for document in documents])
        return len(documents)

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Update metadata fields for documents matching the provided filters.

        :param filters: Haystack metadata filters.
        :param meta: Metadata values to merge into the matched documents.
        :returns: Number of updated documents.
        """
        documents = self._collect_matching_documents(filters=filters)
        updated = 0
        for document in documents:
            response = self.app.update_data(
                schema=self.schema,
                namespace=self.namespace,
                groupname=self.groupname,
                data_id=document.id,
                fields=dict(meta),
                create=False,
            )
            self._ensure_success(response, f"Failed to update document '{document.id}' in Vespa")
            updated += 1
        return updated

    def get_documents_by_id(self, document_ids: list[str]) -> list[Document]:
        """
        Retrieve documents by their ids.

        :param document_ids: Document ids to fetch.
        :returns: Matching documents.
        """
        documents: list[Document] = []
        for document_id in document_ids:
            response = self.app.get_data(
                schema=self.schema,
                namespace=self.namespace,
                groupname=self.groupname,
                data_id=document_id,
                raise_on_not_found=False,
            )
            status_code = getattr(response, "status_code", 200)
            if status_code == HTTP_NOT_FOUND:
                continue
            self._ensure_success(response, f"Failed to retrieve document '{document_id}' from Vespa")
            payload = response.get_json() if hasattr(response, "get_json") else getattr(response, "json", {})
            fields = payload.get("fields", {})
            documents.append(self._fields_to_document(fields, score=None, fallback_id=document_id))
        return documents

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Retrieve documents matching the provided filters.

        :param filters: Haystack metadata filters.
        :returns: Matching documents.
        """
        if filters and _filters_need_python_fallback(filters):
            documents = self._query_documents(where="true", top_k=self.query_limit)
            return [document for document in documents if document_matches_filter(filters=filters, document=document)]

        where = _normalize_filters(filters, content_field=self.content_field) if filters else "true"
        return self._query_documents(where=where, top_k=self.query_limit)

    def get_metadata_fields_info(self) -> dict[str, dict[str, str]]:
        """
        Return best-effort metadata field information based on configured fields.

        :returns: Field metadata information.
        """
        if self.count_documents() == 0:
            return {}

        info = {"content": {"type": "text"}}
        for field in self.metadata_fields or []:
            info[field] = {"type": "keyword"}
        return info

    def _bm25_retrieval(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        ranking: str | None = DEFAULT_BM25_RANKING,
    ) -> list[Document]:
        """
        Retrieve documents using Vespa lexical search.

        :param query: Query text.
        :param top_k: Maximum number of documents to return.
        :param filters: Optional Haystack metadata filters.
        :param ranking: Vespa rank profile for lexical matches, for example `bm25` for a profile that uses
            `bm25(content)`. Defaults to `bm25`. Pass `None` to use the schema default. See
            https://docs.vespa.ai/en/basics/ranking.html.
        :returns: Retrieved documents.
        """
        if not query:
            msg = "query must be a non-empty string"
            raise ValueError(msg)

        where = _normalize_filters(filters, content_field=self.content_field) if filters else "true"
        clauses = [where, "userQuery()"] if where != "true" else ["userQuery()"]
        return self._query_documents(where=" and ".join(clauses), top_k=top_k, query=query, ranking=ranking)

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        ranking: str | None = DEFAULT_SEMANTIC_RANKING,
        query_tensor_name: str = "query_embedding",
        target_hits: int | None = None,
    ) -> list[Document]:
        """
        Retrieve documents using Vespa nearest-neighbor search.

        :param query_embedding: Query embedding vector.
        :param top_k: Maximum number of documents to return.
        :param filters: Optional Haystack metadata filters.
        :param ranking: Vespa rank profile after nearest-neighbor retrieval, for example `semantic` for a profile
            that scores with `closeness(field, embedding)`. Defaults to `semantic`. Pass `None` to use the schema
            default. See https://docs.vespa.ai/en/basics/ranking.html.
        :param query_tensor_name: Name of the query tensor in YQL and in `input.query(...)` in your rank profile.
            For example `query_embedding` matches the default `semantic` profile used by the integration tests. See
            https://docs.vespa.ai/en/nearest-neighbor-search.html.
        :param target_hits: Optional nearest-neighbor `targetHits` value, for example `10` or `100`, controlling how
            many neighbors are considered per content node before first-phase ranking. See
            https://docs.vespa.ai/en/nearest-neighbor-search.html.
        :returns: Retrieved documents.
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        where = _normalize_filters(filters, content_field=self.content_field) if filters else "true"
        effective_target_hits = target_hits or top_k
        nearest = f"{{targetHits:{effective_target_hits}}}nearestNeighbor({self.embedding_field}, {query_tensor_name})"
        clauses = [where, nearest] if where != "true" else [nearest]
        body = {f"input.query({query_tensor_name})": query_embedding}
        return self._query_documents(where=" and ".join(clauses), top_k=top_k, ranking=ranking, body=body)

    def _query_documents(
        self,
        *,
        where: str,
        top_k: int,
        offset: int = 0,
        query: str | None = None,
        ranking: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> list[Document]:
        yql = self._build_yql(where=where, limit=top_k, offset=offset)
        payload = self._query(yql=yql, query=query, hits=top_k, offset=offset, ranking=ranking, body=body or {})
        hits = payload.get("root", {}).get("children", [])
        return [self._hit_to_document(hit) for hit in hits]

    def _query(
        self,
        *,
        yql: str,
        query: str | None = None,
        hits: int | None = None,
        offset: int | None = None,
        ranking: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_body: dict[str, Any] = {"yql": yql}
        if query is not None:
            request_body["query"] = query
        if hits is not None:
            request_body["hits"] = hits
        if offset is not None:
            request_body["offset"] = offset
        if ranking is not None:
            request_body["ranking"] = ranking
        if body:
            request_body.update(body)

        response = self.app.query(body=request_body)
        self._ensure_success(response, "Failed to query Vespa")
        if hasattr(response, "get_json"):
            return response.get_json()
        return getattr(response, "json", {})

    def _document_exists(self, document_id: str) -> bool:
        response = self.app.get_data(
            schema=self.schema,
            namespace=self.namespace,
            groupname=self.groupname,
            data_id=document_id,
            raise_on_not_found=False,
        )
        status_code = getattr(response, "status_code", 200)
        if status_code == HTTP_NOT_FOUND:
            return False
        self._ensure_success(response, f"Failed to check whether document '{document_id}' exists in Vespa")
        return True

    def _ensure_success(self, response: Any, message: str) -> None:
        if hasattr(response, "is_successful") and response.is_successful():
            return
        status_code = getattr(response, "status_code", "unknown")
        payload = response.get_json() if hasattr(response, "get_json") else getattr(response, "json", None)
        error_message = f"{message}. Status code: {status_code}. Response: {payload}"
        raise VespaDocumentStoreError(error_message)

    def _build_yql(self, *, where: str, limit: int, offset: int = 0) -> str:
        return f"select * from sources {self.schema} where {where} limit {limit} offset {offset}"  # noqa: S608

    def _document_to_vespa_fields(self, document: Document) -> dict[str, Any]:
        doc_dict = document.to_dict(flatten=False)
        fields: dict[str, Any] = {}

        if document.content is not None:
            fields[self.content_field] = document.content
        if document.embedding is not None:
            fields[self.embedding_field] = document.embedding

        metadata = doc_dict.get("meta", {}) or {}
        if self.metadata_fields:
            for key in self.metadata_fields:
                if key in metadata:
                    fields[key] = metadata[key]
        else:
            fields.update(metadata)
        return fields

    def _collect_matching_documents(
        self, *, filters: dict[str, Any] | None = None, batch_size: int = DEFAULT_BULK_BATCH_SIZE
    ) -> list[Document]:
        where = _normalize_filters(filters, content_field=self.content_field) if filters else "true"
        documents: list[Document] = []
        offset = 0
        effective_batch_size = min(batch_size, self.query_limit)

        while True:
            batch = self._query_documents(where=where, top_k=effective_batch_size, offset=offset)
            if not batch:
                break
            documents.extend(batch)
            if len(batch) < effective_batch_size:
                break
            offset += effective_batch_size

        return documents

    def _fields_to_document(self, fields: dict[str, Any], *, score: float | None, fallback_id: str | None) -> Document:
        document_id = fields.get(self.id_field) or fallback_id
        if document_id is None:
            msg = "Vespa response does not contain a document id"
            raise VespaDocumentStoreError(msg)

        meta = {
            key: value
            for key, value in fields.items()
            if key not in {self.id_field, self.content_field, self.embedding_field}
        }
        if self.metadata_fields:
            meta = {key: value for key, value in meta.items() if key in self.metadata_fields}

        return Document(
            id=str(document_id),
            content=fields.get(self.content_field),
            embedding=fields.get(self.embedding_field),
            meta=meta,
            score=score,
        )

    def _hit_to_document(self, hit: dict[str, Any]) -> Document:
        fields = hit.get("fields", {})
        fallback_id = self._extract_document_id(hit.get("id"))
        return self._fields_to_document(fields, score=hit.get("relevance"), fallback_id=fallback_id)

    @staticmethod
    def _extract_document_id(raw_id: str | None) -> str | None:
        if raw_id is None:
            return None
        if "::" in raw_id:
            return raw_id.rsplit("::", maxsplit=1)[-1]
        return raw_id.rsplit("/", maxsplit=1)[-1]
