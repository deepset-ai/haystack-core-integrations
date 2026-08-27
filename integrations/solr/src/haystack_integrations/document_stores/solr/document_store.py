# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import math
import os
import re
from dataclasses import replace
from typing import Any, Literal

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from .client import _SolrClient
from .errors import SolrDocumentStoreConfigError
from .filters import normalize_filters
from .schema import (
    BLOB_FIELD,
    CONTENT_FIELD,
    EMBEDDING_FIELD,
    ID_FIELD,
    JSON_TYPE_CODE,
    LIST_TYPE_CODES,
    SCALAR_TYPE_CODES,
    SOLR_TYPE_TO_PYTHON,
    document_to_solr,
    parse_meta_field_name,
    schema_payload,
    solr_to_document,
)

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://localhost:8983/solr"

#: `preFilter` and dependable k-NN pre-filtering landed in Solr 9.6 (SOLR-16858). Below that, a
#: `{!knn}` query combined with `fq` silently returns fewer than `top_k` documents.
MINIMUM_SOLR_VERSION = (9, 6)

#: Matches every document.
MATCH_ALL = "*:*"

#: Solr's own cap on `rows`/`limit` is much higher, but a page this size keeps responses manageable.
DEFAULT_QUERY_PAGE_SIZE = 500

DEFAULT_BATCH_SIZE = 500

#: Solr user property gating the `add-unknown-fields-to-the-schema` update chain.
SCHEMALESS_PROPERTY = "update.autoCreateFields"

#: Divisor applied before the logistic squash in `scale_score`, matching the other search-engine
#: document stores so that scaled scores are comparable across backends.
BM25_SCALING_FACTOR = 8

#: One whitespace-delimited query token, where a quoted span counts as part of the token it sits in.
#: Matching `+"a b"` as a single token is what stops a fuzzy suffix from landing on a phrase.
_FUZZY_TOKEN = re.compile(r'(?:[^\s"]|"[^"]*")+')


class SolrDocumentStore:
    """
    A Document Store for [Apache Solr](https://solr.apache.org/).

    Supports keyword search through Solr's BM25 similarity and dense vector search through
    `DenseVectorField` and the `{!knn}` query parser. Requires **Solr 9.6 or newer**.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.document_stores.solr import SolrDocumentStore

    store = SolrDocumentStore(url="http://localhost:8983/solr", core="haystack", embedding_dim=768)
    store.write_documents([Document(content="Apache Solr is a search platform.")])
    ```

    Metadata is stored in Solr fields whose names encode the Python type of the value, so metadata
    round-trips with its type intact. See the `schema` module for the details of that mapping. Metadata
    keys become Solr field names and must therefore consist of letters, digits and underscores.

    Two things Solr cannot do:

    - `Document.sparse_embedding` is ignored, with a warning, because Solr has no sparse vector field.
    - Comparing `content` with `==` is a phrase match against an analysed field rather than exact
      string equality. Filter on a metadata field when exact matching matters.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        core: str = "haystack",
        embedding_dim: int = 768,
        similarity_function: Literal["cosine", "dot_product", "euclidean"] = "cosine",
        return_embedding: bool = False,
        create_core: bool = False,
        manage_schema: bool = True,
        config_set: str = "_default",
        vector_field_type_params: dict[str, Any] | None = None,
        auth: tuple[Secret, Secret] | tuple[str, str] | None = (
            Secret.from_env_var("SOLR_USERNAME", strict=False),
            Secret.from_env_var("SOLR_PASSWORD", strict=False),
        ),
        verify_certs: bool = True,
        timeout: float = 30.0,
        batch_size: int = DEFAULT_BATCH_SIZE,
        commit: bool = True,
        commit_within_ms: int | None = None,
        query_page_size: int = DEFAULT_QUERY_PAGE_SIZE,
        **kwargs: Any,
    ) -> None:
        """
        Create a new `SolrDocumentStore`.

        :param url: Solr base URL. Falls back to the `SOLR_URL` environment variable, then to
            `http://localhost:8983/solr`.
        :param core: name of the Solr core (or SolrCloud collection) to read from and write to.
        :param embedding_dim: dimension of the embeddings. Solr fixes a vector field's dimension when
            the field is created, so this cannot be changed for an existing core.
        :param similarity_function: vector similarity to use, one of `cosine`, `dot_product` or
            `euclidean`.
        :param return_embedding: whether `filter_documents` and the retrievers return embeddings.
            Leaving this `False` keeps large vectors off the wire.
        :param create_core: whether to create the core if it does not exist. Requires the `config_set`
            to be present in Solr's configset directory (`<solr_home>/configsets`), which is not the
            case for a stock installation, so this defaults to `False` and most deployments should
            create the core out of band.
        :param manage_schema: whether to create the fields the document store needs and disable Solr's
            schemaless field guessing. Set to `False` to manage the schema yourself, in which case
            `schema.schema_payload` is the definitive list of the fields and dynamic fields required.
        :param config_set: configset used when `create_core` is enabled.
        :param vector_field_type_params: extra attributes for the vector field type, for example
            `{"hnswM": 32}` on Solr 10 or `{"hnswMaxConnections": 32}` on Solr 9. Left unset by default
            because Solr 10 renamed these attributes without a compatibility shim.
        :param auth: username and password for basic authentication. Reads the `SOLR_USERNAME` and
            `SOLR_PASSWORD` environment variables by default. Pass `None` to disable authentication.
        :param verify_certs: whether to verify TLS certificates.
        :param timeout: request timeout in seconds.
        :param batch_size: number of documents sent per update request.
        :param commit: whether writes and deletes commit immediately, making them searchable at once.
        :param commit_within_ms: ask Solr to commit within this many milliseconds instead of blocking.
        :param query_page_size: number of documents fetched per page when paginating.
        :param kwargs: extra keyword arguments forwarded to the underlying `httpx` clients, for
            example `proxy` or `headers`.
        """
        self._url = url or os.environ.get("SOLR_URL") or DEFAULT_URL
        self._core = core
        self._embedding_dim = embedding_dim
        self._similarity_function = similarity_function
        self._return_embedding = return_embedding
        self._create_core = create_core
        self._manage_schema = manage_schema
        self._config_set = config_set
        self._vector_field_type_params = vector_field_type_params
        self._auth = auth
        self._verify_certs = verify_certs
        self._timeout = timeout
        self._batch_size = batch_size
        self._commit = commit
        self._commit_within_ms = commit_within_ms
        self._query_page_size = query_page_size
        self._kwargs = kwargs

        self._solr_client = _SolrClient(
            base_url=self._url,
            auth=auth,
            verify_certs=verify_certs,
            timeout=timeout,
            client_kwargs=kwargs,
        )
        self._initialized = False
        self._async_initialized = False

    # -- serialization ---------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: dictionary with serialized data.
        """
        auth: list[dict[str, Any]] | None = None
        if (
            isinstance(self._auth, (tuple, list))
            and len(self._auth) == 2  # noqa: PLR2004
            and all(isinstance(value, Secret) for value in self._auth)
        ):
            auth = [value.to_dict() for value in self._auth]  # type: ignore[union-attr]

        return default_to_dict(
            self,
            url=self._url,
            core=self._core,
            embedding_dim=self._embedding_dim,
            similarity_function=self._similarity_function,
            return_embedding=self._return_embedding,
            create_core=self._create_core,
            manage_schema=self._manage_schema,
            config_set=self._config_set,
            vector_field_type_params=self._vector_field_type_params,
            auth=auth,
            verify_certs=self._verify_certs,
            timeout=self._timeout,
            batch_size=self._batch_size,
            commit=self._commit,
            commit_within_ms=self._commit_within_ms,
            query_page_size=self._query_page_size,
            **self._kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SolrDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data: dictionary to deserialize from.
        :returns: deserialized component.
        """
        init_parameters = data.get("init_parameters", {})
        auth = init_parameters.get("auth")
        if isinstance(auth, (tuple, list)) and all(isinstance(value, dict) for value in auth):
            deserialized = {str(index): value for index, value in enumerate(auth)}
            deserialize_secrets_inplace(deserialized, keys=list(deserialized.keys()))
            init_parameters["auth"] = tuple(deserialized[str(index)] for index in range(len(auth)))
        return default_from_dict(cls, data)

    # -- lifecycle -------------------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client. The store reconnects on the next call."""
        self._solr_client.close()
        self._initialized = False

    async def close_async(self) -> None:
        """Close the underlying async HTTP client. The store reconnects on the next call."""
        await self._solr_client.close_async()
        self._async_initialized = False

    @staticmethod
    def _parse_solr_version(payload: dict[str, Any]) -> tuple[int, int]:
        raw = payload.get("lucene", {}).get("solr-spec-version")
        if not raw:
            msg = "Could not determine the Solr version from /admin/info/system."
            raise SolrDocumentStoreConfigError(msg)
        parts = str(raw).split("-")[0].split(".")
        try:
            return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        except ValueError as error:
            msg = f"Could not parse the Solr version {raw!r}."
            raise SolrDocumentStoreConfigError(msg) from error

    def _check_version(self, payload: dict[str, Any]) -> None:
        version = self._parse_solr_version(payload)
        if version < MINIMUM_SOLR_VERSION:
            wanted = ".".join(str(part) for part in MINIMUM_SOLR_VERSION)
            found = ".".join(str(part) for part in version)
            msg = (
                f"SolrDocumentStore requires Solr {wanted} or newer, but the server at {self._url} "
                f"reports {found}. Earlier versions do not support dependable k-NN pre-filtering, so "
                f"embedding retrieval would silently return fewer documents than requested."
            )
            raise SolrDocumentStoreConfigError(msg)

    @staticmethod
    def _core_exists(payload: dict[str, Any], core: str) -> bool:
        # Solr answers STATUS for an unknown core with an empty dict rather than an error.
        return bool(payload.get("status", {}).get(core))

    def _pending_schema_payload(self, schema: dict[str, Any]) -> dict[str, Any]:
        """
        Build the Schema API payload for whatever the core is still missing.

        :param schema: the response of a Schema API read.
        :returns: the payload to POST, empty when the core already has everything.
        """
        existing = schema.get("schema", {})
        return schema_payload(
            embedding_dim=self._embedding_dim,
            similarity_function=self._similarity_function,
            vector_field_type_params=self._vector_field_type_params,
            existing_field_types={entry["name"] for entry in existing.get("fieldTypes", [])},
            existing_fields={entry["name"] for entry in existing.get("fields", [])},
            existing_dynamic_fields={entry["name"] for entry in existing.get("dynamicFields", [])},
        )

    def _verify_vector_field(self, payload: dict[str, Any]) -> None:
        """Refuse to run against a core whose vector field has a different dimension."""
        schema = payload.get("schema", {})
        fields = {entry["name"]: entry for entry in schema.get("fields", [])}
        field_types = {entry["name"]: entry for entry in schema.get("fieldTypes", [])}
        embedding_field = fields.get(EMBEDDING_FIELD)
        if embedding_field is None:
            return
        field_type = field_types.get(embedding_field.get("type", ""), {})
        if field_type.get("class") != "solr.DenseVectorField":
            return
        existing_dim = field_type.get("vectorDimension")
        if existing_dim is not None and int(existing_dim) != self._embedding_dim:
            msg = (
                f"Core {self._core!r} already has an {EMBEDDING_FIELD!r} field with "
                f"vectorDimension={existing_dim}, but this document store is configured for "
                f"embedding_dim={self._embedding_dim}. A Solr vector dimension cannot be changed after "
                f"the field is created, so use a different core or match the existing dimension."
            )
            raise SolrDocumentStoreConfigError(msg)

    def _core_creation_params(self) -> dict[str, Any]:
        return {"action": "CREATE", "name": self._core, "configSet": self._config_set}

    @staticmethod
    def _disable_schemaless_payload() -> dict[str, Any]:
        # The `_default` configset routes updates through `add-unknown-fields-to-the-schema`, whose
        # `remove-blank` processor drops zero-length values outright - so a document with
        # `content=""` would come back with no content at all.
        return {"set-user-property": {SCHEMALESS_PROPERTY: "false"}}

    @staticmethod
    def _schemaless_already_disabled(overlay: dict[str, Any]) -> bool:
        """
        Report whether the config overlay already disables schemaless field guessing.

        Checked before writing, because a core created from a shared configset stores its overlay in
        that configset rather than in the core, so an unconditional write would keep rewriting state
        shared with every other core using it.
        """
        user_props = overlay.get("overlay", {}).get("userProps", {})
        return str(user_props.get(SCHEMALESS_PROPERTY, "")).lower() == "false"

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        request = self._solr_client.request
        self._check_version(request("GET", "admin/info/system"))
        if self._create_core:
            status = request("GET", "admin/cores", params={"action": "STATUS", "core": self._core})
            if not self._core_exists(status, self._core):
                request("GET", "admin/cores", params=self._core_creation_params())
        if self._manage_schema:
            if not self._schemaless_already_disabled(request("GET", f"{self._core}/config/overlay")):
                request("POST", f"{self._core}/config", json_body=self._disable_schemaless_payload())
            schema = request("GET", f"{self._core}/schema")
            self._verify_vector_field(schema)
            payload = self._pending_schema_payload(schema)
            if payload:
                request("POST", f"{self._core}/schema", json_body=payload)
        self._initialized = True

    async def _ensure_initialized_async(self) -> None:
        if self._async_initialized:
            return
        request = self._solr_client.request_async
        self._check_version(await request("GET", "admin/info/system"))
        if self._create_core:
            status = await request("GET", "admin/cores", params={"action": "STATUS", "core": self._core})
            if not self._core_exists(status, self._core):
                await request("GET", "admin/cores", params=self._core_creation_params())
        if self._manage_schema:
            overlay = await request("GET", f"{self._core}/config/overlay")
            if not self._schemaless_already_disabled(overlay):
                await request("POST", f"{self._core}/config", json_body=self._disable_schemaless_payload())
            schema = await request("GET", f"{self._core}/schema")
            self._verify_vector_field(schema)
            payload = self._pending_schema_payload(schema)
            if payload:
                await request("POST", f"{self._core}/schema", json_body=payload)
        self._async_initialized = True

    # -- request building ------------------------------------------------------------------------

    def _requested_fields(self, *, with_embedding: bool | None = None, with_score: bool = False) -> list[str]:
        """
        Build the `fl` parameter.

        Solr cannot express "everything except one field", but it does accept globs, so listing the
        document fields explicitly is what keeps embeddings off the wire when they are not wanted.
        """
        include_embedding = self._return_embedding if with_embedding is None else with_embedding
        fields = [ID_FIELD, CONTENT_FIELD, BLOB_FIELD, "meta_*"]
        if include_embedding:
            fields.append(EMBEDDING_FIELD)
        if with_score:
            fields.append("score")
        return fields

    def _update_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self._commit_within_ms is not None:
            params["commitWithin"] = self._commit_within_ms
        if self._commit:
            params["commit"] = True
        return params

    @staticmethod
    def _filter_clauses(filters: dict[str, Any] | None) -> list[str]:
        return [normalize_filters(filters)] if filters else []

    def _prepare_page_request(
        self, *, filters: dict[str, Any] | None, cursor: str, with_embedding: bool | None = None
    ) -> dict[str, Any]:
        return {
            "query": MATCH_ALL,
            "filter": self._filter_clauses(filters),
            "limit": self._query_page_size,
            # cursorMark needs a total ordering, and the uniqueKey provides one.
            "sort": f"{ID_FIELD} asc",
            "fields": self._requested_fields(with_embedding=with_embedding),
            "params": {"cursorMark": cursor},
        }

    def _documents_from_response(self, payload: dict[str, Any], *, with_score: bool = False) -> list[Document]:
        docs = payload.get("response", {}).get("docs", [])
        return [solr_to_document(doc, score=doc.get("score") if with_score else None) for doc in docs]

    # -- counting --------------------------------------------------------------------------------

    @staticmethod
    def _count_payload(filters: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "query": MATCH_ALL,
            "filter": [normalize_filters(filters)] if filters else [],
            "limit": 0,
        }

    @staticmethod
    def _count_from_response(payload: dict[str, Any]) -> int:
        return int(payload.get("response", {}).get("numFound", 0))

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns: the number of documents.
        """
        self._ensure_initialized()
        return self._count_from_response(
            self._solr_client.request("POST", f"{self._core}/query", json_body=self._count_payload(None))
        )

    async def count_documents_async(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns: the number of documents.
        """
        await self._ensure_initialized_async()
        return self._count_from_response(
            await self._solr_client.request_async("POST", f"{self._core}/query", json_body=self._count_payload(None))
        )

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Returns how many documents match the given filters.

        :param filters: the filters to apply.
        :returns: the number of matching documents.
        """
        self._ensure_initialized()
        return self._count_from_response(
            self._solr_client.request("POST", f"{self._core}/query", json_body=self._count_payload(filters))
        )

    async def count_documents_by_filter_async(self, filters: dict[str, Any]) -> int:
        """
        Returns how many documents match the given filters.

        :param filters: the filters to apply.
        :returns: the number of matching documents.
        """
        await self._ensure_initialized_async()
        return self._count_from_response(
            await self._solr_client.request_async("POST", f"{self._core}/query", json_body=self._count_payload(filters))
        )

    # -- reading ---------------------------------------------------------------------------------

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters, refer to the
        [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

        All Haystack operators are supported: `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not in`, and the
        `AND`, `OR` and `NOT` logical operators. Three behaviours are worth knowing:

        - `>`, `>=`, `<` and `<=` accept numbers and ISO-8601 date strings. Any other string raises a
          `FilterError`, because Solr would compare it lexicographically and quietly give an answer
          nobody meant.
        - Because the value's Python type selects the Solr field, `{"field": "meta.page", "value": 100}`
          and `{"field": "meta.page", "value": "100"}` match different documents.
        - `==` on `content` is a phrase match against an analysed field, not exact equality.

        :param filters: the filters to apply to the document list.
        :returns: a list of Documents that match the given filters.
        :raises FilterError: if the filters are malformed, or compare a value Solr cannot order.
        """
        self._ensure_initialized()
        documents: list[Document] = []
        cursor = "*"
        while True:
            payload = self._solr_client.request(
                "POST",
                f"{self._core}/query",
                json_body=self._prepare_page_request(filters=filters, cursor=cursor),
            )
            documents.extend(self._documents_from_response(payload))
            next_cursor = payload.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
        return documents

    async def filter_documents_async(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        See `filter_documents` for the supported operators and their caveats.

        :param filters: the filters to apply to the document list.
        :returns: a list of Documents that match the given filters.
        :raises FilterError: if the filters are malformed, or compare a value Solr cannot order.
        """
        await self._ensure_initialized_async()
        documents: list[Document] = []
        cursor = "*"
        while True:
            payload = await self._solr_client.request_async(
                "POST",
                f"{self._core}/query",
                json_body=self._prepare_page_request(filters=filters, cursor=cursor),
            )
            documents.extend(self._documents_from_response(payload))
            next_cursor = payload.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
        return documents

    # -- writing ---------------------------------------------------------------------------------

    @staticmethod
    def _validate_documents(documents: list[Document]) -> None:
        if not isinstance(documents, list):
            msg = "Documents must be a list"
            raise ValueError(msg)
        if any(not isinstance(document, Document) for document in documents):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

    @staticmethod
    def _existing_ids_payload(document_ids: list[str]) -> dict[str, Any]:
        """
        Build the real-time get request asking which of `document_ids` already exist.

        The ids travel as repeated `id` parameters rather than inside a query, because the real-time
        get handler takes them verbatim. Nothing parses them, so no id needs escaping and no
        separator can collide with an id's own characters - a query-based lookup gets both wrong for
        any id containing Lucene syntax, whitespace or a comma.

        Being a real-time get, it also sees documents that have been written but not yet committed,
        which a search cannot. On a core configured without an update log it quietly falls back to
        the committed index, which is no worse than a search.

        The parameters go in the request body so that a large batch cannot overflow the URL.
        """
        return {"params": {"id": list(document_ids), "fl": ID_FIELD}}

    @staticmethod
    def _existing_ids_from_response(payload: dict[str, Any]) -> set[str]:
        """
        Read the ids out of a real-time get response.

        Solr answers a lookup of several ids with a `response` block, but a lookup of exactly one
        with a bare `doc` that is `null` when the document does not exist. Writing a single document
        is common enough that both shapes have to be handled.

        :param payload: the parsed real-time get response.
        :returns: the ids that already exist in the core.
        """
        if "response" in payload:
            return {doc[ID_FIELD] for doc in payload["response"].get("docs", [])}
        document = payload.get("doc")
        return {document[ID_FIELD]} if document else set()

    def _batches(self, documents: list[Document]) -> list[list[dict[str, Any]]]:
        solr_documents = [document_to_solr(document) for document in documents]
        return [
            solr_documents[start : start + self._batch_size]
            for start in range(0, len(solr_documents), self._batch_size)
        ]

    @staticmethod
    def _apply_duplicate_policy(
        documents: list[Document], policy: DuplicatePolicy, existing_ids: set[str]
    ) -> list[Document]:
        if policy == DuplicatePolicy.FAIL:
            duplicates = sorted(document.id for document in documents if document.id in existing_ids)
            if duplicates:
                msg = f"IDs '{', '.join(duplicates)}' already exist in the document store."
                raise DuplicateDocumentError(msg)
            return documents
        return [document for document in documents if document.id not in existing_ids]

    def _resolve_policy(self, policy: DuplicatePolicy) -> DuplicatePolicy:
        # Solr overwrites by id by default; FAIL is the documented default for the other stores too.
        return DuplicatePolicy.FAIL if policy == DuplicatePolicy.NONE else policy

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes Documents to Solr.

        Metadata keys must consist of letters, digits and underscores only, because each key becomes a
        Solr field name. Sparse embeddings are dropped, as Solr has no sparse vector field.

        :param documents: a list of Documents to write.
        :param policy: the policy to apply when a Document with the same id already exists.
        :returns: the number of Documents written.
        :raises ValueError: if `documents` is not a list of Documents, or a metadata key cannot be
            expressed as a Solr field name.
        :raises DuplicateDocumentError: if `policy` is `FAIL` and a Document already exists.
        """
        self._ensure_initialized()
        self._validate_documents(documents)
        if not documents:
            return 0

        self._warn_about_sparse_embeddings(documents)
        policy = self._resolve_policy(policy)
        if policy in (DuplicatePolicy.FAIL, DuplicatePolicy.SKIP):
            payload = self._solr_client.request(
                "POST",
                f"{self._core}/get",
                json_body=self._existing_ids_payload([document.id for document in documents]),
            )
            existing = self._existing_ids_from_response(payload)
            documents = self._apply_duplicate_policy(documents, policy, existing)
            if not documents:
                return 0

        for batch in self._batches(documents):
            self._solr_client.request("POST", f"{self._core}/update", params=self._update_params(), json_body=batch)
        return len(documents)

    async def write_documents_async(
        self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """
        Writes Documents to Solr.

        :param documents: a list of Documents to write.
        :param policy: the policy to apply when a Document with the same id already exists.
        :returns: the number of Documents written.
        """
        await self._ensure_initialized_async()
        self._validate_documents(documents)
        if not documents:
            return 0

        self._warn_about_sparse_embeddings(documents)
        policy = self._resolve_policy(policy)
        if policy in (DuplicatePolicy.FAIL, DuplicatePolicy.SKIP):
            payload = await self._solr_client.request_async(
                "POST",
                f"{self._core}/get",
                json_body=self._existing_ids_payload([document.id for document in documents]),
            )
            existing = self._existing_ids_from_response(payload)
            documents = self._apply_duplicate_policy(documents, policy, existing)
            if not documents:
                return 0

        for batch in self._batches(documents):
            await self._solr_client.request_async(
                "POST", f"{self._core}/update", params=self._update_params(), json_body=batch
            )
        return len(documents)

    @staticmethod
    def _warn_about_sparse_embeddings(documents: list[Document]) -> None:
        if any(document.sparse_embedding for document in documents):
            logger.warning(
                "Documents with sparse embeddings were provided, but Solr has no sparse vector field. "
                "The `sparse_embedding` field will be ignored."
            )

    # -- deleting --------------------------------------------------------------------------------

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes all documents with the given ids.

        :param document_ids: the ids of the documents to delete.
        """
        if not document_ids:
            return
        self._ensure_initialized()
        self._solr_client.request(
            "POST",
            f"{self._core}/update",
            params=self._update_params(),
            json_body={"delete": list(document_ids)},
        )

    async def delete_documents_async(self, document_ids: list[str]) -> None:
        """
        Deletes all documents with the given ids.

        :param document_ids: the ids of the documents to delete.
        """
        if not document_ids:
            return
        await self._ensure_initialized_async()
        await self._solr_client.request_async(
            "POST",
            f"{self._core}/update",
            params=self._update_params(),
            json_body={"delete": list(document_ids)},
        )

    def delete_all_documents(self) -> None:
        """Deletes all documents in the core, leaving the schema in place."""
        self._ensure_initialized()
        self._solr_client.request(
            "POST",
            f"{self._core}/update",
            params=self._update_params(),
            json_body={"delete": {"query": MATCH_ALL}},
        )

    async def delete_all_documents_async(self) -> None:
        """Deletes all documents in the core, leaving the schema in place."""
        await self._ensure_initialized_async()
        await self._solr_client.request_async(
            "POST",
            f"{self._core}/update",
            params=self._update_params(),
            json_body={"delete": {"query": MATCH_ALL}},
        )

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Deletes all documents matching the given filters.

        :param filters: the filters selecting the documents to delete.
        :returns: the number of documents deleted.
        """
        self._ensure_initialized()
        deleted = self.count_documents_by_filter(filters)
        if not deleted:
            return 0
        self._solr_client.request(
            "POST",
            f"{self._core}/update",
            params=self._update_params(),
            json_body={"delete": {"query": normalize_filters(filters)}},
        )
        return deleted

    async def delete_by_filter_async(self, filters: dict[str, Any]) -> int:
        """
        Deletes all documents matching the given filters.

        :param filters: the filters selecting the documents to delete.
        :returns: the number of documents deleted.
        """
        await self._ensure_initialized_async()
        deleted = await self.count_documents_by_filter_async(filters)
        if not deleted:
            return 0
        await self._solr_client.request_async(
            "POST",
            f"{self._core}/update",
            params=self._update_params(),
            json_body={"delete": {"query": normalize_filters(filters)}},
        )
        return deleted

    # -- updating --------------------------------------------------------------------------------

    @staticmethod
    def _merge_meta(documents: list[Document], meta: dict[str, Any]) -> list[Document]:
        """Merge `meta` into each document, keeping the existing ids so identities are preserved."""
        return [replace(document, meta={**document.meta, **meta}) for document in documents]

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Merges `meta` into the metadata of every document matching `filters`.

        Matching documents are read, merged and rewritten in full rather than updated in place. A Solr
        atomic update sets one field at a time, which would leave the previous value behind in another
        field whenever a metadata value changes Python type, since the type is part of the field name.

        :param filters: the filters selecting the documents to update.
        :param meta: the metadata to merge into each matching document.
        :returns: the number of documents updated.
        """
        self._ensure_initialized()
        # The embedding has to come along, otherwise rewriting the document would drop it.
        documents = self._filter_documents_with_embeddings(filters)
        if not documents:
            return 0
        updated = self._merge_meta(documents, meta)
        for batch in self._batches(updated):
            self._solr_client.request("POST", f"{self._core}/update", params=self._update_params(), json_body=batch)
        return len(updated)

    async def update_by_filter_async(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Merges `meta` into the metadata of every document matching `filters`.

        :param filters: the filters selecting the documents to update.
        :param meta: the metadata to merge into each matching document.
        :returns: the number of documents updated.
        """
        await self._ensure_initialized_async()
        documents = await self._filter_documents_with_embeddings_async(filters)
        if not documents:
            return 0
        updated = self._merge_meta(documents, meta)
        for batch in self._batches(updated):
            await self._solr_client.request_async(
                "POST", f"{self._core}/update", params=self._update_params(), json_body=batch
            )
        return len(updated)

    def _filter_documents_with_embeddings(self, filters: dict[str, Any] | None) -> list[Document]:
        documents: list[Document] = []
        cursor = "*"
        while True:
            payload = self._solr_client.request(
                "POST",
                f"{self._core}/query",
                json_body=self._prepare_page_request(filters=filters, cursor=cursor, with_embedding=True),
            )
            documents.extend(self._documents_from_response(payload))
            next_cursor = payload.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
        return documents

    async def _filter_documents_with_embeddings_async(self, filters: dict[str, Any] | None) -> list[Document]:
        documents: list[Document] = []
        cursor = "*"
        while True:
            payload = await self._solr_client.request_async(
                "POST",
                f"{self._core}/query",
                json_body=self._prepare_page_request(filters=filters, cursor=cursor, with_embedding=True),
            )
            documents.extend(self._documents_from_response(payload))
            next_cursor = payload.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor
        return documents

    # -- metadata introspection ------------------------------------------------------------------

    @staticmethod
    def _strip_meta_prefix(metadata_field: str) -> str:
        """Drop the `meta.` prefix Haystack callers may use to address a metadata field."""
        return metadata_field[5:] if metadata_field.startswith("meta.") else metadata_field

    @staticmethod
    def _luke_params() -> dict[str, Any]:
        return {"numTerms": 0}

    @staticmethod
    def _meta_fields_from_luke(payload: dict[str, Any]) -> dict[str, tuple[str, str]]:
        """
        Map each live Solr metadata field to the metadata key and type code it encodes.

        Luke reports the fields actually present in the index, so an empty core yields nothing - which
        is what makes `get_metadata_fields_info` return `{}` for an empty store.
        """
        discovered: dict[str, tuple[str, str]] = {}
        for field in payload.get("fields", {}):
            parsed = parse_meta_field_name(field)
            if parsed is not None:
                type_code, key = parsed
                discovered[field] = (type_code, key)
        return discovered

    @staticmethod
    def _python_type_for_code(type_code: str) -> str:
        if type_code == JSON_TYPE_CODE:
            return "object"
        if type_code in LIST_TYPE_CODES:
            return f"list[{SOLR_TYPE_TO_PYTHON[LIST_TYPE_CODES[type_code]]}]"
        return SOLR_TYPE_TO_PYTHON[SCALAR_TYPE_CODES[type_code]]

    def _fields_info_from_luke(self, payload: dict[str, Any]) -> dict[str, dict[str, str]]:
        info: dict[str, dict[str, str]] = {}
        for _field, (type_code, key) in sorted(self._meta_fields_from_luke(payload).items()):
            # A key stored under several type codes reports the first type seen, in field-name order.
            info.setdefault(key, {"type": self._python_type_for_code(type_code)})
        return info

    def get_metadata_fields_info(self) -> dict[str, dict[str, str]]:
        """
        Returns the metadata fields present in the core and their types.

        :returns: a mapping of metadata field name to a dict with a `type` key.
        """
        self._ensure_initialized()
        payload = self._solr_client.request("GET", f"{self._core}/admin/luke", params=self._luke_params())
        return self._fields_info_from_luke(payload)

    async def get_metadata_fields_info_async(self) -> dict[str, dict[str, str]]:
        """
        Returns the metadata fields present in the core and their types.

        :returns: a mapping of metadata field name to a dict with a `type` key.
        """
        await self._ensure_initialized_async()
        payload = await self._solr_client.request_async("GET", f"{self._core}/admin/luke", params=self._luke_params())
        return self._fields_info_from_luke(payload)

    def _solr_fields_for_key(self, luke_payload: dict[str, Any], key: str) -> dict[str, str]:
        """Return the live Solr fields storing metadata `key`, mapped to their type code."""
        return {
            field: type_code
            for field, (type_code, field_key) in self._meta_fields_from_luke(luke_payload).items()
            if field_key == key
        }

    @staticmethod
    def _unique_counts_payload(filters: dict[str, Any] | None, facets: dict[str, Any]) -> dict[str, Any]:
        return {
            "query": MATCH_ALL,
            "filter": [normalize_filters(filters)] if filters else [],
            "limit": 0,
            "facet": facets,
        }

    def _unique_count_facets(
        self, luke_payload: dict[str, Any], metadata_fields: list[str]
    ) -> tuple[dict[str, Any], dict[str, list[str]]]:
        """Build one `numBuckets` facet per (key, type code) pair, plus the key -> facet-name index."""
        facets: dict[str, Any] = {}
        by_key: dict[str, list[str]] = {}
        for key in metadata_fields:
            names = []
            for index, solr_field in enumerate(sorted(self._solr_fields_for_key(luke_payload, key))):
                name = f"f{len(facets)}_{index}"
                facets[name] = {
                    "type": "terms",
                    "field": solr_field,
                    "limit": 0,
                    "numBuckets": True,
                }
                names.append(name)
            by_key[key] = names
        return facets, by_key

    @staticmethod
    def _unique_counts_from_response(payload: dict[str, Any], by_key: dict[str, list[str]]) -> dict[str, int]:
        facets = payload.get("facets", {})
        # Distinct values are summed across type codes, so values that merely share a string form -
        # the int 1 and the str "1" - stay distinct, consistent with get_metadata_field_unique_values.
        return {
            key: sum(int(facets.get(name, {}).get("numBuckets", 0)) for name in names) for key, names in by_key.items()
        }

    def count_unique_metadata_by_filter(self, filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]:
        """
        Counts the distinct values of each given metadata field among documents matching `filters`.

        :param filters: the filters restricting which documents are considered.
        :param metadata_fields: the metadata fields to count distinct values for.
        :returns: a mapping of metadata field name to its number of distinct values.
        """
        self._ensure_initialized()
        luke = self._solr_client.request("GET", f"{self._core}/admin/luke", params=self._luke_params())
        facets, by_key = self._unique_count_facets(luke, metadata_fields)
        if not facets:
            return dict.fromkeys(metadata_fields, 0)
        payload = self._solr_client.request(
            "POST", f"{self._core}/query", json_body=self._unique_counts_payload(filters, facets)
        )
        return self._unique_counts_from_response(payload, by_key)

    async def count_unique_metadata_by_filter_async(
        self, filters: dict[str, Any], metadata_fields: list[str]
    ) -> dict[str, int]:
        """
        Counts the distinct values of each given metadata field among documents matching `filters`.

        :param filters: the filters restricting which documents are considered.
        :param metadata_fields: the metadata fields to count distinct values for.
        :returns: a mapping of metadata field name to its number of distinct values.
        """
        await self._ensure_initialized_async()
        luke = await self._solr_client.request_async("GET", f"{self._core}/admin/luke", params=self._luke_params())
        facets, by_key = self._unique_count_facets(luke, metadata_fields)
        if not facets:
            return dict.fromkeys(metadata_fields, 0)
        payload = await self._solr_client.request_async(
            "POST", f"{self._core}/query", json_body=self._unique_counts_payload(filters, facets)
        )
        return self._unique_counts_from_response(payload, by_key)

    @staticmethod
    def _numeric_fields(fields_by_code: dict[str, str]) -> list[str]:
        return [field for field, type_code in fields_by_code.items() if type_code in {"l", "d"}]

    @staticmethod
    def _min_max_payload(numeric_fields: list[str]) -> dict[str, Any]:
        facets: dict[str, Any] = {}
        for index, field in enumerate(sorted(numeric_fields)):
            facets[f"min_{index}"] = f"min({field})"
            facets[f"max_{index}"] = f"max({field})"
        return {"query": MATCH_ALL, "limit": 0, "facet": facets}

    @staticmethod
    def _min_max_from_response(payload: dict[str, Any]) -> dict[str, float | int | None]:
        facets = payload.get("facets", {})
        # Solr omits min/max entirely when nothing matched, so a missing key means "no value".
        minima = [value for key, value in facets.items() if key.startswith("min_") and value is not None]
        maxima = [value for key, value in facets.items() if key.startswith("max_") and value is not None]
        return {"min": min(minima) if minima else None, "max": max(maxima) if maxima else None}

    def get_metadata_field_min_max(self, metadata_field: str) -> dict[str, float | int | None]:
        """
        Returns the minimum and maximum value of a numeric metadata field.

        :param metadata_field: the metadata field, with or without a `meta.` prefix.
        :returns: a dict with `min` and `max` keys, both `None` when the field has no numeric values.
        """
        self._ensure_initialized()
        key = self._strip_meta_prefix(metadata_field)
        luke = self._solr_client.request("GET", f"{self._core}/admin/luke", params=self._luke_params())
        numeric = self._numeric_fields(self._solr_fields_for_key(luke, key))
        if not numeric:
            return {"min": None, "max": None}
        payload = self._solr_client.request("POST", f"{self._core}/query", json_body=self._min_max_payload(numeric))
        return self._min_max_from_response(payload)

    async def get_metadata_field_min_max_async(self, metadata_field: str) -> dict[str, float | int | None]:
        """
        Returns the minimum and maximum value of a numeric metadata field.

        :param metadata_field: the metadata field, with or without a `meta.` prefix.
        :returns: a dict with `min` and `max` keys, both `None` when the field has no numeric values.
        """
        await self._ensure_initialized_async()
        key = self._strip_meta_prefix(metadata_field)
        luke = await self._solr_client.request_async("GET", f"{self._core}/admin/luke", params=self._luke_params())
        numeric = self._numeric_fields(self._solr_fields_for_key(luke, key))
        if not numeric:
            return {"min": None, "max": None}
        payload = await self._solr_client.request_async(
            "POST", f"{self._core}/query", json_body=self._min_max_payload(numeric)
        )
        return self._min_max_from_response(payload)

    @staticmethod
    def _unique_values_params(
        solr_field: str, filters: dict[str, Any] | None, search_term: str | None
    ) -> dict[str, Any]:
        """
        Build a classic-facet request for the distinct values of one Solr field.

        The JSON Facet API silently ignores `contains`, so the classic facet API is the only one that
        can filter buckets by substring. Every bucket is requested and paginated in Python, which also
        yields the exact total the caller needs.
        """
        params: dict[str, Any] = {
            "q": MATCH_ALL,
            "rows": 0,
            "facet": True,
            "facet.field": solr_field,
            "facet.limit": -1,
            "facet.mincount": 1,
            "facet.sort": "index",
        }
        if filters:
            params["fq"] = normalize_filters(filters)
        if search_term:
            params["facet.contains"] = search_term
            params["facet.contains.ignoreCase"] = True
        return params

    @staticmethod
    def _decode_facet_value(raw: Any, type_code: str) -> Any:
        # Classic facets render every bucket as a string, so the type code restores the Python type.
        if type_code in ("l", "ls"):
            return int(raw)
        if type_code in ("d", "ds"):
            return float(raw)
        if type_code in ("b", "bs"):
            return raw is True or str(raw).lower() == "true"
        return raw

    @staticmethod
    def _facet_buckets(payload: dict[str, Any], solr_field: str) -> list[Any]:
        counts = payload.get("facet_counts", {}).get("facet_fields", {}).get(solr_field, [])
        # Solr returns a flat [value, count, value, count, ...] list.
        return list(counts[0::2])

    def get_metadata_field_unique_values(
        self,
        metadata_field: str,
        search_term: str | None = None,
        from_: int = 0,
        size: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[Any], int]:
        """
        Returns the distinct values of a metadata field, paginated.

        :param metadata_field: the metadata field, with or without a `meta.` prefix.
        :param search_term: when given, only values containing it (case-insensitively) are returned.
        :param from_: index of the first value to return.
        :param size: how many values to return.
        :param filters: filters restricting which documents are considered.
        :returns: a `(values, total_count)` pair, where `total_count` counts all matching values.
        """
        self._ensure_initialized()
        luke = self._solr_client.request("GET", f"{self._core}/admin/luke", params=self._luke_params())
        key = self._strip_meta_prefix(metadata_field)
        values: list[Any] = []
        for solr_field, type_code in sorted(self._solr_fields_for_key(luke, key).items()):
            if type_code == JSON_TYPE_CODE:
                # JSON-encoded values are opaque to Solr and not indexed, so they cannot be faceted.
                continue
            payload = self._solr_client.request(
                "GET",
                f"{self._core}/select",
                params=self._unique_values_params(solr_field, filters, search_term),
            )
            values.extend(self._decode_facet_value(raw, type_code) for raw in self._facet_buckets(payload, solr_field))
        return values[from_ : from_ + size], len(values)

    async def get_metadata_field_unique_values_async(
        self,
        metadata_field: str,
        search_term: str | None = None,
        from_: int = 0,
        size: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[Any], int]:
        """
        Returns the distinct values of a metadata field, paginated.

        :param metadata_field: the metadata field, with or without a `meta.` prefix.
        :param search_term: when given, only values containing it (case-insensitively) are returned.
        :param from_: index of the first value to return.
        :param size: how many values to return.
        :param filters: filters restricting which documents are considered.
        :returns: a `(values, total_count)` pair, where `total_count` counts all matching values.
        """
        await self._ensure_initialized_async()
        luke = await self._solr_client.request_async("GET", f"{self._core}/admin/luke", params=self._luke_params())
        key = self._strip_meta_prefix(metadata_field)
        values: list[Any] = []
        for solr_field, type_code in sorted(self._solr_fields_for_key(luke, key).items()):
            if type_code == JSON_TYPE_CODE:
                continue
            payload = await self._solr_client.request_async(
                "GET",
                f"{self._core}/select",
                params=self._unique_values_params(solr_field, filters, search_term),
            )
            values.extend(self._decode_facet_value(raw, type_code) for raw in self._facet_buckets(payload, solr_field))
        return values[from_ : from_ + size], len(values)

    # -- retrieval -------------------------------------------------------------------------------

    @staticmethod
    def _apply_fuzziness(query: str, fuzziness: int) -> str:
        if not fuzziness:
            return query
        # edismax accepts a per-term fuzzy suffix; the terms themselves stay unescaped so that the
        # parser keeps handling +/- prefixes the way a user would expect.
        #
        # Any token containing a quote is left alone: `~n` after a phrase is a proximity slop rather
        # than a fuzzy match, and inside the quotes it would just be analysed away - either way the
        # phrase would silently stop meaning what the caller wrote. Substituting rather than
        # re-joining keeps whatever the pattern does not match, so an unbalanced quote survives.
        return _FUZZY_TOKEN.sub(
            lambda match: match.group() if '"' in match.group() else f"{match.group()}~{fuzziness}",
            query,
        )

    def _bm25_payload(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
        fuzziness: int,
        all_terms_must_match: bool,
    ) -> dict[str, Any]:
        return {
            "query": self._apply_fuzziness(query, fuzziness),
            "filter": self._filter_clauses(filters),
            "limit": top_k,
            "fields": self._requested_fields(with_score=True),
            "params": {
                "defType": "edismax",
                "qf": CONTENT_FIELD,
                "mm": "100%" if all_terms_must_match else "0%",
            },
        }

    @staticmethod
    def _scale_scores(documents: list[Document]) -> list[Document]:
        """
        Squash unbounded BM25 scores into `(0, 1)`.

        `replace` returns new documents rather than mutating the originals, so the rebuilt list is what
        gets returned - dropping it on the floor would make `scale_score` silently do nothing.
        """
        return [
            replace(document, score=1 / (1 + math.exp(-document.score / BM25_SCALING_FACTOR)))
            if document.score is not None
            else document
            for document in documents
        ]

    def _bm25_documents(self, payload: dict[str, Any], *, scale_score: bool) -> list[Document]:
        documents = self._documents_from_response(payload, with_score=True)
        return self._scale_scores(documents) if scale_score else documents

    def _bm25_retrieval(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        fuzziness: int = 0,
        scale_score: bool = False,
        all_terms_must_match: bool = False,
    ) -> list[Document]:
        """
        Retrieve documents matching `query` using Solr's BM25 similarity.

        :param query: the query string.
        :param filters: filters applied to the search.
        :param top_k: maximum number of documents to return.
        :param fuzziness: per-term edit distance. `0` disables fuzzy matching.
        :param scale_score: whether to scale scores into the `(0, 1)` range.
        :param all_terms_must_match: whether every query term must match.
        :returns: the matching documents, most relevant first.
        """
        if not query:
            msg = "query must be a non empty string"
            raise ValueError(msg)
        self._ensure_initialized()
        payload = self._solr_client.request(
            "POST",
            f"{self._core}/query",
            json_body=self._bm25_payload(
                query,
                filters=filters,
                top_k=top_k,
                fuzziness=fuzziness,
                all_terms_must_match=all_terms_must_match,
            ),
        )
        return self._bm25_documents(payload, scale_score=scale_score)

    async def _bm25_retrieval_async(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        fuzziness: int = 0,
        scale_score: bool = False,
        all_terms_must_match: bool = False,
    ) -> list[Document]:
        """
        Retrieve documents matching `query` using Solr's BM25 similarity.

        :param query: the query string.
        :param filters: filters applied to the search.
        :param top_k: maximum number of documents to return.
        :param fuzziness: per-term edit distance. `0` disables fuzzy matching.
        :param scale_score: whether to scale scores into the `(0, 1)` range.
        :param all_terms_must_match: whether every query term must match.
        :returns: the matching documents, most relevant first.
        """
        if not query:
            msg = "query must be a non empty string"
            raise ValueError(msg)
        await self._ensure_initialized_async()
        payload = await self._solr_client.request_async(
            "POST",
            f"{self._core}/query",
            json_body=self._bm25_payload(
                query,
                filters=filters,
                top_k=top_k,
                fuzziness=fuzziness,
                all_terms_must_match=all_terms_must_match,
            ),
        )
        return self._bm25_documents(payload, scale_score=scale_score)

    def _embedding_payload(
        self, query_embedding: list[float], *, filters: dict[str, Any] | None, top_k: int
    ) -> dict[str, Any]:
        vector = "[" + ",".join(repr(float(value)) for value in query_embedding) + "]"
        return {
            # `{!knn}` has to be the main query: inside `fq` Solr applies no implicit graph
            # pre-filter, and the search then returns fewer than `top_k` documents once filters bite.
            "query": f"{{!knn f={EMBEDDING_FIELD} topK={top_k}}}{vector}",
            "filter": self._filter_clauses(filters),
            "limit": top_k,
            "fields": self._requested_fields(with_score=True),
        }

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """
        Retrieve documents whose embeddings are closest to `query_embedding`.

        :param query_embedding: the query embedding.
        :param filters: filters applied to the search. They act as a k-NN graph pre-filter, so the
            search still returns up to `top_k` documents.
        :param top_k: maximum number of documents to return.
        :returns: the matching documents, most similar first.
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)
        self._ensure_initialized()
        payload = self._solr_client.request(
            "POST",
            f"{self._core}/query",
            json_body=self._embedding_payload(query_embedding, filters=filters, top_k=top_k),
        )
        return self._documents_from_response(payload, with_score=True)

    async def _embedding_retrieval_async(
        self,
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """
        Retrieve documents whose embeddings are closest to `query_embedding`.

        :param query_embedding: the query embedding.
        :param filters: filters applied to the search.
        :param top_k: maximum number of documents to return.
        :returns: the matching documents, most similar first.
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)
        await self._ensure_initialized_async()
        payload = await self._solr_client.request_async(
            "POST",
            f"{self._core}/query",
            json_body=self._embedding_payload(query_embedding, filters=filters, top_k=top_k),
        )
        return self._documents_from_response(payload, with_score=True)
