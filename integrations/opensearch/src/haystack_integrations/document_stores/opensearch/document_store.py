# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: FBT001, FBT002   boolean-type-hint-positional-argument and boolean-default-value-positional-argument

from collections.abc import Mapping
from math import exp
from typing import Any, Literal

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret
from opensearchpy import AsyncHttpConnection, AsyncOpenSearch, OpenSearch
from opensearchpy.helpers import async_bulk, bulk

from haystack_integrations.document_stores.opensearch.auth import AsyncAWSAuth, AWSAuth
from haystack_integrations.document_stores.opensearch.filters import normalize_filters
from haystack_integrations.document_stores.opensearch.opensearch_scripts import METADATA_SEARCH_JACCARD_SCRIPT

logger = logging.getLogger(__name__)

SPECIAL_FIELDS = {"content", "embedding", "id", "score", "sparse_embedding", "blob"}

Hosts = str | list[str | Mapping[str, str | int]]

# document scores are essentially unbounded and will be scaled to values between 0 and 1 if scale_score is set to
# True. Scaling uses the expit function (inverse of the logit function) after applying a scaling factor
# (e.g., BM25_SCALING_FACTOR for the bm25_retrieval method).
# Larger scaling factor decreases scaled scores. For example, an input of 10 is scaled to 0.99 with
# BM25_SCALING_FACTOR=2 but to 0.78 with BM25_SCALING_FACTOR=8 (default). The defaults were chosen empirically.
# Increase the default if most unscaled scores are larger than expected (>30) and otherwise would incorrectly
# all be mapped to scores ~1.
BM25_SCALING_FACTOR = 8

DEFAULT_SETTINGS = {"index.knn": True}
DEFAULT_MAX_CHUNK_BYTES = 100 * 1024 * 1024


class OpenSearchDocumentStore:
    """
    An instance of an OpenSearch database you can use to store all types of data.

    This document store is a thin wrapper around the OpenSearch client.
    It allows you to store and retrieve documents from an OpenSearch index.

    Usage example:
    ```python
    from haystack_integrations.document_stores.opensearch import (
        OpenSearchDocumentStore,
    )
    from haystack import Document

    document_store = OpenSearchDocumentStore(hosts="localhost:9200")

    document_store.write_documents(
        [
            Document(content="My first document", id="1"),
            Document(content="My second document", id="2"),
        ]
    )

    print(document_store.count_documents())
    # 2

    print(document_store.filter_documents())
    # [Document(id='1', content='My first document', ...), Document(id='2', content='My second document', ...)]
    ```
    """

    def __init__(
        self,
        *,
        hosts: Hosts | None = None,
        index: str = "default",
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
        embedding_dim: int = 768,
        return_embedding: bool = False,
        method: dict[str, Any] | None = None,
        mappings: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = DEFAULT_SETTINGS,
        create_index: bool = True,
        http_auth: Any = (
            Secret.from_env_var("OPENSEARCH_USERNAME", strict=False),  # noqa: B008
            Secret.from_env_var("OPENSEARCH_PASSWORD", strict=False),  # noqa: B008
        ),
        use_ssl: bool | None = None,
        verify_certs: bool | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Creates a new OpenSearchDocumentStore instance.

        The ``embeddings_dim``, ``method``, ``mappings``, and ``settings`` arguments are only used if the index does not
        exist and needs to be created. If the index already exists, its current configurations will be used.

        For more information on connection parameters, see the [official OpenSearch documentation](https://opensearch.org/docs/latest/clients/python-low-level/#connecting-to-opensearch)

        :param hosts: List of hosts running the OpenSearch client. Defaults to None
        :param index: Name of index in OpenSearch, if it doesn't exist it will be created. Defaults to "default"
        :param max_chunk_bytes: Maximum size of the requests in bytes. Defaults to 100MB
        :param embedding_dim: Dimension of the embeddings. Defaults to 768
        :param return_embedding:
            Whether to return the embedding of the retrieved Documents. This parameter also applies to the
            `filter_documents` and `filter_documents_async` methods.
        :param method: The method definition of the underlying configuration of the approximate k-NN algorithm. Please
            see the [official OpenSearch docs](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/#method-definitions)
            for more information. Defaults to None
        :param mappings: The mapping of how the documents are stored and indexed. Please see the [official OpenSearch docs](https://opensearch.org/docs/latest/field-types/)
            for more information. If None, it uses the embedding_dim and method arguments to create default mappings.
            Defaults to None
        :param settings: The settings of the index to be created. Please see the [official OpenSearch docs](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/#index-settings)
            for more information. Defaults to `{"index.knn": True}`.
        :param create_index: Whether to create the index if it doesn't exist. Defaults to True
        :param http_auth: http_auth param passed to the underlying connection class.
            For basic authentication with default connection class `Urllib3HttpConnection` this can be
            - a tuple of (username, password)
            - a list of [username, password]
            - a string of "username:password"
            If not provided, will read values from OPENSEARCH_USERNAME and OPENSEARCH_PASSWORD environment variables.
            For AWS authentication with `Urllib3HttpConnection` pass an instance of `AWSAuth`.
            Defaults to None
        :param use_ssl: Whether to use SSL. Defaults to None
        :param verify_certs: Whether to verify certificates. Defaults to None
        :param timeout: Timeout in seconds. Defaults to None
        :param **kwargs: Optional arguments that ``OpenSearch`` takes. For the full list of supported kwargs,
            see the [official OpenSearch reference](https://opensearch-project.github.io/opensearch-py/api-ref/clients/opensearch_client.html)
        """
        self._hosts = hosts
        self._index = index
        self._max_chunk_bytes = max_chunk_bytes
        self._embedding_dim = embedding_dim
        self._return_embedding = return_embedding
        self._method = method
        self._mappings = mappings or self._get_default_mappings()
        self._settings = settings
        self._create_index = create_index
        self._http_auth_are_secrets = False

        # Handle authentication
        if isinstance(http_auth, (tuple, list)) and len(http_auth) == 2:  # noqa: PLR2004
            username, password = http_auth
            if isinstance(username, Secret) and isinstance(password, Secret):
                self._http_auth_are_secrets = True
                username_val = username.resolve_value()
                password_val = password.resolve_value()
                http_auth = [username_val, password_val] if username_val and password_val else None

        self._http_auth = http_auth
        self._use_ssl = use_ssl
        self._verify_certs = verify_certs
        self._timeout = timeout
        self._kwargs = kwargs

        # Client is initialized lazily to prevent side effects when
        # the document store is instantiated.
        self._client: OpenSearch | None = None
        self._async_client: AsyncOpenSearch | None = None
        self._initialized = False

    def _get_default_mappings(self) -> dict[str, Any]:
        default_mappings: dict[str, Any] = {
            "properties": {
                "embedding": {"type": "knn_vector", "index": True, "dimension": self._embedding_dim},
                "content": {"type": "text"},
            },
            "dynamic_templates": [
                {
                    "strings": {
                        "match_mapping_type": "string",
                        "mapping": {"type": "keyword"},
                    }
                }
            ],
        }
        if self._method:
            default_mappings["properties"]["embedding"]["method"] = self._method
        return default_mappings

    def create_index(
        self,
        index: str | None = None,
        mappings: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an index in OpenSearch.

        Note that this method ignores the `create_index` argument from the constructor.

        :param index: Name of the index to create. If None, the index name from the constructor is used.
        :param mappings: The mapping of how the documents are stored and indexed. Please see the [official OpenSearch docs](https://opensearch.org/docs/latest/field-types/)
            for more information. If None, the mappings from the constructor are used.
        :param settings: The settings of the index to be created. Please see the [official OpenSearch docs](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/#index-settings)
            for more information. If None, the settings from the constructor are used.
        """
        self._ensure_initialized()
        assert self._client is not None

        if not index:
            index = self._index
        if not mappings:
            mappings = self._mappings
        if not settings:
            settings = self._settings

        if not self._client.indices.exists(index=index):
            self._client.indices.create(index=index, body={"mappings": mappings, "settings": settings})

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        # Handle http_auth serialization
        http_auth: list[dict[str, Any]] | dict[str, Any] | tuple[str, str] | list[str] | str = ""
        if isinstance(self._http_auth, list) and self._http_auth_are_secrets:
            # Recreate the Secret objects for serialization
            http_auth = [
                Secret.from_env_var("OPENSEARCH_USERNAME", strict=False).to_dict(),
                Secret.from_env_var("OPENSEARCH_PASSWORD", strict=False).to_dict(),
            ]
        elif isinstance(self._http_auth, AWSAuth):
            http_auth = self._http_auth.to_dict()
        else:
            http_auth = self._http_auth

        return default_to_dict(
            self,
            hosts=self._hosts,
            index=self._index,
            max_chunk_bytes=self._max_chunk_bytes,
            embedding_dim=self._embedding_dim,
            method=self._method,
            mappings=self._mappings,
            settings=self._settings,
            create_index=self._create_index,
            return_embedding=self._return_embedding,
            http_auth=http_auth,
            use_ssl=self._use_ssl,
            verify_certs=self._verify_certs,
            timeout=self._timeout,
            **self._kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenSearchDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        if http_auth := init_params.get("http_auth"):
            if isinstance(http_auth, dict):
                init_params["http_auth"] = AWSAuth.from_dict(http_auth)
            elif isinstance(http_auth, (tuple, list)):
                are_secrets = all(isinstance(item, dict) and "type" in item for item in http_auth)
                init_params["http_auth"] = [Secret.from_dict(item) for item in http_auth] if are_secrets else http_auth
        return default_from_dict(cls, data)

    def _ensure_initialized(self):
        # Ideally, we have a warm-up stage for document stores as well as components.
        if not self._client:
            self._client = OpenSearch(
                hosts=self._hosts,
                http_auth=self._http_auth,
                use_ssl=self._use_ssl,
                verify_certs=self._verify_certs,
                timeout=self._timeout,
                **self._kwargs,
            )
            self._initialized = True

            self._ensure_index_exists()

    async def _ensure_initialized_async(self):
        if not self._async_client:
            async_http_auth = AsyncAWSAuth(self._http_auth) if isinstance(self._http_auth, AWSAuth) else self._http_auth
            self._async_client = AsyncOpenSearch(
                hosts=self._hosts,
                http_auth=async_http_auth,
                use_ssl=self._use_ssl,
                verify_certs=self._verify_certs,
                timeout=self._timeout,
                # IAM Authentication requires AsyncHttpConnection:
                # https://github.com/opensearch-project/opensearch-py/blob/main/guides/auth.md#iam-authentication-with-an-async-client
                connection_class=AsyncHttpConnection,
                **self._kwargs,
            )
            self._initialized = True
            await self._ensure_index_exists_async()

    async def _ensure_index_exists_async(self):
        assert self._async_client is not None

        if await self._async_client.indices.exists(index=self._index):
            logger.debug(
                "The index '{index}' already exists. The `embedding_dim`, `method`, `mappings`, and "
                "`settings` values will be ignored.",
                index=self._index,
            )
        elif self._create_index:
            # Create the index if it doesn't exist
            body = {"mappings": self._mappings, "settings": self._settings}
            await self._async_client.indices.create(index=self._index, body=body)

    def _ensure_index_exists(self):
        assert self._client is not None

        if self._client.indices.exists(index=self._index):
            logger.debug(
                "The index '{index}' already exists. The `embedding_dim`, `method`, `mappings`, and "
                "`settings` values will be ignored.",
                index=self._index,
            )
        elif self._create_index:
            # Create the index if it doesn't exist
            body = {"mappings": self._mappings, "settings": self._settings}
            self._client.indices.create(index=self._index, body=body)

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        self._ensure_initialized()
        assert self._client is not None

        return self._client.count(index=self._index)["count"]

    async def count_documents_async(self) -> int:
        """
        Asynchronously returns the total number of documents in the document store.
        """
        await self._ensure_initialized_async()

        assert self._async_client is not None
        return (await self._async_client.count(index=self._index))["count"]

    @staticmethod
    def _deserialize_search_hits(hits: list[dict[str, Any]]) -> list[Document]:
        out = []
        for hit in hits:
            data = hit["_source"]
            if "highlight" in hit:
                data["metadata"]["highlighted"] = hit["highlight"]
            data["score"] = hit["_score"]
            out.append(Document.from_dict(data))

        return out

    def _prepare_filter_search_request(self, filters: dict[str, Any] | None) -> dict[str, Any]:
        search_kwargs: dict[str, Any] = {"size": 10_000}
        if filters:
            search_kwargs["query"] = {"bool": {"filter": normalize_filters(filters)}}

        # For some applications not returning the embedding can save a lot of bandwidth
        # if you don't need this data not retrieving it can be a good idea
        if not self._return_embedding:
            search_kwargs["_source"] = {"excludes": ["embedding"]}
        return search_kwargs

    def _search_documents(self, request_body: dict[str, Any]) -> list[Document]:
        assert self._client is not None
        search_results = self._client.search(index=self._index, body=request_body)
        return OpenSearchDocumentStore._deserialize_search_hits(search_results["hits"]["hits"])

    async def _search_documents_async(self, request_body: dict[str, Any]) -> list[Document]:
        assert self._async_client is not None
        search_results = await self._async_client.search(index=self._index, body=request_body)
        return OpenSearchDocumentStore._deserialize_search_hits(search_results["hits"]["hits"])

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        self._ensure_initialized()
        return self._search_documents(self._prepare_filter_search_request(filters))

    async def filter_documents_async(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Asynchronously returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        await self._ensure_initialized_async()

        return await self._search_documents_async(self._prepare_filter_search_request(filters))

    def _prepare_bulk_write_request(
        self,
        *,
        documents: list[Document],
        policy: DuplicatePolicy,
        is_async: bool,
        refresh: Literal["wait_for", True, False],
    ) -> dict[str, Any]:
        if len(documents) > 0 and not isinstance(documents[0], Document):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        action = "index" if policy == DuplicatePolicy.OVERWRITE else "create"
        opensearch_actions = []
        for doc in documents:
            doc_dict = doc.to_dict()

            # Extract routing from document metadata
            doc_routing = doc_dict.pop("_routing", None)

            if "sparse_embedding" in doc_dict:
                sparse_embedding = doc_dict.pop("sparse_embedding", None)
                if sparse_embedding:
                    logger.warning(
                        "Document {id} has the `sparse_embedding` field set,"
                        "but storing sparse embeddings in OpenSearch is not currently supported."
                        "The `sparse_embedding` field will be ignored.",
                        id=doc.id,
                    )

            action_dict = {
                "_op_type": action,
                "_id": doc.id,
                "_source": doc_dict,
            }

            if doc_routing is not None:
                action_dict["_routing"] = doc_routing

            opensearch_actions.append(action_dict)

        return {
            "client": self._client if not is_async else self._async_client,
            "actions": opensearch_actions,
            "refresh": refresh,
            "index": self._index,
            "raise_on_error": False,
            "max_chunk_bytes": self._max_chunk_bytes,
            "stats_only": False,
        }

    @staticmethod
    def _process_bulk_write_errors(errors: list[dict[str, Any]], policy: DuplicatePolicy) -> None:
        if len(errors) == 0:
            return

        duplicate_errors_ids = []
        other_errors = []
        for e in errors:
            # OpenSearch might not return a correctly formatted error, in that case we
            # treat it as a generic error
            if "create" not in e:
                other_errors.append(e)
                continue
            error_type = e["create"]["error"]["type"]
            if policy == DuplicatePolicy.FAIL and error_type == "version_conflict_engine_exception":
                duplicate_errors_ids.append(e["create"]["_id"])
            elif policy == DuplicatePolicy.SKIP and error_type == "version_conflict_engine_exception":
                # when the policy is skip, duplication errors are OK and we should not raise an exception
                continue
            else:
                other_errors.append(e)

        if len(duplicate_errors_ids) > 0:
            msg = f"IDs '{', '.join(duplicate_errors_ids)}' already exist in the document store."
            raise DuplicateDocumentError(msg)

        if len(other_errors) > 0:
            msg = f"Failed to write documents to OpenSearch. Errors:\n{other_errors}"
            raise DocumentStoreError(msg)

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
        refresh: Literal["wait_for", True, False] = "wait_for",
    ) -> int:
        """
        Writes documents to the document store.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :param refresh: Controls when changes are made visible to search operations.
            - `True`: Force refresh immediately after the operation.
            - `False`: Do not refresh (better performance for bulk operations).
            - `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
            For more details, see the [OpenSearch refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/index-document/).
        :raises DuplicateDocumentError: If a document with the same id already exists in the document store
             and the policy is set to `DuplicatePolicy.FAIL` (or not specified).
        :returns: The number of documents written to the document store.
        """
        self._ensure_initialized()

        bulk_params = self._prepare_bulk_write_request(
            documents=documents, policy=policy, is_async=False, refresh=refresh
        )
        documents_written, errors = bulk(**bulk_params)
        OpenSearchDocumentStore._process_bulk_write_errors(errors, policy)
        return documents_written

    async def write_documents_async(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
        refresh: Literal["wait_for", True, False] = "wait_for",
    ) -> int:
        """
        Asynchronously writes documents to the document store.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :param refresh: Controls when changes are made visible to search operations.
            - `True`: Force refresh immediately after the operation.
            - `False`: Do not refresh (better performance for bulk operations).
            - `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
            For more details, see the [OpenSearch refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/index-document/).
        :returns: The number of documents written to the document store.
        """
        await self._ensure_initialized_async()
        assert self._async_client is not None
        bulk_params = self._prepare_bulk_write_request(
            documents=documents, policy=policy, is_async=True, refresh=refresh
        )
        documents_written, errors = await async_bulk(**bulk_params)
        # since we call async_bulk with stats_only=False, errors is guaranteed to be a list (not int)
        OpenSearchDocumentStore._process_bulk_write_errors(errors=errors, policy=policy)  # type: ignore[arg-type]
        return documents_written

    @staticmethod
    def _deserialize_document(hit: dict[str, Any]) -> Document:
        """
        Creates a Document from the search hit provided.
        This is mostly useful in self.filter_documents().
        """
        data = hit["_source"]

        if "highlight" in hit:
            data["metadata"]["highlighted"] = hit["highlight"]
        data["score"] = hit["_score"]

        return Document.from_dict(data)

    def _prepare_bulk_delete_request(
        self,
        *,
        document_ids: list[str],
        is_async: bool,
        refresh: Literal["wait_for", True, False],
        routing: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        def action_generator():
            for id_ in document_ids:
                action = {"_op_type": "delete", "_id": id_}
                # Add routing if provided for this document ID
                if routing and id_ in routing and routing[id_] is not None:
                    action["_routing"] = routing[id_]
                yield action

        return {
            "client": self._client if not is_async else self._async_client,
            "actions": action_generator(),
            "refresh": refresh,
            "index": self._index,
            "raise_on_error": False,
            "max_chunk_bytes": self._max_chunk_bytes,
        }

    def delete_documents(
        self,
        document_ids: list[str],
        refresh: Literal["wait_for", True, False] = "wait_for",
        routing: dict[str, str] | None = None,
    ) -> None:
        """
        Deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        :param refresh: Controls when changes are made visible to search operations.
            - `True`: Force refresh immediately after the operation.
            - `False`: Do not refresh (better performance for bulk operations).
            - `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
            For more details, see the [OpenSearch refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/index-document/).
        :param routing: A dictionary mapping document IDs to their routing values.
            Routing values are used to determine the shard where documents are stored.
            If provided, the routing value for each document will be used during deletion.
        """

        self._ensure_initialized()

        bulk(
            **self._prepare_bulk_delete_request(
                document_ids=document_ids, is_async=False, refresh=refresh, routing=routing
            )
        )

    async def delete_documents_async(
        self,
        document_ids: list[str],
        refresh: Literal["wait_for", True, False] = "wait_for",
        routing: dict[str, str] | None = None,
    ) -> None:
        """
        Asynchronously deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        :param refresh: Controls when changes are made visible to search operations.
            - `True`: Force refresh immediately after the operation.
            - `False`: Do not refresh (better performance for bulk operations).
            - `"wait_for"`: Wait for the next refresh cycle (default, ensures read-your-writes consistency).
            For more details, see the [OpenSearch refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/index-document/).
        :param routing: A dictionary mapping document IDs to their routing values.
            Routing values are used to determine the shard where documents are stored.
            If provided, the routing value for each document will be used during deletion.
        """
        await self._ensure_initialized_async()
        assert self._async_client is not None

        await async_bulk(
            **self._prepare_bulk_delete_request(
                document_ids=document_ids, is_async=True, refresh=refresh, routing=routing
            )
        )

    def _prepare_delete_all_request(self, *, refresh: bool) -> dict[str, Any]:
        return {
            "index": self._index,
            "body": {"query": {"match_all": {}}},  # Delete all documents
            "wait_for_completion": True,  # Always wait to ensure documents are deleted before returning
            "refresh": refresh,
        }

    def delete_all_documents(self, recreate_index: bool = False, refresh: bool = True) -> None:
        """
        Deletes all documents in the document store.

        :param recreate_index: If True, the index will be deleted and recreated with the original mappings and
            settings. If False, all documents will be deleted using the `delete_by_query` API.
        :param refresh: If True, OpenSearch refreshes all shards involved in the delete by query after the request
            completes. If False, no refresh is performed. For more details, see the
            [OpenSearch delete_by_query refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/delete-by-query/).
        """
        self._ensure_initialized()
        assert self._client is not None

        try:
            if recreate_index:
                # get the current index mappings and settings
                index_name = self._index
                index_info = self._client.indices.get(index=self._index)
                body = {
                    "mappings": index_info[index_name]["mappings"],
                    "settings": index_info[index_name]["settings"],
                }
                body["settings"]["index"].pop("uuid", None)
                body["settings"]["index"].pop("creation_date", None)
                body["settings"]["index"].pop("provided_name", None)
                body["settings"]["index"].pop("version", None)
                self._client.indices.delete(index=self._index)
                self._client.indices.create(index=self._index, body=body)
                logger.info(
                    "The index '{index}' recreated with the original mappings and settings.",
                    index=self._index,
                )

            else:
                result = self._client.delete_by_query(**self._prepare_delete_all_request(refresh=refresh))
                logger.info(
                    "Deleted all the {n_docs} documents from the index '{index}'.",
                    index=self._index,
                    n_docs=result["deleted"],
                )
        except Exception as e:
            msg = f"Failed to delete all documents from OpenSearch: {e!s}"
            raise DocumentStoreError(msg) from e

    async def delete_all_documents_async(self, recreate_index: bool = False, refresh: bool = True) -> None:
        """
        Asynchronously deletes all documents in the document store.

        :param recreate_index: If True, the index will be deleted and recreated with the original mappings and
            settings. If False, all documents will be deleted using the `delete_by_query` API.
        :param refresh: If True, OpenSearch refreshes all shards involved in the delete by query after the request
            completes. If False, no refresh is performed. For more details, see the
            [OpenSearch delete_by_query refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/delete-by-query/).
        """
        await self._ensure_initialized_async()
        assert self._async_client is not None

        try:
            if recreate_index:
                # get the current index mappings and settings
                index_name = self._index
                index_info = await self._async_client.indices.get(index=self._index)
                body = {
                    "mappings": index_info[index_name]["mappings"],
                    "settings": index_info[index_name]["settings"],
                }
                body["settings"]["index"].pop("uuid", None)
                body["settings"]["index"].pop("creation_date", None)
                body["settings"]["index"].pop("provided_name", None)
                body["settings"]["index"].pop("version", None)

                await self._async_client.indices.delete(index=self._index)
                await self._async_client.indices.create(index=self._index, body=body)
            else:
                # use delete_by_query for more efficient deletion without index recreation
                await self._async_client.delete_by_query(**self._prepare_delete_all_request(refresh=refresh))

        except Exception as e:
            msg = f"Failed to delete all documents from OpenSearch: {e!s}"
            raise DocumentStoreError(msg) from e

    def delete_by_filter(self, filters: dict[str, Any], refresh: bool = False) -> int:
        """
        Deletes all documents that match the provided filters.

        :param filters: The filters to apply to select documents for deletion.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param refresh: If True, OpenSearch refreshes all shards involved in the delete by query after the request
            completes so that subsequent reads (e.g. count_documents) see the update. If False, no refresh is
            performed (better for bulk deletes). For more details, see the
            [OpenSearch delete_by_query refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/delete-by-query/).
        :returns: The number of documents deleted.
        """
        self._ensure_initialized()
        assert self._client is not None

        try:
            normalized_filters = normalize_filters(filters)
            body = {"query": {"bool": {"filter": normalized_filters}}}

            result = self._client.delete_by_query(index=self._index, body=body, refresh=refresh)
            deleted_count = result.get("deleted", 0)
            logger.info(
                "Deleted {n_docs} documents from index '{index}' using filters.",
                n_docs=deleted_count,
                index=self._index,
            )
            return deleted_count
        except Exception as e:
            msg = f"Failed to delete documents by filter from OpenSearch: {e!s}"
            raise DocumentStoreError(msg) from e

    async def delete_by_filter_async(self, filters: dict[str, Any], refresh: bool = False) -> int:
        """
        Asynchronously deletes all documents that match the provided filters.

        :param filters: The filters to apply to select documents for deletion.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param refresh: If True, OpenSearch refreshes all shards involved in the delete by query after the request
            completes so that subsequent reads see the update. If False, no refresh is performed. For more details,
            see the [OpenSearch delete_by_query refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/delete-by-query/).
        :returns: The number of documents deleted.
        """
        await self._ensure_initialized_async()
        assert self._async_client is not None

        try:
            normalized_filters = normalize_filters(filters)
            body = {"query": {"bool": {"filter": normalized_filters}}}
            result = await self._async_client.delete_by_query(index=self._index, body=body, refresh=refresh)
            deleted_count = result.get("deleted", 0)
            logger.info(
                "Deleted {n_docs} documents from index '{index}' using filters.",
                n_docs=deleted_count,
                index=self._index,
            )
            return deleted_count
        except Exception as e:
            msg = f"Failed to delete documents by filter from OpenSearch: {e!s}"
            raise DocumentStoreError(msg) from e

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any], refresh: bool = False) -> int:
        """
        Updates the metadata of all documents that match the provided filters.

        :param filters: The filters to apply to select documents for updating.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param meta: The metadata fields to update.
        :param refresh: If True, OpenSearch refreshes all shards involved in the update by query after the request
            completes. If False, no refresh is performed. For more details, see the
            [OpenSearch update_by_query refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/update-by-query/).
        :returns: The number of documents updated.
        """
        self._ensure_initialized()
        assert self._client is not None

        try:
            normalized_filters = normalize_filters(filters)
            # Build the update script to modify metadata fields
            # Documents are stored with flattened metadata, so update fields directly in ctx._source
            update_script_lines = []
            for key in meta.keys():
                update_script_lines.append(f"ctx._source.{key} = params.{key};")
            update_script = " ".join(update_script_lines)

            body = {
                "query": {"bool": {"filter": normalized_filters}},
                "script": {"source": update_script, "params": meta, "lang": "painless"},
            }
            result = self._client.update_by_query(index=self._index, body=body, refresh=refresh)
            updated_count = result.get("updated", 0)
            logger.info(
                "Updated {n_docs} documents in index '{index}' using filters.",
                n_docs=updated_count,
                index=self._index,
            )
            return updated_count
        except Exception as e:
            msg = f"Failed to update documents by filter in OpenSearch: {e!s}"
            raise DocumentStoreError(msg) from e

    async def update_by_filter_async(self, filters: dict[str, Any], meta: dict[str, Any], refresh: bool = False) -> int:
        """
        Asynchronously updates the metadata of all documents that match the provided filters.

        :param filters: The filters to apply to select documents for updating.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param meta: The metadata fields to update.
        :param refresh: If True, OpenSearch refreshes all shards involved in the update by query after the request
            completes. If False, no refresh is performed. For more details, see the
            [OpenSearch update_by_query refresh documentation](https://opensearch.org/docs/latest/api-reference/document-apis/update-by-query/).
        :returns: The number of documents updated.
        """
        await self._ensure_initialized_async()
        assert self._async_client is not None

        try:
            normalized_filters = normalize_filters(filters)
            # Build the update script to modify metadata fields
            # Documents are stored with flattened metadata, so update fields directly in ctx._source
            update_script_lines = []
            for key in meta.keys():
                update_script_lines.append(f"ctx._source.{key} = params.{key};")
            update_script = " ".join(update_script_lines)

            body = {
                "query": {"bool": {"filter": normalized_filters}},
                "script": {"source": update_script, "params": meta, "lang": "painless"},
            }
            result = await self._async_client.update_by_query(index=self._index, body=body, refresh=refresh)
            updated_count = result.get("updated", 0)
            logger.info(
                "Updated {n_docs} documents in index '{index}' using filters.",
                n_docs=updated_count,
                index=self._index,
            )
            return updated_count
        except Exception as e:
            msg = f"Failed to update documents by filter in OpenSearch: {e!s}"
            raise DocumentStoreError(msg) from e

    def _prepare_bm25_search_request(
        self,
        *,
        query: str,
        filters: dict[str, Any] | None,
        fuzziness: int | str,
        top_k: int,
        all_terms_must_match: bool,
        custom_query: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not query:
            body: dict[str, Any] = {"query": {"bool": {"must": {"match_all": {}}}}}
            if filters:
                body["query"]["bool"]["filter"] = normalize_filters(filters)

        if isinstance(custom_query, dict):
            body = self._render_custom_query(
                custom_query,
                {
                    "$query": query,
                    "$filters": normalize_filters(filters) if filters else None,
                },
            )

        else:
            operator = "AND" if all_terms_must_match else "OR"
            body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fuzziness": fuzziness,
                                    "type": "most_fields",
                                    "operator": operator,
                                }
                            }
                        ]
                    }
                },
            }

            if filters:
                body["query"]["bool"]["filter"] = normalize_filters(filters)

        body["size"] = top_k

        # For some applications not returning the embedding can save a lot of bandwidth
        # if you don't need this data not retrieving it can be a good idea
        if not self._return_embedding:
            body["_source"] = {"excludes": ["embedding"]}

        return body

    @staticmethod
    def _postprocess_bm25_search_results(*, results: list[Document], scale_score: bool) -> None:
        if not scale_score:
            return

        for doc in results:
            if doc.score is None:
                continue
            doc.score = float(1 / (1 + exp(-(doc.score / float(BM25_SCALING_FACTOR)))))

    def _bm25_retrieval(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        fuzziness: int | str = "AUTO",
        top_k: int = 10,
        scale_score: bool = False,
        all_terms_must_match: bool = False,
        custom_query: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieves documents that match the provided `query` using the BM25 search algorithm.

        OpenSearch by defaults uses BM25 search algorithm.
        Even though this method is called `_bm25_retrieval` it searches for `query`
        using the search algorithm `_client` was configured with.

        This method is not meant to be part of the public interface of
        `OpenSearchDocumentStore` nor called directly.
        `OpenSearchBM25Retriever` uses this method directly and is the public interface for it.

        See `OpenSearchBM25Retriever` for more information.
        """
        self._ensure_initialized()

        search_params = self._prepare_bm25_search_request(
            query=query,
            filters=filters,
            fuzziness=fuzziness,
            top_k=top_k,
            all_terms_must_match=all_terms_must_match,
            custom_query=custom_query,
        )
        documents = self._search_documents(search_params)
        OpenSearchDocumentStore._postprocess_bm25_search_results(results=documents, scale_score=scale_score)
        return documents

    async def _bm25_retrieval_async(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        fuzziness: str = "AUTO",
        top_k: int = 10,
        scale_score: bool = False,
        all_terms_must_match: bool = False,
        custom_query: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Asynchronously retrieves documents that match the provided `query` using the BM25 search algorithm.

        OpenSearch by defaults uses BM25 search algorithm.
        Even though this method is called `_bm25_retrieval` it searches for `query`
        using the search algorithm `_client` was configured with.

        This method is not meant to be part of the public interface of
        `OpenSearchDocumentStore` nor called directly.
        `OpenSearchBM25Retriever` uses this method directly and is the public interface for it.

        See `OpenSearchBM25Retriever` for more information.
        """

        await self._ensure_initialized_async()
        assert self._async_client is not None

        search_params = self._prepare_bm25_search_request(
            query=query,
            filters=filters,
            fuzziness=fuzziness,
            top_k=top_k,
            all_terms_must_match=all_terms_must_match,
            custom_query=custom_query,
        )
        documents = await self._search_documents_async(search_params)
        OpenSearchDocumentStore._postprocess_bm25_search_results(results=documents, scale_score=scale_score)
        return documents

    @staticmethod
    def _build_metadata_search_query(
        query_part: str,
        fields: list[str],
        mode: Literal["strict", "fuzzy"],
        fuzziness: int | Literal["AUTO"] = 2,
        prefix_length: int = 0,
        max_expansions: int = 200,
        tie_breaker: float = 0.7,
        jaccard_n: int = 3,
    ) -> dict[str, Any]:
        """
        Build an OpenSearch query for metadata search.

        The query uses a script_score query with a Jaccard similarity script (n-gram based)
        to score results in both "strict" and "fuzzy" modes. The mode only affects the query
        structure used to find matching documents, while the Jaccard similarity script is used
        to rank/score all results. The n-gram size is controlled by jaccard_n.

        :param query_part: The cleaned query part to search for.
        :param fields: List of metadata field names to search within.
        :param mode: Search mode. "strict" uses prefix and wildcard matching,
            "fuzzy" uses fuzzy matching with dis_max queries.
        :param fuzziness: Maximum allowed Damerau-Levenshtein distance (edit distance) for fuzzy matching.
            Accepts an integer (e.g., 0, 1, 2) or "AUTO" which chooses based on term length.
            Default is 2. Only applies when mode is "fuzzy".
        :param prefix_length: Number of leading characters that must match exactly before fuzzy matching applies.
            Default is 0 (no prefix requirement). Only applies when mode is "fuzzy".
        :param max_expansions: Maximum number of term variations the fuzzy query can generate.
            Default is 200. Only applies when mode is "fuzzy".
        :param tie_breaker: Weight (0..1) for other matching clauses in dis_max; boosts docs matching multiple clauses.
            Default 0.7. Only applies when mode is "fuzzy".
        :param jaccard_n: N-gram size for Jaccard similarity scoring. Default 3; larger n favors longer token matches.
        :returns: OpenSearch query dictionary with script_score using Jaccard similarity.
        """
        if mode == "strict":
            # Strict mode: prefix and wildcard matching with Jaccard similarity
            should_clauses: list[dict[str, Any]] = []
            for field in fields:
                should_clauses.append({"prefix": {field: query_part}})
                should_clauses.append({"wildcard": {field: {"value": f"*{query_part}*", "case_insensitive": True}}})

            return {
                "script_score": {
                    "query": {
                        "bool": {
                            "filter": [{"exists": {"field": field}} for field in fields],
                            "should": should_clauses,
                            "minimum_should_match": 1,
                        }
                    },
                    "script": {
                        "lang": "painless",
                        "source": METADATA_SEARCH_JACCARD_SCRIPT,
                        "params": {"field": fields[0], "q": query_part, "n": jaccard_n},
                    },
                }
            }
        else:
            # Fuzzy mode: fuzzy matching with dis_max queries
            dis_max_queries: list[dict[str, Any]] = []
            for field in fields:
                dis_max_queries.append(
                    {
                        "match": {
                            field: {
                                "query": query_part,
                                "fuzziness": fuzziness,
                                "prefix_length": prefix_length,
                                "max_expansions": max_expansions,
                            }
                        }
                    }
                )
                dis_max_queries.append({"wildcard": {f"{field}.keyword": {"value": f"*{query_part}*"}}})
                # Use fuzziness value for query_string if it's an integer, otherwise default to 2
                fuzziness_value = fuzziness if isinstance(fuzziness, int) else 2
                dis_max_queries.append(
                    {
                        "query_string": {
                            "fields": [field],
                            "query": f"*{query_part}~{fuzziness_value}*",
                            "analyze_wildcard": True,
                        }
                    }
                )

            return {
                "script_score": {
                    "query": {
                        "dis_max": {
                            "tie_breaker": tie_breaker,
                            "queries": dis_max_queries,
                        }
                    },
                    "script": {
                        "lang": "painless",
                        "source": METADATA_SEARCH_JACCARD_SCRIPT,
                        "params": {"field": fields[0], "q": query_part, "n": jaccard_n},
                    },
                }
            }

    @staticmethod
    def _apply_metadata_search_filters(
        os_query: dict[str, Any], normalized_filters: list[dict[str, Any]], mode: Literal["strict", "fuzzy"]
    ) -> None:
        """
        Apply filters to a metadata search query.

        :param os_query: The OpenSearch query dictionary to modify.
        :param normalized_filters: Normalized filters to apply.
        :param mode: Search mode to determine how to apply filters.
        """
        if mode == "strict":
            bool_query = os_query["script_score"]["query"]["bool"]
            filter_list = bool_query.get("filter", [])
            filter_list.extend(normalized_filters)
        else:
            # For fuzzy mode, wrap the dis_max query in a bool query to add filters
            original_query = os_query["script_score"]["query"]
            os_query["script_score"]["query"] = {
                "bool": {
                    "must": [original_query],
                    "filter": normalized_filters,
                }
            }

    @staticmethod
    def _apply_multi_field_boosting(
        hit_list: list[dict[str, Any]], query: str, fields: list[str], exact_match_weight: float
    ) -> None:
        """
        Apply multi-field exact match boosting to hits.

        :param hit_list: List of search hits to modify.
        :param query: The full query string (may contain comma-separated parts).
        :param fields: List of metadata fields to check.
        :param exact_match_weight: Weight to add for multi-field matches.
        """
        for hit in hit_list:
            matched = 0
            for query_part in query.split(","):
                query_part_clean = query_part.strip()
                if not query_part_clean:
                    continue
                for field in fields:
                    if field in hit["_source"]:
                        if query_part_clean.lower() in str(hit["_source"][field]).lower():
                            matched += 1
            if matched > 1:
                hit["_score"] = hit["_score"] + exact_match_weight * matched

    @staticmethod
    def _process_metadata_search_results(
        hit_list: list[dict[str, Any]], fields: list[str], top_k: int
    ) -> list[dict[str, Any]]:
        """
        Process and deduplicate metadata search results.

        :param hit_list: List of search hits.
        :param fields: List of metadata fields to include in results.
        :param top_k: Maximum number of results to return.
        :returns: Deduplicated list of metadata dictionaries.
        """
        # Sort, deduplicate, and filter fields
        sorted_hit_list = sorted(hit_list, key=lambda x: x["_score"], reverse=True)
        top_k_hit_list = sorted_hit_list[:top_k]

        # Extract only specified fields
        filtered_results = [{k: v for k, v in hit["_source"].items() if k in fields} for hit in top_k_hit_list]

        # Deduplicate
        deduplicated = []
        for x in filtered_results:
            if x not in deduplicated:
                deduplicated.append(x)

        return deduplicated

    def _metadata_search(
        self,
        query: str,
        fields: list[str],
        *,
        mode: Literal["strict", "fuzzy"] = "fuzzy",
        top_k: int = 20,
        exact_match_weight: float = 0.6,
        fuzziness: int | Literal["AUTO"] = 2,
        prefix_length: int = 0,
        max_expansions: int = 200,
        tie_breaker: float = 0.7,
        jaccard_n: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search across multiple metadata fields with custom scoring and ranking.

        This method searches specified metadata fields for matches to a given query, ranks the results based on
        relevance using Jaccard similarity (n-gram based, see jaccard_n), and returns the top-k results containing only
        the specified metadata fields.

        The Jaccard similarity is computed server-side via a Painless script in both "strict" and "fuzzy" modes.
        The mode parameter only affects the query structure (prefix/wildcard vs fuzzy matching), while the
        Jaccard similarity script is used to score all results regardless of mode.

        :param query: The search query string, which can contain multiple comma-separated parts.
            Each part will be searched across all specified fields.
        :param fields: List of metadata field names to search within.
        :param mode: Search mode. "strict" uses prefix and wildcard matching,
            "fuzzy" uses fuzzy matching with dis_max queries. Default is "fuzzy".
            Both modes use Jaccard similarity for scoring results.
        :param top_k: Maximum number of top results to return based on relevance. Default is 20.
            The search retrieves up to 1000 hits from OpenSearch, then applies boosting and filters
            the results to the top_k most relevant matches.
        :param exact_match_weight: Weight to boost the score of exact matches in metadata fields.
            Default is 0.6. Applied after the search executes, in addition to Jaccard similarity scoring.
        :param fuzziness: Maximum allowed Damerau-Levenshtein distance (edit distance) for fuzzy matching.
            Accepts an integer (e.g., 0, 1, 2) or "AUTO" which chooses based on term length.
            Default is 2. Only applies when mode is "fuzzy".
        :param prefix_length: Number of leading characters that must match exactly before fuzzy matching applies.
            Default is 0 (no prefix requirement). Only applies when mode is "fuzzy".
        :param max_expansions: Maximum number of term variations the fuzzy query can generate.
            Default is 200. Only applies when mode is "fuzzy".
        :param tie_breaker: Weight (0..1) for other matching clauses; boosts docs matching multiple clauses.
            Default 0.7. Only applies when mode is "fuzzy".
        :param jaccard_n: N-gram size for Jaccard similarity scoring. Default 3; larger n favors longer token matches.
        :param filters: Additional filters to apply to the search query.
        :returns: List of dictionaries containing only the specified metadata fields,
            ranked by relevance score.

        Example:
            ```python
            # Search for documents with metadata matching "Python, active"
            results = document_store._metadata_search(
                query="Python, active",
                fields=["category", "status", "language"],
                mode="fuzzy",
                top_k=10
            )
            # Returns: [{"category": "Python", "status": "active", ...}, ...]
            ```
        """
        self._ensure_initialized()
        assert self._client is not None

        if not fields:
            return []

        hit_list = []

        # Split query by commas and search each part
        for query_part in query.split(","):
            query_part_clean = query_part.strip()
            if not query_part_clean:
                continue

            # Build query
            os_query = self._build_metadata_search_query(
                query_part_clean, fields, mode, fuzziness, prefix_length, max_expansions, tie_breaker, jaccard_n
            )

            # Add filters if provided
            if filters:
                normalized_filters = normalize_filters(filters)
                self._apply_metadata_search_filters(os_query, [normalized_filters], mode)

            body = {"size": 1000, "query": os_query}

            # Execute search
            try:
                response = self._client.search(index=self._index, body=body)
                hits = response["hits"]["hits"]
                hit_list.extend(hits)
            except Exception as e:
                msg = f"Failed to execute metadata search in OpenSearch: {e!s}"
                raise DocumentStoreError(msg) from e

        # Add multi-field exact match boosting
        if exact_match_weight > 0:
            self._apply_multi_field_boosting(hit_list, query, fields, exact_match_weight)

        # Process and return results
        return self._process_metadata_search_results(hit_list, fields, top_k)

    async def _metadata_search_async(
        self,
        query: str,
        fields: list[str],
        *,
        mode: Literal["strict", "fuzzy"] = "fuzzy",
        top_k: int = 20,
        exact_match_weight: float = 0.6,
        fuzziness: int | Literal["AUTO"] = 2,
        prefix_length: int = 0,
        max_expansions: int = 200,
        tie_breaker: float = 0.7,
        jaccard_n: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Asynchronously search across multiple metadata fields with custom scoring and ranking.

        This method searches specified metadata fields for matches to a given query,
        ranks the results based on relevance using Jaccard similarity (n-gram based, see jaccard_n),
        and returns the top-k results containing only the specified metadata fields.

        The Jaccard similarity is computed server-side via a Painless script in both "strict" and "fuzzy" modes.
        The mode parameter only affects the query structure (prefix/wildcard vs fuzzy matching), while the
        Jaccard similarity script is used to score all results regardless of mode.

        :param query: The search query string, which can contain multiple comma-separated parts.
            Each part will be searched across all specified fields.
        :param fields: List of metadata field names to search within.
        :param mode: Search mode. "strict" uses prefix and wildcard matching,
            "fuzzy" uses fuzzy matching with dis_max queries. Default is "fuzzy".
            Both modes use Jaccard similarity for scoring results.
        :param top_k: Maximum number of top results to return based on relevance. Default is 20.
            The search retrieves up to 1000 hits from OpenSearch, then applies boosting and filters
            the results to the top_k most relevant matches.
        :param exact_match_weight: Weight to boost the score of exact matches in metadata fields.
            Default is 0.6. Applied after the search executes, in addition to Jaccard similarity scoring.
        :param fuzziness: Maximum allowed Damerau-Levenshtein distance (edit distance) for fuzzy matching.
            Accepts an integer (e.g., 0, 1, 2) or "AUTO" which chooses based on term length.
            Default is 2. Only applies when mode is "fuzzy".
        :param prefix_length: Number of leading characters that must match exactly before fuzzy matching applies.
            Default is 0 (no prefix requirement). Only applies when mode is "fuzzy".
        :param max_expansions: Maximum number of term variations the fuzzy query can generate.
            Default is 200. Only applies when mode is "fuzzy".
        :param tie_breaker: Weight (0..1) for other matching clauses; boosts docs matching multiple clauses.
            Default 0.7. Only applies when mode is "fuzzy".
        :param jaccard_n: N-gram size for Jaccard similarity scoring. Default 3; larger n favors longer token matches.
        :param filters: Additional filters to apply to the search query.
        :returns: List of dictionaries containing only the specified metadata fields,
            ranked by relevance score.

        Example:
            ```python
            # Search for documents with metadata matching "Python, active"
            results = await document_store._metadata_search_async(
                query="Python, active",
                fields=["category", "status", "language"],
                mode="fuzzy",
                top_k=10
            )
            # Returns: [{"category": "Python", "status": "active", ...}, ...]
            ```
        """
        await self._ensure_initialized_async()
        assert self._async_client is not None

        if not fields:
            return []

        hit_list = []

        # Split query by commas and search each part
        for query_part in query.split(","):
            query_part_clean = query_part.strip()
            if not query_part_clean:
                continue

            # Build query
            os_query = self._build_metadata_search_query(
                query_part_clean, fields, mode, fuzziness, prefix_length, max_expansions, tie_breaker, jaccard_n
            )

            # Add filters if provided
            if filters:
                normalized_filters = normalize_filters(filters)
                self._apply_metadata_search_filters(os_query, [normalized_filters], mode)

            body = {"size": 1000, "query": os_query}

            # Execute search
            try:
                response = await self._async_client.search(index=self._index, body=body)
                hits = response["hits"]["hits"]
                hit_list.extend(hits)
            except Exception as e:
                msg = f"Failed to execute metadata search in OpenSearch: {e!s}"
                raise DocumentStoreError(msg) from e

        # Add multi-field exact match boosting
        if exact_match_weight > 0:
            self._apply_multi_field_boosting(hit_list, query, fields, exact_match_weight)

        # Process and return results
        return self._process_metadata_search_results(hit_list, fields, top_k)

    def _prepare_embedding_search_request(
        self,
        *,
        query_embedding: list[float],
        filters: dict[str, Any] | None,
        top_k: int,
        custom_query: dict[str, Any] | None,
        efficient_filtering: bool = False,
        search_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        body: dict[str, Any]
        if isinstance(custom_query, dict):
            body = self._render_custom_query(
                custom_query,
                {
                    "$query_embedding": query_embedding,
                    "$filters": normalize_filters(filters) if filters else None,
                },
            )

        else:
            body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_embedding,
                                        "k": top_k,
                                        **(search_kwargs or {}),
                                    }
                                }
                            }
                        ],
                    }
                },
            }

            if filters:
                if efficient_filtering:
                    body["query"]["bool"]["must"][0]["knn"]["embedding"]["filter"] = normalize_filters(filters)
                else:
                    body["query"]["bool"]["filter"] = normalize_filters(filters)

        body["size"] = top_k

        # For some applications not returning the embedding can save a lot of bandwidth
        # if you don't need this data not retrieving it can be a good idea
        if not self._return_embedding:
            body["_source"] = {"excludes": ["embedding"]}

        return body

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        custom_query: dict[str, Any] | None = None,
        efficient_filtering: bool = False,
        search_kwargs: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.
        It uses the OpenSearch's Approximate k-Nearest Neighbors search algorithm.

        This method is not meant to be part of the public interface of
        `OpenSearchDocumentStore` nor called directly.
        `OpenSearchEmbeddingRetriever` uses this method directly and is the public interface for it.

        See `OpenSearchEmbeddingRetriever` for more information.
        """
        self._ensure_initialized()

        search_params = self._prepare_embedding_search_request(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            custom_query=custom_query,
            efficient_filtering=efficient_filtering,
            search_kwargs=search_kwargs,
        )
        return self._search_documents(search_params)

    async def _embedding_retrieval_async(
        self,
        query_embedding: list[float],
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        custom_query: dict[str, Any] | None = None,
        efficient_filtering: bool = False,
        search_kwargs: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Asynchronously retrieves documents that are most similar to the query embedding using a vector similarity
        metric. It uses the OpenSearch's Approximate k-Nearest Neighbors search algorithm.

        This method is not meant to be part of the public interface of
        `OpenSearchDocumentStore` nor called directly.
        `OpenSearchEmbeddingRetriever` uses this method directly and is the public interface for it.

        See `OpenSearchEmbeddingRetriever` for more information.
        """
        await self._ensure_initialized_async()
        assert self._async_client is not None

        search_params = self._prepare_embedding_search_request(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            custom_query=custom_query,
            efficient_filtering=efficient_filtering,
            search_kwargs=search_kwargs,
        )
        return await self._search_documents_async(search_params)

    def _render_custom_query(self, custom_query: Any, substitutions: dict[str, Any]) -> Any:
        """
        Recursively replaces the placeholders in the custom_query with the actual values.

        :param custom_query: The custom query to replace the placeholders in.
        :param substitutions: The dictionary containing the actual values to replace the placeholders with.
        :returns: The custom query with the placeholders replaced.
        """
        if isinstance(custom_query, dict):
            return {key: self._render_custom_query(value, substitutions) for key, value in custom_query.items()}
        elif isinstance(custom_query, list):
            return [self._render_custom_query(entry, substitutions) for entry in custom_query]
        elif isinstance(custom_query, str):
            return substitutions.get(custom_query, custom_query)

        return custom_query

    def count_documents_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Returns the number of documents that match the provided filters.

        :param filters: The filters to apply to count documents.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents that match the filters.
        """
        self._ensure_initialized()
        assert self._client is not None

        normalized_filters = normalize_filters(filters)
        body = {"query": {"bool": {"filter": normalized_filters}}}
        return self._client.count(index=self._index, body=body)["count"]

    async def count_documents_by_filter_async(self, filters: dict[str, Any]) -> int:
        """
        Asynchronously returns the number of documents that match the provided filters.

        :param filters: The filters to apply to count documents.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents that match the filters.
        """
        await self._ensure_initialized_async()
        assert self._async_client is not None

        normalized_filters = normalize_filters(filters)
        body = {"query": {"bool": {"filter": normalized_filters}}}
        return (await self._async_client.count(index=self._index, body=body))["count"]

    @staticmethod
    def _build_cardinality_aggregations(index_mapping: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        """
        Builds cardinality aggregations for specified metadata fields in the index mapping.

        :param index_mapping: The index mapping containing field definitions.
        :param fields: List of field names to build aggregations for.
        :returns: Dictionary of cardinality aggregations.

        See: https://docs.opensearch.org/latest/aggregations/metric/cardinality/
        """
        aggs = {}
        for field_name in fields:
            if field_name not in SPECIAL_FIELDS and field_name in index_mapping:
                aggs[f"{field_name}_cardinality"] = {"cardinality": {"field": field_name}}
        return aggs

    @staticmethod
    def _build_distinct_values_query_body(filters: dict[str, Any] | None, aggs: dict[str, Any]) -> dict[str, Any]:
        """
        Builds the query body for distinct values counting with filters and aggregations.
        """
        if filters:
            normalized_filters = normalize_filters(filters)
            return {
                "query": {"bool": {"filter": normalized_filters}},
                "aggs": aggs,
                "size": 0,  # We only need aggregations, not documents
            }
        else:
            # No filters - match all documents
            return {
                "query": {"match_all": {}},
                "aggs": aggs,
                "size": 0,  # We only need aggregations, not documents
            }

    @staticmethod
    def _extract_distinct_counts_from_aggregations(
        aggregations: dict[str, Any], index_mapping: dict[str, Any], fields: list[str]
    ) -> dict[str, int]:
        """
        Extracts distinct value counts from search result aggregations.

        :param aggregations: The aggregations result from the search query.
        :param index_mapping: The index mapping containing field definitions.
        :param fields: List of field names to extract counts for.
        :returns: Dictionary mapping field names to their distinct value counts.
        """
        distinct_counts = {}
        for field_name in fields:
            if field_name not in SPECIAL_FIELDS and field_name in index_mapping:
                agg_key = f"{field_name}_cardinality"
                if agg_key in aggregations:
                    distinct_counts[field_name] = aggregations[agg_key]["value"]
        return distinct_counts

    def count_unique_metadata_by_filter(self, filters: dict[str, Any], metadata_fields: list[str]) -> dict[str, int]:
        """
        Returns the number of unique values for each specified metadata field of the documents
        that match the provided filters.

        :param filters: The filters to apply to count documents.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param metadata_fields: List of field names to calculate unique values for.
            Field names can include or omit the "meta." prefix.
        :returns: A dictionary mapping each metadata field name to the count of its unique values among the filtered
                  documents.
        :raises ValueError: If any of the requested fields don't exist in the index mapping.
        """
        self._ensure_initialized()
        assert self._client is not None

        # use index mapping to get all fields
        mapping = self._client.indices.get_mapping(index=self._index)
        index_mapping = mapping[self._index]["mappings"]["properties"]

        # normalize field names
        normalized_metadata_fields = [self._normalize_metadata_field_name(field) for field in metadata_fields]
        # validate that all requested fields exist in the index mapping
        missing_fields = [f for f in normalized_metadata_fields if f not in index_mapping]
        if missing_fields:
            msg = f"Fields not found in index mapping: {missing_fields}"
            raise ValueError(msg)

        # build aggregations for specified metadata fields
        aggs = self._build_cardinality_aggregations(index_mapping, normalized_metadata_fields)
        if not aggs:
            return {}

        # build and execute search query
        body = self._build_distinct_values_query_body(filters, aggs)
        result = self._client.search(index=self._index, body=body)

        # extract cardinality values from aggregations
        return self._extract_distinct_counts_from_aggregations(
            result.get("aggregations", {}), index_mapping, normalized_metadata_fields
        )

    async def count_unique_metadata_by_filter_async(
        self, filters: dict[str, Any], metadata_fields: list[str]
    ) -> dict[str, int]:
        """
        Asynchronously returns the number of unique values for each specified metadata field of the documents
        that match the provided filters.

        :param filters: The filters to apply to count documents.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param metadata_fields: List of field names to calculate unique values for.
            Field names can include or omit the "meta." prefix.
        :returns: A dictionary mapping each metadata field name to the count of its unique values among the filtered
                  documents.
        :raises ValueError: If any of the requested fields don't exist in the index mapping.
        """
        await self._ensure_initialized_async()
        assert self._async_client is not None

        # use index mapping to get all fields
        mapping = await self._async_client.indices.get_mapping(index=self._index)
        index_mapping = mapping[self._index]["mappings"]["properties"]

        # normalize field names
        normalized_metadata_fields = [self._normalize_metadata_field_name(field) for field in metadata_fields]
        # validate that all requested fields exist in the index mapping
        missing_fields = [f for f in normalized_metadata_fields if f not in index_mapping]
        if missing_fields:
            msg = f"Fields not found in index mapping: {missing_fields}"
            raise ValueError(msg)

        # build aggregations for specified metadata fields
        aggs = self._build_cardinality_aggregations(index_mapping, normalized_metadata_fields)
        if not aggs:
            return {}

        # build and execute search query
        body = self._build_distinct_values_query_body(filters, aggs)
        result = await self._async_client.search(index=self._index, body=body)

        # extract cardinality values from aggregations
        return self._extract_distinct_counts_from_aggregations(
            result.get("aggregations", {}), index_mapping, normalized_metadata_fields
        )

    def get_metadata_fields_info(self) -> dict[str, dict[str, str]]:
        """
        Returns the information about the fields in the index.

        If we populated the index with documents like:

        ```python
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1})
            Document(content="Doc 2", meta={"category": "B", "status": "inactive"})
        ```

        This method would return:

        ```python
            {
                'content': {'type': 'text'},
                'category': {'type': 'keyword'},
                'status': {'type': 'keyword'},
                'priority': {'type': 'long'},
            }
        ```

        :returns: The information about the fields in the index.
        """
        self._ensure_initialized()
        assert self._client is not None

        mapping = self._client.indices.get_mapping(index=self._index)
        index_mapping = mapping[self._index]["mappings"]["properties"]
        # remove all fields that are not metadata fields
        index_mapping = {k: v for k, v in index_mapping.items() if k not in SPECIAL_FIELDS}
        return index_mapping

    async def get_metadata_fields_info_async(self) -> dict[str, dict[str, str]]:
        """
        Asynchronously returns the information about the fields in the index.

        If we populated the index with documents like:

        ```python
            Document(content="Doc 1", meta={"category": "A", "status": "active", "priority": 1})
            Document(content="Doc 2", meta={"category": "B", "status": "inactive"})
        ```

        This method would return:

        ```python
            {
                'content': {'type': 'text'},
                'category': {'type': 'keyword'},
                'status': {'type': 'keyword'},
                'priority': {'type': 'long'},
            }
        ```

        :returns: The information about the fields in the index.
        """
        await self._ensure_initialized_async()
        assert self._async_client is not None

        mapping = await self._async_client.indices.get_mapping(index=self._index)
        index_mapping = mapping[self._index]["mappings"]["properties"]
        # remove all fields that are not metadata fields
        index_mapping = {k: v for k, v in index_mapping.items() if k not in SPECIAL_FIELDS}
        return index_mapping

    @staticmethod
    def _normalize_metadata_field_name(metadata_field: str) -> str:
        """
        Normalizes a metadata field name by removing the "meta." prefix if present.
        """
        return metadata_field[5:] if metadata_field.startswith("meta.") else metadata_field

    @staticmethod
    def _build_min_max_query_body(field_name: str) -> dict[str, Any]:
        """
        Builds the query body for getting min and max values using stats aggregation.
        """
        return {
            "query": {"match_all": {}},
            "aggs": {
                "field_stats": {
                    "stats": {
                        "field": field_name,
                    }
                }
            },
            "size": 0,  # We only need aggregations, not documents
        }

    @staticmethod
    def _extract_min_max_from_stats(stats: dict[str, Any]) -> dict[str, int | None]:
        """
        Extracts min and max values from stats aggregation results.
        """
        min_value = stats.get("min")
        max_value = stats.get("max")
        return {"min": min_value, "max": max_value}

    def get_metadata_field_min_max(self, metadata_field: str) -> dict[str, int | None]:
        """
        Returns the minimum and maximum values for the given metadata field.

        :param metadata_field: The metadata field to get the minimum and maximum values for.
        :returns: A dictionary with the keys "min" and "max", where each value is the minimum or maximum value of the
                  metadata field across all documents.
        """
        self._ensure_initialized()
        assert self._client is not None

        field_name = self._normalize_metadata_field_name(metadata_field)
        body = self._build_min_max_query_body(field_name)
        result = self._client.search(index=self._index, body=body)
        stats = result.get("aggregations", {}).get("field_stats", {})

        return self._extract_min_max_from_stats(stats)

    async def get_metadata_field_min_max_async(self, metadata_field: str) -> dict[str, int | None]:
        """
        Asynchronously returns the minimum and maximum values for the given metadata field.

        :param metadata_field: The metadata field to get the minimum and maximum values for.
        :returns: A dictionary with the keys "min" and "max", where each value is the minimum or maximum value of the
                  metadata field across all documents.
        """
        await self._ensure_initialized_async()
        assert self._async_client is not None

        field_name = self._normalize_metadata_field_name(metadata_field)
        body = self._build_min_max_query_body(field_name)
        result = await self._async_client.search(index=self._index, body=body)
        stats = result.get("aggregations", {}).get("field_stats", {})

        return self._extract_min_max_from_stats(stats)

    def get_metadata_field_unique_values(
        self,
        metadata_field: str,
        search_term: str | None = None,
        size: int | None = 10000,
        after: dict[str, Any] | None = None,
    ) -> tuple[list[str], dict[str, Any] | None]:
        """
        Returns unique values for a metadata field, optionally filtered by a search term in the content.
        Uses composite aggregations for proper pagination beyond 10k results.

        :param metadata_field: The metadata field to get unique values for.
        :param search_term: Optional search term to filter documents by matching in the content field.
        :param size: The number of unique values to return per page. Defaults to 10000.
        :param after: Optional pagination key from the previous response. Use None for the first page.
            For subsequent pages, pass the `after_key` from the previous response.
        :returns: A tuple containing (list of unique values, after_key for pagination).
            The after_key is None when there are no more results. Use it in the `after` parameter
            for the next page.
        """
        self._ensure_initialized()
        assert self._client is not None

        field_name = self._normalize_metadata_field_name(metadata_field)

        # filter by search_term if provided
        query: dict[str, Any] = {"match_all": {}}
        if search_term:
            # Use match_phrase for exact phrase matching to avoid tokenization issues
            query = {"match_phrase": {"content": search_term}}

        # Build composite aggregation for proper pagination
        composite_agg: dict[str, Any] = {
            "size": size,
            "sources": [{field_name: {"terms": {"field": field_name}}}],
        }
        if after is not None:
            composite_agg["after"] = after

        body = {
            "query": query,
            "aggs": {
                "unique_values": {
                    "composite": composite_agg,
                }
            },
            "size": 0,  # we only need aggregations, not documents
        }

        result = self._client.search(index=self._index, body=body)
        aggregations = result.get("aggregations", {})

        # Extract unique values from composite aggregation buckets
        unique_values_agg = aggregations.get("unique_values", {})
        unique_values_buckets = unique_values_agg.get("buckets", [])
        unique_values = [str(bucket["key"][field_name]) for bucket in unique_values_buckets]

        # Extract after_key for pagination
        # If we got fewer results than requested, we've reached the end
        after_key = unique_values_agg.get("after_key")
        if after_key is not None and size is not None and len(unique_values_buckets) < size:
            after_key = None

        return unique_values, after_key

    async def get_metadata_field_unique_values_async(
        self,
        metadata_field: str,
        search_term: str | None = None,
        size: int | None = 10000,
        after: dict[str, Any] | None = None,
    ) -> tuple[list[str], dict[str, Any] | None]:
        """
        Asynchronously returns unique values for a metadata field, optionally filtered by a search term in the content.
        Uses composite aggregations for proper pagination beyond 10k results.

        :param metadata_field: The metadata field to get unique values for.
        :param search_term: Optional search term to filter documents by matching in the content field.
        :param size: The number of unique values to return per page. Defaults to 10000.
        :param after: Optional pagination key from the previous response. Use None for the first page.
            For subsequent pages, pass the `after_key` from the previous response.
        :returns: A tuple containing (list of unique values, after_key for pagination).
            The after_key is None when there are no more results. Use it in the `after` parameter
            for the next page.
        """
        await self._ensure_initialized_async()
        assert self._async_client is not None

        field_name = self._normalize_metadata_field_name(metadata_field)

        # filter by search_term if provided
        query: dict[str, Any] = {"match_all": {}}
        if search_term:
            # Use match_phrase for exact phrase matching to avoid tokenization issues
            query = {"match_phrase": {"content": search_term}}

        # Build composite aggregation for proper pagination
        composite_agg: dict[str, Any] = {
            "size": size,
            "sources": [{field_name: {"terms": {"field": field_name}}}],
        }
        if after is not None:
            composite_agg["after"] = after

        body = {
            "query": query,
            "aggs": {
                "unique_values": {
                    "composite": composite_agg,
                }
            },
            "size": 0,  # we only need aggregations, not documents
        }

        result = await self._async_client.search(index=self._index, body=body)
        aggregations = result.get("aggregations", {})

        # Extract unique values from composite aggregation buckets
        unique_values_agg = aggregations.get("unique_values", {})
        unique_values_buckets = unique_values_agg.get("buckets", [])
        unique_values = [str(bucket["key"][field_name]) for bucket in unique_values_buckets]

        # Extract after_key for pagination
        # If we got fewer results than requested, we've reached the end
        after_key = unique_values_agg.get("after_key")
        if after_key is not None and size is not None and len(unique_values_buckets) < size:
            after_key = None

        return unique_values, after_key

    def _query_sql(self, query: str, fetch_size: int | None = None) -> dict[str, Any]:
        """
        Execute a raw OpenSearch SQL query against the index.

        This method is not meant to be part of the public interface of
        `OpenSearchDocumentStore` nor called directly.
        `OpenSearchSQLRetriever` uses this method directly and is the public interface for it.

        See `OpenSearchSQLRetriever` for more information.

        :param query: The OpenSearch SQL query to execute
        :param fetch_size: Optional number of results to fetch per page.
        :returns: The raw JSON response from OpenSearch SQL API (OpenSearch DSL format).
        """
        self._ensure_initialized()
        assert self._client is not None

        try:
            body: dict[str, Any] = {"query": query}
            if fetch_size is not None:
                body["fetch_size"] = fetch_size

            response_data = self._client.transport.perform_request(
                method="POST",
                url="/_plugins/_sql",
                body=body,
            )

            return response_data
        except Exception as e:
            msg = f"Failed to execute SQL query in OpenSearch: {e!s}"
            raise DocumentStoreError(msg) from e

    async def _query_sql_async(self, query: str, fetch_size: int | None = None) -> dict[str, Any]:
        """
        Asynchronously execute a raw OpenSearch SQL query against the index.

        This method is not meant to be part of the public interface of
        `OpenSearchDocumentStore` nor called directly.
        `OpenSearchSQLRetriever` uses this method directly and is the public interface for it.

        See `OpenSearchSQLRetriever` for more information.

        :param query: The OpenSearch SQL query to execute
        :param fetch_size: Optional number of results to fetch per page.
        :returns: The raw JSON response from OpenSearch SQL API (OpenSearch DSL format).
        """
        await self._ensure_initialized_async()
        assert self._async_client is not None

        try:
            body: dict[str, Any] = {"query": query}
            if fetch_size is not None:
                body["fetch_size"] = fetch_size

            response_data = await self._async_client.transport.perform_request(
                method="POST",
                url="/_plugins/_sql",
                body=body,
            )

            return response_data
        except Exception as e:
            msg = f"Failed to execute SQL query in OpenSearch: {e!s}"
            raise DocumentStoreError(msg) from e
