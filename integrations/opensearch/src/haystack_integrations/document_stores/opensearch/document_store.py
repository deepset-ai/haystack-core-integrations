# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from math import exp
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret
from opensearchpy import AsyncHttpConnection, AsyncOpenSearch, OpenSearch
from opensearchpy.helpers import async_bulk, bulk

from haystack_integrations.document_stores.opensearch.auth import AsyncAWSAuth, AWSAuth
from haystack_integrations.document_stores.opensearch.filters import normalize_filters

logger = logging.getLogger(__name__)

Hosts = Union[str, List[Union[str, Mapping[str, Union[str, int]]]]]

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
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
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
        hosts: Optional[Hosts] = None,
        index: str = "default",
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
        embedding_dim: int = 768,
        return_embedding: bool = False,
        method: Optional[Dict[str, Any]] = None,
        mappings: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = DEFAULT_SETTINGS,
        create_index: bool = True,
        http_auth: Any = (
            Secret.from_env_var("OPENSEARCH_USERNAME", strict=False),  # noqa: B008
            Secret.from_env_var("OPENSEARCH_PASSWORD", strict=False),  # noqa: B008
        ),
        use_ssl: Optional[bool] = None,
        verify_certs: Optional[bool] = None,
        timeout: Optional[int] = None,
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
        self._client: Optional[OpenSearch] = None
        self._async_client: Optional[AsyncOpenSearch] = None
        self._initialized = False

    def _get_default_mappings(self) -> Dict[str, Any]:
        default_mappings: Dict[str, Any] = {
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
        index: Optional[str] = None,
        mappings: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        # Handle http_auth serialization
        http_auth: Union[List[Dict[str, Any]], Dict[str, Any], Tuple[str, str], List[str], str] = ""
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
    def from_dict(cls, data: Dict[str, Any]) -> "OpenSearchDocumentStore":
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
        if not self._initialized:
            self._client = OpenSearch(
                hosts=self._hosts,
                http_auth=self._http_auth,
                use_ssl=self._use_ssl,
                verify_certs=self._verify_certs,
                timeout=self._timeout,
                **self._kwargs,
            )
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

            self._ensure_index_exists()

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
        self._ensure_initialized()

        assert self._async_client is not None
        return (await self._async_client.count(index=self._index))["count"]

    @staticmethod
    def _deserialize_search_hits(hits: List[Dict[str, Any]]) -> List[Document]:
        out = []
        for hit in hits:
            data = hit["_source"]
            if "highlight" in hit:
                data["metadata"]["highlighted"] = hit["highlight"]
            data["score"] = hit["_score"]
            out.append(Document.from_dict(data))

        return out

    def _prepare_filter_search_request(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        search_kwargs: Dict[str, Any] = {"size": 10_000}
        if filters:
            search_kwargs["query"] = {"bool": {"filter": normalize_filters(filters)}}

        # For some applications not returning the embedding can save a lot of bandwidth
        # if you don't need this data not retrieving it can be a good idea
        if not self._return_embedding:
            search_kwargs["_source"] = {"excludes": ["embedding"]}
        return search_kwargs

    def _search_documents(self, request_body: Dict[str, Any]) -> List[Document]:
        assert self._client is not None
        search_results = self._client.search(index=self._index, body=request_body)
        return OpenSearchDocumentStore._deserialize_search_hits(search_results["hits"]["hits"])

    async def _search_documents_async(self, request_body: Dict[str, Any]) -> List[Document]:
        assert self._async_client is not None
        search_results = await self._async_client.search(index=self._index, body=request_body)
        return OpenSearchDocumentStore._deserialize_search_hits(search_results["hits"]["hits"])

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        self._ensure_initialized()
        return self._search_documents(self._prepare_filter_search_request(filters))

    async def filter_documents_async(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Asynchronously returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """
        self._ensure_initialized()
        return await self._search_documents_async(self._prepare_filter_search_request(filters))

    def _prepare_bulk_write_request(
        self, *, documents: List[Document], policy: DuplicatePolicy, is_async: bool
    ) -> Dict[str, Any]:
        if len(documents) > 0 and not isinstance(documents[0], Document):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        action = "index" if policy == DuplicatePolicy.OVERWRITE else "create"
        opensearch_actions = []
        for doc in documents:
            doc_dict = doc.to_dict()
            if "sparse_embedding" in doc_dict:
                sparse_embedding = doc_dict.pop("sparse_embedding", None)
                if sparse_embedding:
                    logger.warning(
                        "Document {id} has the `sparse_embedding` field set,"
                        "but storing sparse embeddings in OpenSearch is not currently supported."
                        "The `sparse_embedding` field will be ignored.",
                        id=doc.id,
                    )
            opensearch_actions.append(
                {
                    "_op_type": action,
                    "_id": doc.id,
                    "_source": doc_dict,
                }
            )

        return {
            "client": self._client if not is_async else self._async_client,
            "actions": opensearch_actions,
            "refresh": "wait_for",
            "index": self._index,
            "raise_on_error": False,
            "max_chunk_bytes": self._max_chunk_bytes,
            "stats_only": False,
        }

    @staticmethod
    def _process_bulk_write_errors(errors: List[Dict[str, Any]], policy: DuplicatePolicy) -> None:
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

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents to the document store.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises DuplicateDocumentError: If a document with the same id already exists in the document store
             and the policy is set to `DuplicatePolicy.FAIL` (or not specified).
        :returns: The number of documents written to the document store.
        """
        self._ensure_initialized()

        bulk_params = self._prepare_bulk_write_request(documents=documents, policy=policy, is_async=False)
        documents_written, errors = bulk(**bulk_params)
        OpenSearchDocumentStore._process_bulk_write_errors(errors, policy)
        return documents_written

    async def write_documents_async(
        self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE
    ) -> int:
        """
        Asynchronously writes documents to the document store.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :returns: The number of documents written to the document store.
        """
        self._ensure_initialized()
        bulk_params = self._prepare_bulk_write_request(documents=documents, policy=policy, is_async=True)
        documents_written, errors = await async_bulk(**bulk_params)
        # since we call async_bulk with stats_only=False, errors is guaranteed to be a list (not int)
        OpenSearchDocumentStore._process_bulk_write_errors(errors=errors, policy=policy)  # type: ignore[arg-type]
        return documents_written

    @staticmethod
    def _deserialize_document(hit: Dict[str, Any]) -> Document:
        """
        Creates a Document from the search hit provided.
        This is mostly useful in self.filter_documents().
        """
        data = hit["_source"]

        if "highlight" in hit:
            data["metadata"]["highlighted"] = hit["highlight"]
        data["score"] = hit["_score"]

        return Document.from_dict(data)

    def _prepare_bulk_delete_request(self, *, document_ids: List[str], is_async: bool) -> Dict[str, Any]:
        return {
            "client": self._client if not is_async else self._async_client,
            "actions": ({"_op_type": "delete", "_id": id_} for id_ in document_ids),
            "refresh": "wait_for",
            "index": self._index,
            "raise_on_error": False,
            "max_chunk_bytes": self._max_chunk_bytes,
        }

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """

        self._ensure_initialized()

        bulk(**self._prepare_bulk_delete_request(document_ids=document_ids, is_async=False))

    async def delete_documents_async(self, document_ids: List[str]) -> None:
        """
        Asynchronously deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """
        self._ensure_initialized()

        await async_bulk(**self._prepare_bulk_delete_request(document_ids=document_ids, is_async=True))

    def _prepare_delete_all_request(self, *, is_async: bool) -> Dict[str, Any]:
        return {
            "index": self._index,
            "body": {"query": {"match_all": {}}},  # Delete all documents
            "wait_for_completion": False if is_async else True,  # block until done (set False for async)
        }

    def delete_all_documents(self, recreate_index: bool = False) -> None:  # noqa: FBT002, FBT001
        """
        Deletes all documents in the document store.

        :param recreate_index: If True, the index will be deleted and recreated with the original mappings and
            settings. If False, all documents will be deleted using the `delete_by_query` API.
        """
        self._ensure_initialized()
        assert self._client is not None

        try:
            if recreate_index:
                # get the current index mappings and settings
                index_name = self._index
                body = {
                    "mappings": self._client.indices.get(self._index)[index_name]["mappings"],
                    "settings": self._client.indices.get(self._index)[index_name]["settings"],
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
                result = self._client.delete_by_query(**self._prepare_delete_all_request(is_async=False))
                logger.info(
                    "Deleted all the {n_docs} documents from the index '{index}'.",
                    index=self._index,
                    n_docs=result["deleted"],
                )
        except Exception as e:
            msg = f"Failed to delete all documents from OpenSearch: {e!s}"
            raise DocumentStoreError(msg) from e

    async def delete_all_documents_async(self, recreate_index: bool = False) -> None:  # noqa: FBT002, FBT001
        """
        Asynchronously deletes all documents in the document store.

        :param recreate_index: If True, the index will be deleted and recreated with the original mappings and
            settings. If False, all documents will be deleted using the `delete_by_query` API.
        """
        self._ensure_initialized()
        assert self._async_client is not None

        try:
            if recreate_index:
                # get the current index mappings and settings
                index_name = self._index
                index_info = await self._async_client.indices.get(self._index)
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
                await self._async_client.delete_by_query(**self._prepare_delete_all_request(is_async=True))

        except Exception as e:
            msg = f"Failed to delete all documents from OpenSearch: {e!s}"
            raise DocumentStoreError(msg) from e

    def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        """
        Deletes all documents that match the provided filters.

        :param filters: The filters to apply to select documents for deletion.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents deleted.
        """
        self._ensure_initialized()
        assert self._client is not None

        try:
            normalized_filters = normalize_filters(filters)
            body = {"query": {"bool": {"filter": normalized_filters}}}
            result = self._client.delete_by_query(index=self._index, body=body)
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

    async def delete_by_filter_async(self, filters: Dict[str, Any]) -> int:
        """
        Asynchronously deletes all documents that match the provided filters.

        :param filters: The filters to apply to select documents for deletion.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :returns: The number of documents deleted.
        """
        self._ensure_initialized()
        assert self._async_client is not None

        try:
            normalized_filters = normalize_filters(filters)
            body = {"query": {"bool": {"filter": normalized_filters}}}
            result = await self._async_client.delete_by_query(index=self._index, body=body)
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

    def update_by_filter(self, filters: Dict[str, Any], meta: Dict[str, Any]) -> int:
        """
        Updates the metadata of all documents that match the provided filters.

        :param filters: The filters to apply to select documents for updating.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param meta: The metadata fields to update.
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
            result = self._client.update_by_query(index=self._index, body=body)
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

    async def update_by_filter_async(self, filters: Dict[str, Any], meta: Dict[str, Any]) -> int:
        """
        Asynchronously updates the metadata of all documents that match the provided filters.

        :param filters: The filters to apply to select documents for updating.
            For filter syntax, see [Haystack metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
        :param meta: The metadata fields to update.
        :returns: The number of documents updated.
        """
        self._ensure_initialized()
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
            result = await self._async_client.update_by_query(index=self._index, body=body)
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
        filters: Optional[Dict[str, Any]],
        fuzziness: Union[int, str],
        top_k: int,
        all_terms_must_match: bool,
        custom_query: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not query:
            body: Dict[str, Any] = {"query": {"bool": {"must": {"match_all": {}}}}}
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
    def _postprocess_bm25_search_results(*, results: List[Document], scale_score: bool) -> None:
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
        filters: Optional[Dict[str, Any]] = None,
        fuzziness: Union[int, str] = "AUTO",
        top_k: int = 10,
        scale_score: bool = False,
        all_terms_must_match: bool = False,
        custom_query: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
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
        filters: Optional[Dict[str, Any]] = None,
        fuzziness: str = "AUTO",
        top_k: int = 10,
        scale_score: bool = False,
        all_terms_must_match: bool = False,
        custom_query: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
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

        self._ensure_initialized()

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

    def _prepare_embedding_search_request(
        self,
        *,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]],
        top_k: int,
        custom_query: Optional[Dict[str, Any]],
        efficient_filtering: bool = False,
    ) -> Dict[str, Any]:
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        body: Dict[str, Any]
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
        query_embedding: List[float],
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        custom_query: Optional[Dict[str, Any]] = None,
        efficient_filtering: bool = False,
    ) -> List[Document]:
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
        )
        return self._search_documents(search_params)

    async def _embedding_retrieval_async(
        self,
        query_embedding: List[float],
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        custom_query: Optional[Dict[str, Any]] = None,
        efficient_filtering: bool = False,
    ) -> List[Document]:
        """
        Asynchronously retrieves documents that are most similar to the query embedding using a vector similarity
        metric. It uses the OpenSearch's Approximate k-Nearest Neighbors search algorithm.

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
        )
        return await self._search_documents_async(search_params)

    def _render_custom_query(self, custom_query: Any, substitutions: Dict[str, Any]) -> Any:
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
