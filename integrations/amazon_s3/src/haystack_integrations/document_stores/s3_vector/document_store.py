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

DEFAULT_SETTINGS = {"index.knn": True}
DEFAULT_MAX_CHUNK_BYTES = 100 * 1024 * 1024


class S3VectorDocumentStore:
    """
    An instance of an S3 Vector database you can use to store all types of data.

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
        index: str = "default",
        vector_bucket_name: str = 'default',
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
        embedding_dim: int = 768,
        return_embedding: bool = False,
        aws_access_key_id: Optional[Secret] = Secret.from_env_var(["AWS_ACCESS_KEY_ID"], strict=False),  # noqa: B008
        aws_secret_access_key: Optional[Secret] = Secret.from_env_var(  # noqa: B008
            ["AWS_SECRET_ACCESS_KEY"], strict=False
        ),
        aws_session_token: Optional[Secret] = Secret.from_env_var(["AWS_SESSION_TOKEN"], strict=False),  # noqa: B008
        aws_region_name: Optional[Secret] = Secret.from_env_var(["AWS_DEFAULT_REGION"], strict=False),  # noqa: B008
        aws_profile_name: Optional[Secret] = Secret.from_env_var(["AWS_PROFILE"], strict=False),  # noqa: B008
        create_index: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Creates a new OpenSearchDocumentStore instance.

        The ``embeddings_dim``, ``method``, ``mappings``, and ``settings`` arguments are only used if the index does not
        exists and needs to be created. If the index already exists, its current configurations will be used.

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
            for more information. Defaults to {"index.knn": True}
        :param create_index: Whether to create the index if it doesn't exist. Defaults to True
        :param http_auth: http_auth param passed to the underying connection class.
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
        self._vector_bucket_name = vector_bucket_name
        self._max_chunk_bytes = max_chunk_bytes
        self._embedding_dim = embedding_dim
        self._return_embedding = return_embedding
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token
        self._aws_region_name = aws_region_name
        self._aws_profile_name = aws_profile_name
        self._create_index = create_index
        self._kwargs = kwargs

        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None

        try:
            session = get_aws_session(
                aws_access_key_id=resolve_secret(aws_access_key_id),
                aws_secret_access_key=resolve_secret(aws_secret_access_key),
                aws_session_token=resolve_secret(aws_session_token),
                aws_region_name=resolve_secret(aws_region_name),
                aws_profile_name=resolve_secret(aws_profile_name),
            )
            self._bedrock_client = session.client("s3vectors")
        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception


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
        bucket: str = 'default',
        index: str = 'default',
        dimension: int = 1024,
        distance_metric: str = "euclidean",
        data_type: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Creates an index in an S3 Vector Bucket.

        Note that this method ignores the `create_index` argument from the constructor.

        :param index: Name of the index to create. If None, the index name from the constructor is used.
        :param mappings: The mapping of how the documents are stored and indexed. Please see the [official OpenSearch docs](https://opensearch.org/docs/latest/field-types/)
            for more information. If None, the mappings from the constructor are used.
        :param settings: The settings of the index to be created. Please see the [official OpenSearch docs](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/#index-settings)
            for more information. If None, the settings from the constructor are used.
        """
        assert self._client is not None

        if not index:
            index = self._index
        if not bucket:
            bucket = self._bucket

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            index=self._index,
            vector_bucket_name=self._vector_bucket_name,
            max_chunk_bytes=self._max_chunk_bytes,
            embedding_dim=self._embedding_dim,
            return_embedding=self._return_embedding,
            aws_access_key_id=self._aws_access_key_id.to_dict() if self._aws_access_key_id else None,
            aws_secret_access_key=self._aws_secret_access_key.to_dict() if self._aws_secret_access_key else None,
            aws_session_token=self._aws_session_token.to_dict() if self._aws_session_token else None,
            aws_region_name=self._aws_region_name.to_dict() if self._aws_region_name else None,
            aws_profile_name=self._aws_profile_name.to_dict() if self._aws_profile_name else None,
            create_index=self._create_index,
            **self._kwargs
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmazonBedrockRanker":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_secrets_inplace(
            data["init_parameters"],
            ["aws_access_key_id", "aws_secret_access_key", "aws_session_token", "aws_region_name", "aws_profile_name"],
        )
        return default_from_dict(cls, data)

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

    def _deserialize_search_hits(self, hits: List[Dict[str, Any]]) -> List[Document]:
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
        return self._deserialize_search_hits(search_results["hits"]["hits"])

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
        self._process_bulk_write_errors(errors, policy)
        return documents_written

    def _deserialize_document(self, hit: Dict[str, Any]) -> Document:
        """
        Creates a Document from the search hit provided.
        This is mostly useful in self.filter_documents().
        """
        data = hit["_source"]

        if "highlight" in hit:
            data["metadata"]["highlighted"] = hit["highlight"]
        data["score"] = hit["_score"]

        return Document.from_dict(data)

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """

        self._ensure_initialized()

        bulk(**self._prepare_bulk_delete_request(document_ids=document_ids, is_async=False))

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
