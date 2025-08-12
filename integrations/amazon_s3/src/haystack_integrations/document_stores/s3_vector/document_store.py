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
from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

logger = logging.getLogger(__name__)

Hosts = Union[str, List[Union[str, Mapping[str, Union[str, int]]]]]

class S3VectorDocumentStore:
    """
    A document store that uses Amazon S3 Vector Search for storing and retrieving documents with embeddings.

    This document store leverages AWS S3's vector search capabilities to store documents and their embeddings,
    enabling efficient similarity search and retrieval operations.

    Usage example:
    ```python
    from haystack_integrations.document_stores.s3_vector import S3VectorDocumentStore
    from haystack import Document

    document_store = S3VectorDocumentStore(
        bucket="my-vector-bucket",
        index="my-index",
        embedding_dim=1024
    )

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
        bucket: str = 'default',
        embedding_dim: int = 1024,
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
        Creates a new S3VectorDocumentStore instance.

        The S3VectorDocumentStore uses Amazon S3 Vector Search to store and retrieve document embeddings.
        The index will be created if it doesn't exist and create_index is True.

        For more information on AWS configuration, see the [AWS boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration)

        :param index: Name of the vector index in S3, if it doesn't exist it will be created. Defaults to "default"
        :param bucket: Name of the S3 bucket for vector storage. Defaults to "default"
        :param embedding_dim: Dimension of the embeddings. Defaults to 1024
        :param aws_access_key_id: AWS access key ID. If not provided, will use environment variable AWS_ACCESS_KEY_ID
        :param aws_secret_access_key: AWS secret access key. If not provided, will use environment variable AWS_SECRET_ACCESS_KEY
        :param aws_session_token: AWS session token for temporary credentials. If not provided, will use environment variable AWS_SESSION_TOKEN
        :param aws_region_name: AWS region name. If not provided, will use environment variable AWS_DEFAULT_REGION
        :param aws_profile_name: AWS profile name for credential resolution. If not provided, will use environment variable AWS_PROFILE
        :param create_index: Whether to create the index if it doesn't exist. Defaults to True
        :param **kwargs: Additional keyword arguments passed to the underlying AWS session
        """
        self._bucket = bucket
        self._index = index
        self._embedding_dim = embedding_dim
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
            self._s3_vector_client = session.client("s3vectors")
        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

    def create_index(
        self,
        bucket: str = 'default',
        index: str = 'default',
        dimension: int = 1024,
        distance_metric: str = "cosine",
        data_type: Optional[str] = None,
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
        response = client.create_index(
            vectorBucketName=bucket,
            indexName=index,
            # TODO handle optional data type
            dataType='float32',
            dimension=dimension,
            distanceMetric=distance_metric,
            # metadataConfiguration={
            #     'nonFilterableMetadataKeys': [
            #         'string',
            #     ]
            # }
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            index=self._index,
            bucket=self._bucket,
            embedding_dim=self._embedding_dim,
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
        assert self._s3_vector_client is not None

        return self._s3_vector_client.count(index=self._index)["count"]

    def _deserialize_search_hits(self, results: List[Dict[str, Any]]) -> List[Document]:
        out = []
        for hit in hits:
            data = hit["_source"]
            if "highlight" in hit:
                data["metadata"]["highlighted"] = hit["highlight"]
            data["score"] = hit["_score"]
            out.append(Document.from_dict(data))

        return out

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

        vectors = [
            {
                'key': document.content,
                'data': {
                    'float32': document.embedding
                },
                'metadata': document.meta
            }
        for document in documents]
        response = self._s3_vector_client.put_vectors(
            vectorBucketName=self._bucket,
            indexName=self._index,
            vectors=vectors
        )
        return len(vectors)

    def _deserialize_document(self, result: Dict[str, Any]) -> Document:
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
        self._s3_vector_client.delete_vectors(
            vectorBucketName='string',
            indexName='string',
            indexArn='string',
            keys=[
                'string',
            ]
        )

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
        response = self._s3_vector_client.query_vectors(
            vectorBucketName=self._bucket,
            indexName=self._index,
            topK=top_k,
            queryVector={
                'float32': query_embedding
            },
            # filter={...}|[...]|123|123.4|'string'|True|None,
            # returnMetadata=True|False,
            # returnDistance=True|False
        )
        documents = [Document(content=document['key']) for document in response['vectors']]
        return {'documents': documents}