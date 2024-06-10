# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import io
import logging
from copy import copy
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.filters import convert

from pinecone import Pinecone, PodSpec, ServerlessSpec

from .filters import _normalize_filters

logger = logging.getLogger(__name__)

# Pinecone has a limit of 1000 documents that can be returned in a query
# with include_metadata=True or include_data=True
# https://docs.pinecone.io/docs/limits
TOP_K_LIMIT = 1_000


DEFAULT_STARTER_PLAN_SPEC = {"serverless": {"region": "us-east-1", "cloud": "aws"}}


class PineconeDocumentStore:
    """
    A Document Store using [Pinecone vector database](https://www.pinecone.io/).
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("PINECONE_API_KEY"),  # noqa: B008
        index: str = "default",
        namespace: str = "default",
        batch_size: int = 100,
        dimension: int = 768,
        spec: Optional[Dict[str, Any]] = None,
        metric: Literal["cosine", "euclidean", "dotproduct"] = "cosine",
    ):
        """
        Creates a new PineconeDocumentStore instance.
        It is meant to be connected to a Pinecone index and namespace.

        :param api_key: The Pinecone API key.
        :param index: The Pinecone index to connect to. If the index does not exist, it will be created.
        :param namespace: The Pinecone namespace to connect to. If the namespace does not exist, it will be created
            at the first write.
        :param batch_size: The number of documents to write in a single batch. When setting this parameter,
            consider [documented Pinecone limits](https://docs.pinecone.io/reference/quotas-and-limits).
        :param dimension: The dimension of the embeddings. This parameter is only used when creating a new index.
        :param spec: The Pinecone spec to use when creating a new index. Allows choosing between serverless and pod
            deployment options and setting additional parameters. Refer to the
            [Pinecone documentation](https://docs.pinecone.io/reference/api/control-plane/create_index) for more
            details.
            If not provided, a default spec with serverless deployment in the `us-east-1` region will be used
            (compatible with the free tier).
        :param metric: The metric to use for similarity search. This parameter is only used when creating a new index.

        """
        self.api_key = api_key
        spec = spec or DEFAULT_STARTER_PLAN_SPEC
        self.namespace = namespace
        self.batch_size = batch_size
        self.metric = metric
        self.spec = spec
        self.dimension = dimension
        self.index_name = index

        self._index = None
        self._dummy_vector = [-10.0] * self.dimension

    @property
    def index(self):
        if self._index is not None:
            return self._index

        client = Pinecone(api_key=self.api_key.resolve_value(), source_tag="haystack")

        if self.index_name not in client.list_indexes().names():
            logger.info(f"Index {self.index_name} does not exist. Creating a new index.")
            pinecone_spec = self._convert_dict_spec_to_pinecone_object(self.spec)
            client.create_index(name=self.index_name, dimension=self.dimension, spec=pinecone_spec, metric=self.metric)
        else:
            logger.info(
                f"Connecting to existing index {self.index_name}. `dimension`, `spec`, and `metric` will be ignored."
            )

        self._index = client.Index(name=self.index_name)

        actual_dimension = self._index.describe_index_stats().get("dimension")
        if actual_dimension and actual_dimension != self.dimension:
            logger.warning(
                f"Dimension of index {self.index_name} is {actual_dimension}, but {self.dimension} was specified. "
                "The specified dimension will be ignored."
                "If you need an index with a different dimension, please create a new one."
            )
        self.dimension = actual_dimension or self.dimension
        self._dummy_vector = [-10.0] * self.dimension

        return self._index

    @staticmethod
    def _convert_dict_spec_to_pinecone_object(spec: Dict[str, Any]):
        """Convert the spec dictionary to a Pinecone spec object"""

        if "serverless" in spec:
            serverless_spec = spec["serverless"]
            return ServerlessSpec(**serverless_spec)
        if "pod" in spec:
            pod_spec = spec["pod"]
            return PodSpec(**pod_spec)

        msg = (
            "Invalid spec. Must contain either `serverless` or `pod` key. "
            "Refer to https://docs.pinecone.io/reference/api/control-plane/create_index for more details."
        )
        raise ValueError(msg)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PineconeDocumentStore":
        """
        Deserializes the component from a dictionary.
        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.
        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            spec=self.spec,
            index=self.index_name,
            dimension=self.dimension,
            namespace=self.namespace,
            batch_size=self.batch_size,
            metric=self.metric,
        )

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        try:
            count = self.index.describe_index_stats()["namespaces"][self.namespace]["vector_count"]
        except KeyError:
            count = 0
        return count

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes Documents to Pinecone.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
            PineconeDocumentStore only supports `DuplicatePolicy.OVERWRITE`.

        :returns: The number of documents written to the document store.
        """
        if len(documents) > 0 and not isinstance(documents[0], Document):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

        if policy not in [DuplicatePolicy.NONE, DuplicatePolicy.OVERWRITE]:
            logger.warning(
                f"PineconeDocumentStore only supports `DuplicatePolicy.OVERWRITE`"
                f"but got {policy}. Overwriting duplicates is enabled by default."
            )

        documents_for_pinecone = self._convert_documents_to_pinecone_format(documents)

        result = self.index.upsert(vectors=documents_for_pinecone, namespace=self.namespace, batch_size=self.batch_size)

        written_docs = result["upserted_count"]
        return written_docs

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :returns: A list of Documents that match the given filters.
        """

        # Pinecone only performs vector similarity search
        # here we are querying with a dummy vector and the max compatible top_k
        documents = self._embedding_retrieval(query_embedding=self._dummy_vector, filters=filters, top_k=TOP_K_LIMIT)

        # when simply filtering, we don't want to return any scores
        # furthermore, we are querying with a dummy vector, so the scores are meaningless
        for doc in documents:
            doc.score = None

        if len(documents) == TOP_K_LIMIT:
            logger.warning(
                f"PineconeDocumentStore can return at most {TOP_K_LIMIT} documents and the query has hit this limit. "
                f"It is likely that there are more matching documents in the document store. "
            )
        return documents

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """
        self.index.delete(ids=document_ids, namespace=self.namespace)

    def _embedding_retrieval(
        self,
        query_embedding: List[float],
        *,
        namespace: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.

        This method is not mean to be part of the public interface of
        `PineconeDocumentStore` nor called directly.
        `PineconeEmbeddingRetriever` uses this method directly and is the public interface for it.

        :param query_embedding: Embedding of the query.
        :param namespace: Pinecone namespace to query. Defaults the namespace of the document store.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.

        :returns: List of Document that are most similar to `query_embedding`
        """

        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        if filters and "operator" not in filters and "conditions" not in filters:
            filters = convert(filters)
        filters = _normalize_filters(filters) if filters else None

        result = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace or self.namespace,
            filter=filters,
            include_values=True,
            include_metadata=True,
        )

        return self._convert_query_result_to_documents(result)

    def _convert_query_result_to_documents(self, query_result: Dict[str, Any]) -> List[Document]:
        pinecone_docs = query_result["matches"]
        documents = []
        for pinecone_doc in pinecone_docs:
            content = pinecone_doc["metadata"].pop("content", None)

            dataframe = None
            dataframe_string = pinecone_doc["metadata"].pop("dataframe", None)
            if dataframe_string:
                dataframe = pd.read_json(io.StringIO(dataframe_string))

            # we always store vectors during writing
            # but we don't want to return them if they are dummy vectors
            embedding = None
            if pinecone_doc["values"] != self._dummy_vector:
                embedding = pinecone_doc["values"]

            doc = Document(
                id=pinecone_doc["id"],
                content=content,
                dataframe=dataframe,
                meta=pinecone_doc["metadata"],
                embedding=embedding,
                score=pinecone_doc["score"],
            )
            documents.append(doc)

        return documents

    def _convert_documents_to_pinecone_format(self, documents: List[Document]) -> List[Dict[str, Any]]:
        documents_for_pinecone = []
        for document in documents:
            embedding = copy(document.embedding)
            if embedding is None:
                logger.warning(
                    f"Document {document.id} has no embedding. Pinecone is a purely vector database. "
                    "A dummy embedding will be used, but this can affect the search results. "
                )
                embedding = self._dummy_vector
            doc_for_pinecone = {"id": document.id, "values": embedding, "metadata": dict(document.meta)}

            # we save content/dataframe as metadata
            if document.content is not None:
                doc_for_pinecone["metadata"]["content"] = document.content
            if document.dataframe is not None:
                doc_for_pinecone["metadata"]["dataframe"] = document.dataframe.to_json()
            # currently, storing blob in Pinecone is not supported
            if document.blob is not None:
                logger.warning(
                    f"Document {document.id} has the `blob` field set, but storing `ByteStream` "
                    "objects in Pinecone is not supported. "
                    "The content of the `blob` field will be ignored."
                )
            if hasattr(document, "sparse_embedding") and document.sparse_embedding is not None:
                logger.warning(
                    "Document %s has the `sparse_embedding` field set,"
                    "but storing sparse embeddings in Pinecone is not currently supported."
                    "The `sparse_embedding` field will be ignored.",
                    document.id,
                )

            documents_for_pinecone.append(doc_for_pinecone)
        return documents_for_pinecone
