# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import io
import logging
from copy import copy
from typing import Any, Dict, List, Optional

import pandas as pd
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.filters import convert

import pinecone

from .filters import _normalize_filters

logger = logging.getLogger(__name__)

# Pinecone has a limit of 1000 documents that can be returned in a query
# with include_metadata=True or include_data=True
# https://docs.pinecone.io/docs/limits
TOP_K_LIMIT = 1_000


class PineconeDocumentStore:
    """
    A Document Store using [Pinecone vector database](https://www.pinecone.io/).
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("PINECONE_API_KEY"),  # noqa: B008
        environment: str = "us-west1-gcp",
        index: str = "default",
        namespace: str = "default",
        batch_size: int = 100,
        dimension: int = 768,
        **index_creation_kwargs,
    ):
        """
        Creates a new PineconeDocumentStore instance.
        It is meant to be connected to a Pinecone index and namespace.

        :param api_key: The Pinecone API key.
        :param environment: The Pinecone environment to connect to.
        :param index: The Pinecone index to connect to. If the index does not exist, it will be created.
        :param namespace: The Pinecone namespace to connect to. If the namespace does not exist, it will be created
            at the first write.
        :param batch_size: The number of documents to write in a single batch. When setting this parameter,
            consider [documented Pinecone limits](https://docs.pinecone.io/docs/limits).
        :param dimension: The dimension of the embeddings. This parameter is only used when creating a new index.
        :param index_creation_kwargs: Additional keyword arguments to pass to the index creation method.
            You can find the full list of supported arguments in the
            [API reference](https://docs.pinecone.io/reference/create_index).

        """
        self.api_key = api_key

        pinecone.init(api_key=api_key.resolve_value(), environment=environment)

        if index not in pinecone.list_indexes():
            logger.info(f"Index {index} does not exist. Creating a new index.")
            pinecone.create_index(name=index, dimension=dimension, **index_creation_kwargs)
        else:
            logger.info(f"Index {index} already exists. Connecting to it.")

        self._index = pinecone.Index(index_name=index)

        actual_dimension = self._index.describe_index_stats().get("dimension")
        if actual_dimension and actual_dimension != dimension:
            logger.warning(
                f"Dimension of index {index} is {actual_dimension}, but {dimension} was specified. "
                "The specified dimension will be ignored."
                "If you need an index with a different dimension, please create a new one."
            )
        self.dimension = actual_dimension or dimension

        self._dummy_vector = [-10.0] * self.dimension
        self.environment = environment
        self.index = index
        self.namespace = namespace
        self.batch_size = batch_size
        self.index_creation_kwargs = index_creation_kwargs

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
            environment=self.environment,
            index=self.index,
            dimension=self.dimension,
            namespace=self.namespace,
            batch_size=self.batch_size,
            **self.index_creation_kwargs,
        )

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        try:
            count = self._index.describe_index_stats()["namespaces"][self.namespace]["vector_count"]
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

        result = self._index.upsert(
            vectors=documents_for_pinecone, namespace=self.namespace, batch_size=self.batch_size
        )

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
        self._index.delete(ids=document_ids, namespace=self.namespace)

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

        result = self._index.query(
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
