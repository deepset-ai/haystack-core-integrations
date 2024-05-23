# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.filters import convert
from haystack_integrations.document_stores.opensearch.filters import normalize_filters
from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

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


class OpenSearchDocumentStore:
    def __init__(
        self,
        *,
        hosts: Optional[Hosts] = None,
        index: str = "default",
        **kwargs,
    ):
        """
        Creates a new OpenSearchDocumentStore instance.

        For more information on connection parameters, see the [official OpenSearch documentation](https://opensearch.org/docs/latest/clients/python-low-level/#connecting-to-opensearch)

        For the full list of supported kwargs, see the [official OpenSearch reference](https://opensearch-project.github.io/opensearch-py/api-ref/clients/opensearch_client.html)

        :param hosts: List of hosts running the OpenSearch client. Defaults to None
        :param index: Name of index in OpenSearch, if it doesn't exist it will be created. Defaults to "default"
        :param **kwargs: Optional arguments that ``OpenSearch`` takes.
        """
        self._client = None
        self._hosts = hosts
        self._index = index
        self._kwargs = kwargs

    @property
    def client(self) -> OpenSearch:
        if not self._client:
            self._client = OpenSearch(self._hosts, **self._kwargs)
            # Check client connection, this will raise if not connected
            self._client.info()  # type:ignore

            # Create the index if it doesn't exist
            if not self._client.indices.exists(index=self._index):  # type:ignore
                # configure fallback mapping for the embedding field
                method = self._kwargs.get("method", None)
                embedding_dim = self._kwargs.get("embedding_dim", 768)
                default_mappings: Dict[str, Any] = {
                    "properties": {
                        "embedding": {"type": "knn_vector", "index": True, "dimension": embedding_dim},
                        "content": {"type": "text"},
                    },
                    "dynamic_templates": [
                        {
                            "strings": {
                                "match_mapping_type": "string",
                                "mapping": {
                                    "type": "keyword",
                                },
                            }
                        }
                    ],
                }
                if method:
                    default_mappings["properties"]["embedding"]["method"] = method

                body = {
                    "mappings": self._kwargs.get("mappings", default_mappings),
                    "settings": self._kwargs.get("settings", {"index.knn": True}),
                }
                self._client.indices.create(index=self._index, body=body)  # type:ignore

        return self._client

    def to_dict(self) -> Dict[str, Any]:
        # This is not the best solution to serialise this class but is the fastest to implement.
        # Not all kwargs types can be serialised to text so this can fail. We must serialise each
        # type explicitly to handle this properly.
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            hosts=self._hosts,
            index=self._index,
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
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        return self.client.count(index=self._index)["count"]

    def _search_documents(self, **kwargs) -> List[Document]:
        """
        Calls the OpenSearch client's search method and handles pagination.
        """
        res = self.client.search(
            index=self._index,
            body=kwargs,
        )
        documents: List[Document] = [self._deserialize_document(hit) for hit in res["hits"]["hits"]]
        return documents

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        if filters and "operator" not in filters and "conditions" not in filters:
            filters = convert(filters)

        if filters:
            query = {"bool": {"filter": normalize_filters(filters)}}
            documents = self._search_documents(query=query, size=10_000)
        else:
            documents = self._search_documents(size=10_000)

        return documents

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes Documents to OpenSearch.
        If policy is not specified or set to DuplicatePolicy.NONE, it will raise an exception if a document with the
        same ID already exists in the document store.
        """
        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        action = "index" if policy == DuplicatePolicy.OVERWRITE else "create"
        documents_written, errors = bulk(
            client=self.client,
            actions=(
                {
                    "_op_type": action,
                    "_id": doc.id,
                    "_source": doc.to_dict(),
                }
                for doc in documents
            ),
            refresh="wait_for",
            index=self._index,
            raise_on_error=False,
        )

        if errors:
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
        Deletes all documents with a matching document_ids from the document store.

        :param object_ids: the object_ids to delete
        """

        bulk(
            client=self.client,
            actions=({"_op_type": "delete", "_id": id_} for id_ in document_ids),
            refresh="wait_for",
            index=self._index,
            raise_on_error=False,
        )

    def _bm25_retrieval(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        fuzziness: str = "AUTO",
        top_k: int = 10,
        scale_score: bool = False,
        all_terms_must_match: bool = False,
    ) -> List[Document]:
        """
        OpenSearch by defaults uses BM25 search algorithm.
        Even though this method is called `bm25_retrieval` it searches for `query`
        using the search algorithm `_client` was configured with.

        This method is not meant to be part of the public interface of
        `OpenSearchDocumentStore` nor called directly.
        `OpenSearchBM25Retriever` uses this method directly and is the public interface for it.

        `query` must be a non empty string, otherwise a `ValueError` will be raised.

        :param query: String to search in saved Documents' text.
        :param filters: Optional filters to narrow down the search space.
        :param fuzziness: Fuzziness parameter passed to OpenSearch, defaults to "AUTO". see the official documentation
                          for valid [fuzziness values](https://www.elastic.co/guide/en/OpenSearch/reference/current/common-options.html#fuzziness)
        :param top_k: Maximum number of Documents to return, defaults to 10
        :param scale_score: If `True` scales the Document`s scores between 0 and 1, defaults to False
        :param all_terms_must_match: If `True` all terms in `query` must be present in the Document, defaults to False
        :raises ValueError: If `query` is an empty string
        :returns: List of Document that match `query`
        """

        if not query:
            msg = "query must be a non empty string"
            raise ValueError(msg)

        operator = "AND" if all_terms_must_match else "OR"
        body: Dict[str, Any] = {
            "size": top_k,
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

        documents = self._search_documents(**body)

        if scale_score:
            for doc in documents:
                doc.score = float(1 / (1 + np.exp(-np.asarray(doc.score / BM25_SCALING_FACTOR))))  # type:ignore

        return documents

    def _embedding_retrieval(
        self,
        query_embedding: List[float],
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.
        It uses the OpenSearch's Approximate k-Nearest Neighbors search algorithm.

        This method is not meant to be part of the public interface of
        `OpenSearchDocumentStore` nor called directly.
        `OpenSearchEmbeddingRetriever` uses this method directly and is the public interface for it.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents. Defaults to None.
            Filters are applied during the approximate kNN search to ensure that top_k matching documents are returned.
        :param top_k: Maximum number of Documents to return, defaults to 10
        :raises ValueError: If `query_embedding` is an empty list
        :returns: List of Document that are most similar to `query_embedding`
        """

        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        body: Dict[str, Any] = {
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
            "size": top_k,
        }

        if filters:
            body["query"]["bool"]["filter"] = normalize_filters(filters)

        docs = self._search_documents(**body)
        return docs
