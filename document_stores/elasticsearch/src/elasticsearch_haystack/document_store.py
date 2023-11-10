# SPDX-FileCopyrightText: 2023-present Silvano Cerza <silvanocerza@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np

# There are no import stubs for elastic_transport and elasticsearch so mypy fails
from elastic_transport import NodeConfig  # type: ignore[import-not-found]
from elasticsearch import Elasticsearch, helpers  # type: ignore[import-not-found]
from haystack.preview import default_from_dict, default_to_dict
from haystack.preview.dataclasses import Document
from haystack.preview.document_stores.decorator import document_store
from haystack.preview.document_stores.errors import DuplicateDocumentError
from haystack.preview.document_stores.protocols import DuplicatePolicy

from elasticsearch_haystack.filters import _normalize_filters

logger = logging.getLogger(__name__)

Hosts = Union[str, List[Union[str, Mapping[str, Union[str, int]], NodeConfig]]]

# document scores are essentially unbounded and will be scaled to values between 0 and 1 if scale_score is set to
# True (default). Scaling uses the expit function (inverse of the logit function) after applying a scaling factor
# (e.g., BM25_SCALING_FACTOR for the bm25_retrieval method).
# Larger scaling factor decreases scaled scores. For example, an input of 10 is scaled to 0.99 with
# BM25_SCALING_FACTOR=2 but to 0.78 with BM25_SCALING_FACTOR=8 (default). The defaults were chosen empirically.
# Increase the default if most unscaled scores are larger than expected (>30) and otherwise would incorrectly
# all be mapped to scores ~1.
BM25_SCALING_FACTOR = 8


@document_store
class ElasticsearchDocumentStore:
    def __init__(self, *, hosts: Optional[Hosts] = None, index: str = "default", **kwargs):
        """
        Creates a new ElasticsearchDocumentStore instance.

        For more information on connection parameters, see the official Elasticsearch documentation:
        https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/connecting.html

        For the full list of supported kwargs, see the official Elasticsearch reference:
        https://elasticsearch-py.readthedocs.io/en/stable/api.html#module-elasticsearch

        :param hosts: List of hosts running the Elasticsearch client. Defaults to None
        :param index: Name of index in Elasticsearch, if it doesn't exist it will be created. Defaults to "default"
        :param **kwargs: Optional arguments that ``Elasticsearch`` takes.
        """
        self._hosts = hosts
        self._client = Elasticsearch(hosts, **kwargs)
        self._index = index
        self._kwargs = kwargs

        # Check client connection, this will raise if not connected
        self._client.info()

        # Create the index if it doesn't exist
        if not self._client.indices.exists(index=index):
            self._client.indices.create(index=index)

    def to_dict(self) -> Dict[str, Any]:
        # This is not the best solution to serialise this class but is the fastest to implement.
        # Not all kwargs types can be serialised to text so this can fail. We must serialise each
        # type explicitly to handle this properly.
        return default_to_dict(
            self,
            hosts=self._hosts,
            index=self._index,
            **self._kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElasticsearchDocumentStore":
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        return self._client.count(index=self._index)["count"]

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical operator (`"$and"`,
        `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `$ne`, `"$in"`, `$nin`, `"$gt"`, `"$gte"`, `"$lt"`,
        `"$lte"`) or a metadata field name.

        Logical operator keys take a dictionary of metadata field names and/or logical operators as value. Metadata
        field names take a dictionary of comparison operators as value. Comparison operator keys take a single value or
        (in case of `"$in"`) a list of values as value. If no logical operator is provided, `"$and"` is used as default
        operation. If no comparison operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used
        as default operation.

        Example:

        ```python
        filters = {
            "$and": {
                "type": {"$eq": "article"},
                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                "rating": {"$gte": 3},
                "$or": {
                    "genre": {"$in": ["economy", "politics"]},
                    "publisher": {"$eq": "nytimes"}
                }
            }
        }
        # or simpler using default operators
        filters = {
            "type": "article",
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": ["economy", "politics"],
                "publisher": "nytimes"
            }
        }
        ```

        To use the same logical operator multiple times on the same level, logical operators can take a list of
        dictionaries as value.

        Example:

        ```python
        filters = {
            "$or": [
                {
                    "$and": {
                        "Type": "News Paper",
                        "Date": {
                            "$lt": "2019-01-01"
                        }
                    }
                },
                {
                    "$and": {
                        "Type": "Blog Post",
                        "Date": {
                            "$gte": "2019-01-01"
                        }
                    }
                }
            ]
        }
        ```

        :param filters: the filters to apply to the document list.
        :return: a list of Documents that match the given filters.
        """
        query = {"bool": {"filter": _normalize_filters(filters)}} if filters else None

        documents: List[Document] = []
        from_ = 0
        # Handle pagination
        while True:
            res = self._client.search(
                index=self._index,
                query=query,
                from_=from_,
            )
            documents.extend(self._deserialize_document(hit) for hit in res["hits"]["hits"])
            from_ = len(documents)
            if from_ >= res["hits"]["total"]["value"]:
                break
        return documents

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> None:
        """
        Writes (or overwrites) documents into the store.

        :param documents: a list of documents.
        :param policy: documents with the same ID count as duplicates. When duplicates are met,
            the store can:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised
        :raises DuplicateDocumentError: Exception trigger on duplicate document if `policy=DuplicatePolicy.FAIL`
        :return: None
        """
        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        action = "index" if policy == DuplicatePolicy.OVERWRITE else "create"
        _, errors = helpers.bulk(
            client=self._client,
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
        if errors and policy == DuplicatePolicy.FAIL:
            # TODO: Handle errors in a better way, we're assuming that all errors
            # are related to duplicate documents but that could be very well be wrong.

            # mypy complains that `errors`` could be either `int` or a `list` of `dict`s.
            # Since the type depends on the parameters passed to `helpers.bulk()`` we know
            # for sure that it will be a `list`.
            ids = ", ".join(e["create"]["_id"] for e in errors)  # type: ignore[union-attr]
            msg = f"IDs '{ids}' already exist in the document store."
            raise DuplicateDocumentError(msg)

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

        #
        helpers.bulk(
            client=self._client,
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
        scale_score: bool = True,
    ) -> List[Document]:
        """
        Elasticsearch by defaults uses BM25 search algorithm.
        Even though this method is called `bm25_retrieval` it searches for `query`
        using the search algorithm `_client` was configured with.

        This method is not mean to be part of the public interface of
        `ElasticsearchDocumentStore` nor called directly.
        `ElasticsearchBM25Retriever` uses this method directly and is the public interface for it.

        `query` must be a non empty string, otherwise a `ValueError` will be raised.

        :param query: String to search in saved Documents' text.
        :param filters: Filters applied to the retrieved Documents, for more info
                        see `ElasticsearchDocumentStore.filter_documents`, defaults to None
        :param fuzziness: Fuzziness parameter passed to Elasticsearch, defaults to "AUTO".
                          see the official documentation for valid values:
                          https://www.elastic.co/guide/en/elasticsearch/reference/current/common-options.html#fuzziness
        :param top_k: Maximum number of Documents to return, defaults to 10
        :param scale_score: If `True` scales the Document`s scores between 0 and 1, defaults to True
        :raises ValueError: If `query` is an empty string
        :return: List of Document that match `query`
        """

        if not query:
            msg = "query must be a non empty string"
            raise ValueError(msg)

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
                                "operator": "AND",
                            }
                        }
                    ]
                }
            },
        }

        if filters:
            body["query"]["bool"]["filter"] = _normalize_filters(filters)

        res = self._client.search(index=self._index, **body)

        docs = []
        for hit in res["hits"]["hits"]:
            if scale_score:
                hit["_score"] = float(1 / (1 + np.exp(-np.asarray(hit["_score"] / BM25_SCALING_FACTOR))))
            docs.append(self._deserialize_document(hit))
        return docs
