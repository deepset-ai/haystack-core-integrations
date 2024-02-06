# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.api.types import GetResult, QueryResult, validate_where, validate_where_document
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy

from .errors import ChromaDocumentStoreFilterError
from .utils import get_embedding_function

logger = logging.getLogger(__name__)


class ChromaDocumentStore:
    """
    We use the `collection.get` API to implement the document store protocol,
    the `collection.search` API will be used in the retriever instead.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_function: str = "default",
        persist_path: Optional[str] = None,
        **embedding_function_params,
    ):
        """
        Initializes the store. The __init__ constructor is not part of the Store Protocol
        and the signature can be customized to your needs. For example, parameters needed
        to set up a database client would be passed to this method.

        Note: for the component to be part of a serializable pipelie, the __init__
        parameters must be serializable, reason why we use a registry to configure the
        embedding function passing a string.
        """
        # Store the params for marshalling
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._embedding_function_params = embedding_function_params
        # Create the client instance
        if persist_path is None:
            self._chroma_client = chromadb.Client()
        else:
            self._chroma_client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=get_embedding_function(embedding_function, **embedding_function_params),
        )

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        return self._collection.count()

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
        if filters:
            ids, where, where_document = self._normalize_filters(filters)
            kwargs: Dict[str, Any] = {"where": where}

            if ids:
                kwargs["ids"] = ids
            if where_document:
                kwargs["where_document"] = where_document

            result = self._collection.get(**kwargs)
        else:
            result = self._collection.get()

        return self._get_result_to_documents(result)

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> None:
        """
        Writes (or overwrites) documents into the store.

        :param documents: a list of documents.
        :param policy: not supported at the moment
        :raises DuplicateDocumentError: Exception trigger on duplicate document if `policy=DuplicatePolicy.FAIL`
        :return: None
        """
        for doc in documents:
            if not isinstance(doc, Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

            if doc.content is None:
                logger.warning(
                    "ChromaDocumentStore can only store the text field of Documents: "
                    "'array', 'dataframe' and 'blob' will be dropped."
                )
            data = {"ids": [doc.id], "documents": [doc.content]}

            if doc.meta:
                data["metadatas"] = [doc.meta]

            if doc.embedding is not None:
                data["embeddings"] = [doc.embedding]

            self._collection.add(**data)

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.
        Fails with `MissingDocumentError` if no document with this id is present in the store.

        :param object_ids: the object_ids to delete
        """
        self._collection.delete(ids=document_ids)

    def search(self, queries: List[str], top_k: int) -> List[List[Document]]:
        """
        Perform vector search on the stored documents
        """
        results = self._collection.query(
            query_texts=queries, n_results=top_k, include=["embeddings", "documents", "metadatas", "distances"]
        )
        return self._query_result_to_documents(results)

    def search_embeddings(
        self, query_embeddings: List[List[float]], top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[List[Document]]:
        """
        Perform vector search on the stored document, pass the embeddings of the queries
        instead of their text.

        Accepts filters in haystack format.
        """
        if filters is None:
            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                include=["embeddings", "documents", "metadatas", "distances"],
            )
        else:
            chroma_filters = self._normalize_filters(filters=filters)
            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                where=chroma_filters[1],
                where_document=chroma_filters[2],
                include=["embeddings", "documents", "metadatas", "distances"],
            )

        return self._query_result_to_documents(results)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChromaDocumentStore":
        return ChromaDocumentStore(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection_name": self._collection_name,
            "embedding_function": self._embedding_function,
            **self._embedding_function_params,
        }

    def _normalize_filters(self, filters: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
        """
        Translate Haystack filters to Chroma filters. It returns three dictionaries, to be
        passed to `ids`, `where` and `where_document` respectively.
        """
        if not isinstance(filters, dict):
            msg = "'filters' parameter must be a dictionary"
            raise ChromaDocumentStoreFilterError(msg)

        ids = []
        where = defaultdict(list)
        where_document = defaultdict(list)
        keys_to_remove = []

        for field, value in filters.items():
            if field == "content":
                # Schedule for removal the original key, we're going to change it
                keys_to_remove.append(field)
                where_document["$contains"] = value
            elif field == "id":
                # Schedule for removal the original key, we're going to change it
                keys_to_remove.append(field)
                ids.append(value)
            elif isinstance(value, (list, tuple)):
                # Schedule for removal the original key, we're going to change it
                keys_to_remove.append(field)

                # if the list is empty the filter is invalid, let's just remove it
                if len(value) == 0:
                    continue

                # if the list has a single item, just make it a regular key:value filter pair
                if len(value) == 1:
                    where[field] = value[0]
                    continue

                # if the list contains multiple items, we need an $or chain
                for v in value:
                    where["$or"].append({field: v})

        for k in keys_to_remove:
            del filters[k]

        final_where = dict(filters | where)
        try:
            if final_where:
                validate_where(final_where)
            if where_document:
                validate_where_document(where_document)
        except ValueError as e:
            raise ChromaDocumentStoreFilterError(e) from e

        return ids, final_where, where_document

    def _get_result_to_documents(self, result: GetResult) -> List[Document]:
        """
        Helper function to convert Chroma results into Haystack Documents
        """
        retval = []
        for i in range(len(result.get("documents", []))):
            document_dict: Dict[str, Any] = {"id": result["ids"][i]}

            result_documents = result.get("documents")
            if result_documents:
                document_dict["content"] = result_documents[i]

            result_metadata = result.get("metadatas")
            # Ensure metadata[i] is not None or don't add it to the document dict
            if result_metadata and result_metadata[i]:
                document_dict["meta"] = result_metadata[i]

            result_embeddings = result.get("embeddings")
            if result_embeddings:
                document_dict["embedding"] = list(result_embeddings[i])

            retval.append(Document.from_dict(document_dict))

        return retval

    def _query_result_to_documents(self, result: QueryResult) -> List[List[Document]]:
        """
        Helper function to convert Chroma results into Haystack Documents
        """
        retval: List[List[Document]] = []
        documents = result.get("documents")
        if documents is None:
            return retval

        for i, answers in enumerate(documents):
            converted_answers = []
            for j in range(len(answers)):
                document_dict: Dict[str, Any] = {
                    "id": result["ids"][i][j],
                    "content": documents[i][j],
                }

                # prepare metadata
                if metadatas := result.get("metadatas"):
                    document_dict["meta"] = dict(metadatas[i][j])

                if embeddings := result.get("embeddings"):
                    document_dict["embedding"] = np.array(embeddings[i][j])

                if distances := result.get("distances"):
                    document_dict["score"] = distances[i][j]

                converted_answers.append(Document.from_dict(document_dict))
            retval.append(converted_answers)

        return retval
