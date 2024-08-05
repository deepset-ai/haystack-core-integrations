# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

import chromadb
import numpy as np
from chromadb.api.types import GetResult, QueryResult, validate_where, validate_where_document
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy

from .errors import ChromaDocumentStoreFilterError
from .utils import get_embedding_function

logger = logging.getLogger(__name__)


VALID_DISTANCE_FUNCTIONS = "l2", "cosine", "ip"
SUPPORTED_TYPES_FOR_METADATA_VALUES = str, int, float, bool


class ChromaDocumentStore:
    """
    A document store using [Chroma](https://docs.trychroma.com/) as the backend.

    We use the `collection.get` API to implement the document store protocol,
    the `collection.search` API will be used in the retriever instead.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_function: str = "default",
        persist_path: Optional[str] = None,
        distance_function: Literal["l2", "cosine", "ip"] = "l2",
        metadata: Optional[dict] = None,
        **embedding_function_params,
    ):
        """
        Initializes the store. The __init__ constructor is not part of the Store Protocol
        and the signature can be customized to your needs. For example, parameters needed
        to set up a database client would be passed to this method.

        Note: for the component to be part of a serializable pipeline, the __init__
        parameters must be serializable, reason why we use a registry to configure the
        embedding function passing a string.

        :param collection_name: the name of the collection to use in the database.
        :param embedding_function: the name of the embedding function to use to embed the query
        :param persist_path: where to store the database. If None, the database will be `in-memory`.
        :param distance_function: The distance metric for the embedding space.
            - `"l2"` computes the Euclidean (straight-line) distance between vectors,
            where smaller scores indicate more similarity.
            - `"cosine"` computes the cosine similarity between vectors,
            with higher scores indicating greater similarity.
            - `"ip"` stands for inner product, where higher scores indicate greater similarity between vectors.
            **Note**: `distance_function` can only be set during the creation of a collection.
            To change the distance metric of an existing collection, consider cloning the collection.
        :param metadata: a dictionary of chromadb collection parameters passed directly to chromadb's client
            method `create_collection`. If it contains the key `"hnsw:space"`, the value will take precedence over the
            `distance_function` parameter above.

        :param embedding_function_params: additional parameters to pass to the embedding function.
        """

        if distance_function not in VALID_DISTANCE_FUNCTIONS:
            error_message = (
                f"Invalid distance_function: '{distance_function}' for the collection. "
                f"Valid options are: {VALID_DISTANCE_FUNCTIONS}."
            )
            raise ValueError(error_message)

        # Store the params for marshalling
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._embedding_function_params = embedding_function_params
        self._persist_path = persist_path
        self._distance_function = distance_function
        # Create the client instance
        if persist_path is None:
            self._chroma_client = chromadb.Client()
        else:
            self._chroma_client = chromadb.PersistentClient(path=persist_path)

        embedding_func = get_embedding_function(embedding_function, **embedding_function_params)

        metadata = metadata or {}
        if "hnsw:space" not in metadata:
            metadata["hnsw:space"] = distance_function

        if collection_name in [c.name for c in self._chroma_client.list_collections()]:
            self._collection = self._chroma_client.get_collection(collection_name, embedding_function=embedding_func)

            if metadata != self._collection.metadata:
                logger.warning(
                    "Collection already exists. The `distance_function` and `metadata` parameters will be ignored."
                )
        else:
            self._collection = self._chroma_client.create_collection(
                name=collection_name,
                metadata=metadata,
                embedding_function=embedding_func,
            )

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns: how many documents are present in the document store.
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
        :returns: a list of Documents that match the given filters.
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

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int:
        """
        Writes (or overwrites) documents into the store.

        :param documents:
            A list of documents to write into the document store.
        :param policy:
            Not supported at the moment.

        :raises ValueError:
            When input is not valid.

        :returns:
            The number of documents written
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
                valid_meta = {}
                discarded_keys = []

                for k, v in doc.meta.items():
                    if isinstance(v, SUPPORTED_TYPES_FOR_METADATA_VALUES):
                        valid_meta[k] = v
                    else:
                        discarded_keys.append(k)

                if discarded_keys:
                    logger.warning(
                        "Document %s contains `meta` values of unsupported types for the keys: %s. "
                        "These items will be discarded. Supported types are: %s.",
                        doc.id,
                        ", ".join(discarded_keys),
                        ", ".join([t.__name__ for t in SUPPORTED_TYPES_FOR_METADATA_VALUES]),
                    )

                if valid_meta:
                    data["metadatas"] = [valid_meta]

            if doc.embedding is not None:
                data["embeddings"] = [doc.embedding]

            if hasattr(doc, "sparse_embedding") and doc.sparse_embedding is not None:
                logger.warning(
                    "Document %s has the `sparse_embedding` field set,"
                    "but storing sparse embeddings in Chroma is not currently supported."
                    "The `sparse_embedding` field will be ignored.",
                    doc.id,
                )

            self._collection.add(**data)

        return len(documents)

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.

        :param document_ids: the object_ids to delete
        """
        self._collection.delete(ids=document_ids)

    def search(self, queries: List[str], top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[List[Document]]:
        """Search the documents in the store using the provided text queries.

        :param queries: the list of queries to search for.
        :param top_k: top_k documents to return for each query.
        :param filters: a dictionary of filters to apply to the search. Accepts filters in haystack format.
        :returns: matching documents for each query.
        """
        if filters is None:
            results = self._collection.query(
                query_texts=queries,
                n_results=top_k,
                include=["embeddings", "documents", "metadatas", "distances"],
            )
        else:
            chroma_filters = self._normalize_filters(filters=filters)
            results = self._collection.query(
                query_texts=queries,
                n_results=top_k,
                where=chroma_filters[1],
                where_document=chroma_filters[2],
                include=["embeddings", "documents", "metadatas", "distances"],
            )

        return self._query_result_to_documents(results)

    def search_embeddings(
        self, query_embeddings: List[List[float]], top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[List[Document]]:
        """
        Perform vector search on the stored document, pass the embeddings of the queries instead of their text.

        :param query_embeddings: a list of embeddings to use as queries.
        :param top_k: the maximum number of documents to retrieve.
        :param filters: a dictionary of filters to apply to the search. Accepts filters in haystack format.

        :returns: a list of lists of documents that match the given filters.

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
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            collection_name=self._collection_name,
            embedding_function=self._embedding_function,
            persist_path=self._persist_path,
            distance_function=self._distance_function,
            **self._embedding_function_params,
        )

    @staticmethod
    def _normalize_filters(filters: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
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

        final_where = dict(filters)
        final_where.update(dict(where))
        try:
            if final_where:
                validate_where(final_where)
            if where_document:
                validate_where_document(where_document)
        except ValueError as e:
            raise ChromaDocumentStoreFilterError(e) from e

        return ids, final_where, where_document

    @staticmethod
    def _get_result_to_documents(result: GetResult) -> List[Document]:
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

    @staticmethod
    def _query_result_to_documents(result: QueryResult) -> List[List[Document]]:
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
                metadatas = result.get("metadatas")
                try:
                    if metadatas and metadatas[i][j] is not None:
                        document_dict["meta"] = metadatas[i][j]
                except IndexError:
                    pass

                if embeddings := result.get("embeddings"):
                    document_dict["embedding"] = np.array(embeddings[i][j])

                if distances := result.get("distances"):
                    document_dict["score"] = distances[i][j]

                converted_answers.append(Document.from_dict(document_dict))
            retval.append(converted_answers)

        return retval
