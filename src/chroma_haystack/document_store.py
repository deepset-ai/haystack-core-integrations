# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
import pandas as pd
from chromadb.api.types import GetResult, validate_where, validate_where_document
from haystack.preview.dataclasses import ContentType, Document
from haystack.preview.document_stores.decorator import document_store
from haystack.preview.document_stores.protocols import DuplicatePolicy

from chroma_haystack.errors import ChromaDocumentStoreError, ChromaDocumentStoreFilterError
from chroma_haystack.utils import get_embedding_function

logger = logging.getLogger(__name__)


@document_store
class ChromaDocumentStore:
    """
    We use the `collection.get` API to implement the document store protocol,
    the `collection.search` API will be used in the retriever instead.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_function: str = "default",
    ):
        """
        Initializes the store. The __init__ constructor is not part of the Store Protocol
        and the signature can be customized to your needs. For example, parameters needed
        to set up a database client would be passed to this method.

        Note: for the component to be part of a serializable pipelie, the __init__
        parameters must be serializable, reason why we use a registry to configure the
        embedding function passing a string.
        """
        self._chroma_client = chromadb.Client()
        self._collection = self._chroma_client.create_collection(
            name=collection_name, embedding_function=get_embedding_function(embedding_function)()
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

        return self._result_to_documents(result)

    def write_documents(self, documents: List[Document], _: DuplicatePolicy = DuplicatePolicy.FAIL) -> None:
        """
        Writes (or overwrites) documents into the store.

        :param documents: a list of documents.
        :param policy: not supported at the moment
        :raises DuplicateDocumentError: Exception trigger on duplicate document if `policy=DuplicatePolicy.FAIL`
        :return: None
        """
        for d in documents:
            if not isinstance(d, Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

            doc = self._prepare(d)
            content = self._content_as_text(doc.content_type, doc.content)

            data = {"ids": [doc.id], "documents": [content], "metadatas": [doc.metadata]}

            if doc.embedding is not None:
                data["embeddings"] = [doc.embedding.tolist()]

            self._collection.add(**data)

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.
        Fails with `MissingDocumentError` if no document with this id is present in the store.

        :param object_ids: the object_ids to delete
        """
        self._collection.delete(ids=document_ids)

    def _normalize_filters(self, filters: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
        """
        Translate Haystack filters to Chroma filters. It returns three dictionaries, to be
        passed to `ids`, `where` and `where_document` respectively.
        """
        if type(filters) is not dict:
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
            elif field == "content_type":
                # Schedule for removal the original key, we're going to change it
                keys_to_remove.append(field)
                where["_content_type"] = value

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

    def _content_as_text(self, content_type: ContentType, content: Any) -> str:
        if content_type == "text":
            return content
        if content_type == "table":
            return content.to_json()
        elif content_type == "audio":
            return content.absolute()
        elif content_type == "image":
            return content.absolute()

        msg = f"Unknown content_type: {content_type}"
        raise ChromaDocumentStoreError(msg)

    def _content_from_text(self, content_type: ContentType, content: str) -> Any:
        if content_type == "text":
            return content
        elif content_type == "table":
            return pd.read_json(content)
        elif content_type == "audio":
            return Path(content)
        elif content_type == "image":
            return Path(content)

        msg = f"Unknown content_type: {content_type}"
        raise ChromaDocumentStoreError(msg)

    def _prepare(self, d: Document) -> Document:
        """
        Change the document in a way we can better store it into Chroma.
        Fore example, we store as metadata additional fields Chroma doesn't manage
        """
        new_meta = {"_content_type": d.content_type} | d.metadata
        orig = d.to_dict()
        orig["metadata"] = new_meta
        # return a copy
        return Document.from_dict(orig)

    def _result_to_documents(self, result: GetResult) -> List[Document]:
        """
        Helper function to convert Chroma results into Haystack Documents
        """
        retval = []
        for i in range(len(result["documents"])):
            # prepare metadata
            metadata = result["metadatas"][i]
            content_type = metadata.pop("_content_type")
            content = self._content_from_text(content_type, result["documents"][i])

            document_dict = {
                "id": result["ids"][i],
                "content": content,
                "metadata": metadata,
                "content_type": content_type,
            }

            if result["embeddings"]:
                document_dict["embedding"] = np.ndarray(result["embeddings"][i])

            retval.append(Document.from_dict(document_dict))

        return retval
