# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from copy import copy
from typing import Any, Dict, List, Optional

import numpy as np
import vecs
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError, MissingDocumentError
from haystack.document_stores.protocol import DuplicatePolicy

from supabase_haystack.filters import _normalize_filters

logger = logging.getLogger(__name__)


class SupabaseDocumentStore:
    def __init__(
            self,
            user:str,
            password:str,
            host:str,
            port:str,
            db_name:str,
            collection_name:str = "documents",
            dimension:int = 768,
            **collection_creation_kwargs,
    ):
        """
        Creates a new SupabaseDocumentStore instance.

        For more information on connection parameters, see the official supabase vector documentation: https://supabase.github.io/vecs/0.4/

        :param user: The username for connecting to the Supabase database.
        :param password: The password for connecting to the Supabase database.
        :param host: The host address of the Supabase database server.
        :param port: The port number on which the Supanase database server is listening.
        :param db_name: The name of the Supabase database.
        :param collection_name: The name of the collection or table in the database where vectors will be stored.
        :param dimension: The dimensionality of the vectors to be stored in the document store.
        :param **collection_creation_kwargs: Optional arguments that ``Supabase Document Store`` takes.
        """
        self._collection_name = collection_name
        self._dummy_vector = [0.0]*dimension
        self._adapter = collection_creation_kwargs["adapter"]
        db_connection = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
        self._pgvector_client = vecs.create_client(db_connection)
        self._collection = self._pgvector_client.get_or_create_collection(name=collection_name, dimension=dimension, **collection_creation_kwargs)

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        return self._collection.__len__()

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        Filters are defined as nested dictionaries that can be of two types:
        - Comparison
        - Logic

        Comparison dictionaries must contain the keys:

        - `field`
        - `operator`
        - `value`

        Logic dictionaries must contain the keys:

        - `operator`
        - `conditions`

        The `conditions` key must be a list of dictionaries, either of type Comparison or Logic.

        The `operator` value in Comparison dictionaries must be one of:

        - `==`
        - `!=`
        - `>`
        - `>=`
        - `<`
        - `<=`
        - `in`
        - `not in`

        The `operator` values in Logic dictionaries must be one of:

        - `NOT`
        - `OR`
        - `AND`


        A simple filter:
        ```python
        filters = {"field": "meta.type", "operator": "==", "value": "article"}
        ```

        A more complex filter:
        ```python
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.date", "operator": ">=", "value": 1420066800},
                {"field": "meta.date", "operator": "<", "value": 1609455600},
                {"field": "meta.rating", "operator": ">=", "value": 3},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
            ],
        }

        :param filters: the filters to apply to the document list.
        :return: a list of Documents that match the given filters.
        """
        filters = _normalize_filters(filters)

        # pgvector store performs vector similarity search
        # here we are querying with a dummy vector and the max compatible top_k
        documents = self._embedding_retrieval(
            query_embedding=self._dummy_vector,
            filters=filters,
        )

        return self._convert_query_result_to_documents(documents)

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes (or overwrites) documents into the store.

        :param documents: a list of documents.
        :param policy: The duplicate policy to use when writing documents.
            SupabaseDocumentStore only supports `DuplicatePolicy.OVERWRITE`.

        :return: None
        """
        if policy not in [DuplicatePolicy.NONE, DuplicatePolicy.OVERWRITE]:
            logger.warning(
                f"SupabaseDocumentStore only supports `DuplicatePolicy.OVERWRITE`"
                f"but got {policy}. Overwriting duplicates is enabled by default."
            )


        for doc in documents:
            if not isinstance(doc, Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)
            if doc.content is None:
                logger.warning(
                    "SupabaseDocumentStore can only store the text field of Documents: "
                    "'array', 'dataframe' and 'blob' will be dropped."
                )

            if self._adapter is not None:
                data = (doc.id, doc.content, {"content":doc.content, **doc.meta})
                self._collection.upsert(records=[data])
            else:
                embedding = copy(doc.embedding)
                if doc.embedding is None:
                    logger.warning(
                    f"Document {doc.id} has no embedding. pgvector is a purely vector database. "
                    "A dummy embedding will be used, but this can affect the search results. "
                )
                    embedding = self._dummy_vector

                data = (doc.id, embedding, {"content":doc.content, **doc.meta})
                self._collection.upsert(records=[data])

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.

        :param document_ids: the document ids to delete
        """
        self._collection.delete(document_ids)

    def _convert_query_result_to_documents(self, result) -> List[Document]:
        """
        Helper function to convert Supabase results into Haystack Documents
        """
        documents = []
        for i in result:
            document_dict: Dict[str, Any] = {"id":i[0]}
            document_dict["embedding"] = np.array(i[1])
            metadata = i[2]
            document_dict["content"] = metadata["content"]
            del metadata["content"]
            document_dict["meta"] = metadata
            documents.append(Document.from_dict(dict))

        return documents

    def _embedding_retrieval(
        self,
        query_embedding: List[float],
        *,
        filters: Optional[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents. Defaults to None.
        :param top_k: Maximum number of Documents to return, defaults to 10

        :return: List of Document that are most similar to `query_embedding`
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)

        filters = _normalize_filters(filters)

        results = self._collection.query(
            data=query_embedding,
            limit=top_k,
            filters=filters,
            include_value=True,
            include_metadata=True
        )

        return self._convert_query_result_to_documents(result=results)

