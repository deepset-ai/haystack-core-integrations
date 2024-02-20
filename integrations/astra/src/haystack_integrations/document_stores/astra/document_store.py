# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0
import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError, MissingDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from .astra_client import AstraClient
from .errors import AstraDocumentStoreFilterError
from .filters import _convert_filters

logger = logging.getLogger(__name__)


MAX_BATCH_SIZE = 20


def _batches(input_list, batch_size):
    input_length = len(input_list)
    for ndx in range(0, input_length, batch_size):
        yield input_list[ndx : min(ndx + batch_size, input_length)]


class AstraDocumentStore:
    """
    An AstraDocumentStore document store for Haystack.
    """

    def __init__(
        self,
        api_endpoint: Secret = Secret.from_env_var("ASTRA_DB_API_ENDPOINT"),  # noqa: B008
        token: Secret = Secret.from_env_var("ASTRA_DB_APPLICATION_TOKEN"),  # noqa: B008
        collection_name: str = "documents",
        embedding_dimension: int = 768,
        duplicates_policy: DuplicatePolicy = DuplicatePolicy.NONE,
        similarity: str = "cosine",
    ):
        """
        The connection to Astra DB is established and managed through the JSON API.
        The required credentials (api endpoint andapplication token) can be generated
        through the UI by clicking and the connect tab, and then selecting JSON API and
        Generate Configuration.

        :param api_endpoint: The Astra DB API endpoint.
        :param token: The Astra DB application token.
        :param collection_name: The current collection in the keyspace in the current Astra DB.
        :param embedding_dimension: Dimension of embedding vector.
        :param duplicates_policy: Handle duplicate documents based on DuplicatePolicy parameter options.
              Parameter options : (SKIP, OVERWRITE, FAIL, NONE)
              - `DuplicatePolicy.NONE`: Default policy, If a Document with the same id already exists,
                    it is skipped and not written.
              - `DuplicatePolicy.SKIP`: If a Document with the same id already exists, it is skipped and not written.
              - `DuplicatePolicy.OVERWRITE`: If a Document with the same id already exists, it is overwritten.
              - `DuplicatePolicy.FAIL`: If a Document with the same id already exists, an error is raised.
        :param similarity: The similarity function used to compare document vectors.
        """
        resolved_api_endpoint = api_endpoint.resolve_value()
        if resolved_api_endpoint is None:
            msg = (
                "AstraDocumentStore expects the API endpoint. "
                "Set the ASTRA_DB_API_ENDPOINT environment variable (recommended) or pass it explicitly."
            )
            raise ValueError(msg)

        resolved_token = token.resolve_value()
        if resolved_token is None:
            msg = (
                "AstraDocumentStore expects an authentication token. "
                "Set the ASTRA_DB_APPLICATION_TOKEN environment variable (recommended) or pass it explicitly."
            )
            raise ValueError(msg)

        self.api_endpoint = api_endpoint
        self.token = token
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.duplicates_policy = duplicates_policy
        self.similarity = similarity

        self.index = AstraClient(
            resolved_api_endpoint,
            resolved_token,
            self.collection_name,
            self.embedding_dimension,
            self.similarity,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AstraDocumentStore":
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_endpoint", "token"])
        return default_from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            api_endpoint=self.api_endpoint.to_dict(),
            token=self.token.to_dict(),
            collection_name=self.collection_name,
            embedding_dimension=self.embedding_dimension,
            duplicates_policy=self.duplicates_policy.name,
            similarity=self.similarity,
        )

    def write_documents(
        self,
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ):
        """
        Indexes documents for later queries.

        :param documents: a list of Haystack Document objects.
        :param policy:  Handle duplicate documents based on DuplicatePolicy parameter options.
                      Parameter options : (SKIP, OVERWRITE, FAIL, NONE)
                      - `DuplicatePolicy.NONE`: Default policy, If a Document with the same id already exists,
                            it is skipped and not written.
                      - `DuplicatePolicy.SKIP`: If a Document with the same id already exists,
                            it is skipped and not written.
                      - `DuplicatePolicy.OVERWRITE`: If a Document with the same id already exists, it is overwritten.
                      - `DuplicatePolicy.FAIL`: If a Document with the same id already exists, an error is raised.
        :return: int
        """
        if policy is None or policy == DuplicatePolicy.NONE:
            if self.duplicates_policy is not None and self.duplicates_policy != DuplicatePolicy.NONE:
                policy = self.duplicates_policy
            else:
                policy = DuplicatePolicy.SKIP

        batch_size = MAX_BATCH_SIZE

        def _convert_input_document(document: Union[dict, Document]):
            if isinstance(document, Document):
                document_dict = asdict(document)
            elif isinstance(document, dict):
                document_dict = document
            else:
                msg = f"Unsupported type for documents, documents is of type {type(document)}."
                raise ValueError(msg)

            if "id" in document_dict:
                if "_id" not in document_dict:
                    document_dict["_id"] = document_dict.pop("id")
                elif "_id" in document_dict:
                    msg = f"Duplicate id definitions, both 'id' and '_id' present in document {document_dict}"
                    raise Exception(msg)
            if "_id" in document_dict:
                if not isinstance(document_dict["_id"], str):
                    msg = (
                        f"Document id {document_dict['_id']} is not a string, "
                        f"but is of type {type(document_dict['_id'])}"
                    )
                    raise Exception(msg)

            if "dataframe" in document_dict and document_dict["dataframe"] is not None:
                document_dict["dataframe"] = document_dict.pop("dataframe").to_json()
            if embedding := document_dict.pop("embedding", []):
                document_dict["$vector"] = embedding

            return document_dict

        documents_to_write = [_convert_input_document(doc) for doc in documents]

        duplicate_documents = []
        new_documents = []
        i = 0
        while i < len(documents_to_write):
            doc = documents_to_write[i]
            response = self.index.find_documents({"filter": {"_id": doc["_id"]}})
            if response:
                if policy == DuplicatePolicy.FAIL:
                    msg = f"ID '{doc['_id']}' already exists."
                    raise DuplicateDocumentError(msg)
                duplicate_documents.append(doc)
            else:
                new_documents.append(doc)
            i = i + 1

        insertion_counter = 0
        if policy == DuplicatePolicy.SKIP:
            if len(new_documents) > 0:
                for batch in _batches(new_documents, batch_size):
                    inserted_ids = self.index.insert(batch)  # type: ignore
                    insertion_counter += len(inserted_ids)
                    logger.info(f"write_documents inserted documents with id {inserted_ids}")
            else:
                logger.warning("No documents written. Argument policy set to SKIP")

        elif policy == DuplicatePolicy.OVERWRITE:
            if len(new_documents) > 0:
                for batch in _batches(new_documents, batch_size):
                    inserted_ids = self.index.insert(batch)  # type: ignore
                    insertion_counter += len(inserted_ids)
                    logger.info(f"write_documents inserted documents with id {inserted_ids}")
            else:
                logger.warning("No documents written. Argument policy set to OVERWRITE")

            if len(duplicate_documents) > 0:
                updated_ids = []
                for duplicate_doc in duplicate_documents:
                    updated = self.index.update_document(duplicate_doc, "_id")  # type: ignore
                    if updated:
                        updated_ids.append(duplicate_doc["_id"])
                insertion_counter = insertion_counter + len(updated_ids)
                logger.info(f"write_documents updated documents with id {updated_ids}")
            else:
                logger.info("No documents updated. Argument policy set to OVERWRITE")

        elif policy == DuplicatePolicy.FAIL:
            if len(new_documents) > 0:
                for batch in _batches(new_documents, batch_size):
                    inserted_ids = self.index.insert(batch)  # type: ignore
                    insertion_counter = insertion_counter + len(inserted_ids)
                    logger.info(f"write_documents inserted documents with id {inserted_ids}")
            else:
                logger.warning("No documents written. Argument policy set to FAIL")

        return insertion_counter

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        return self.index.count_documents()

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Returns at most 1000 documents that match the filter

        Args:
            filters (Optional[Dict[str, Any]], optional): Filters to apply. Defaults to None.

        Raises:
            AstraDocumentStoreFilterError: If the filter is invalid or not supported by this class.

        Returns:
            List[Document]: A list of matching documents.
        """
        if not isinstance(filters, dict) and filters is not None:
            msg = "Filters must be a dictionary or None"
            raise AstraDocumentStoreFilterError(msg)

        if filters is not None:
            if "id" in filters:
                filters["_id"] = filters.pop("id")
        vector = None
        if filters is not None and "embedding" in filters.keys():
            if "$in" in filters["embedding"]:
                embeds = filters.pop("embedding")
                vectors = embeds["$in"]
            else:
                filters["$vector"] = filters.pop("embedding")
                vectors = [filters.pop("$vector")]
            documents = []
            for vector in vectors:
                converted_filters = _convert_filters(filters)
                results = self.index.query(
                    vector=vector,
                    query_filter=converted_filters,
                    top_k=1000,
                    include_values=True,
                    include_metadata=True,
                )
                documents.extend(self._get_result_to_documents(results))
        else:
            converted_filters = _convert_filters(filters)
            results = self.index.query(
                vector=vector, query_filter=converted_filters, top_k=1000, include_values=True, include_metadata=True
            )
            documents = self._get_result_to_documents(results)
        return documents

    @staticmethod
    def _get_result_to_documents(results) -> List[Document]:
        documents = []
        for match in results.matches:
            dataframe = match.metadata.pop("dataframe", None)
            if dataframe is not None:
                df = pd.DataFrame.from_dict(json.loads(dataframe))
            else:
                df = None
            document = Document(
                content=match.text,
                id=match.document_id,
                embedding=match.values,
                dataframe=df,
                blob=match.metadata.pop("blob", None),
                meta=match.metadata.pop("meta", None),
                score=match.score,
            )
            documents.append(document)
        return documents

    def get_documents_by_id(self, ids: List[str]) -> List[Document]:
        """
        Returns documents with given ids.
        """
        results = self.index.get_documents(ids=ids)
        ret = self._get_result_to_documents(results)
        return ret

    def get_document_by_id(self, document_id: str) -> Document:
        """
        :param document_id: id of the document to retrieve
        Returns documents with given ids.
        Raises MissingDocumentError when document_id does not exist in document store
        """
        document = self.index.get_documents(ids=[document_id])
        ret = self._get_result_to_documents(document)
        if not ret:
            msg = f"Document {document_id} does not exist"
            raise MissingDocumentError(msg)
        return ret[0]

    def search(
        self, query_embedding: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Perform a search for a list of queries.

        Args:
            query_embedding (List[float]): A list of query embeddings.
            top_k (int): The number of results to return.
            filters (Optional[Dict[str, Any]], optional): Filters to apply during search. Defaults to None.

        Returns:
            List[Document]: A list of matching documents.
        """
        converted_filters = _convert_filters(filters)

        result = self._get_result_to_documents(
            self.index.query(
                vector=query_embedding,
                top_k=top_k,
                query_filter=converted_filters,
                include_metadata=True,
                include_values=True,
            )
        )
        logger.debug(f"Raw responses: {result}")  # leaving for debugging

        return result

    def delete_documents(self, document_ids: Optional[List[str]] = None, delete_all: Optional[bool] = None) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.
        Fails with `MissingDocumentError` if no document with this id is present in the store.

        :param document_ids: the document_ids to delete.
        :param delete_all: delete all documents.
        """

        deletion_counter = 0
        if self.index.count_documents() > 0:
            if document_ids is not None:
                for batch in _batches(document_ids, MAX_BATCH_SIZE):
                    deletion_counter += self.index.delete(ids=batch)
            else:
                deletion_counter = self.index.delete(delete_all=delete_all)
            logger.info(f"{deletion_counter} documents deleted")

            if document_ids is not None and deletion_counter == 0:
                msg = f"Document {document_ids} does not exist"
                raise MissingDocumentError(msg)
        else:
            logger.info("No documents in document store")
