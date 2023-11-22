# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Optional
from pydantic import validate_arguments

from haystack.preview.dataclasses import Document
from haystack.preview.document_stores.decorator import document_store
from haystack.preview.document_stores.errors import DuplicateDocumentError, MissingDocumentError
from haystack.preview.document_stores.protocols import DuplicatePolicy

from astra_client import AstraClient

from errors import AstraDocumentStoreFilterError

import json
import requests
from batching import get_batches_from_generator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@document_store
class AstraDocumentStore:
    """
        An AstraDocumentStore document store for Haystack.
    """

    @validate_arguments
    def __init__(
        self,
        astra_id: str,
        astra_region: str,
        astra_application_token: str,
        astra_keyspace: str,
        astra_collection: str,
        embedding_dim: int,
        similarity: str = "cosine",
    ):
        """
        The connection to Astra DB is established and managed through the JSON API.
        The required credentials (databse ID, region, and application token) can be generated
        through the UI by clicking and the connect tab, and then selecting JSON API and
        Generate Configuration.

        :param astra_id: id of the Astra DB instance.
        :param astra_region: Region of cloud servers (can be found when generating the token).
        :param astra_application_token: the connection token for Astra.
        :param astra_keyspace: The keyspace for the current Astra DB.
        :param astra_collection: The current collection in the keyspace in the current Astra DB.
        :param embedding_dim: Dimension of embedding vector.
        :param similarity: The similarity function used to compare document vectors.
        :param duplicate_documents: Handle duplicate documents based on parameter options.\
            Parameter options:
                - `"skip"`: Ignore the duplicate documents.
                - `"overwrite"`: Update any existing documents with the same ID when adding documents.
                - `"fail"`: An error is raised if the document ID of the document being added already exists.
        """

        self.astra_id = astra_id
        self.astra_region = astra_region
        self.astra_application_token = astra_application_token
        self.astra_keyspace = astra_keyspace
        self.astra_collection = astra_collection
        self.embedding_dim = embedding_dim
        self.similarity = similarity

        self.index = AstraClient(
            astra_id = self.astra_id,
            astra_region = self.astra_region,
            astra_application_token = self.astra_application_token,
            keyspace_name = self.astra_keyspace,
            collection_name = self.astra_collection,
            embedding_dim = self.embedding_dim,
            similarity_function = self.similarity
        )
        self.batch_size = 2


    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 20,
        policy: DuplicatePolicy = DuplicatePolicy.FAIL
    ):
        """
        Indexes documents for later queries.

        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
        :param index: Optional name of index where the documents shall be written to.
                      If None, the DocumentStore's default index (self.index) will be used.
        :param batch_size: Number of documents that are passed to bulk function at a time.
        :param policy:  Handle duplicate documents based on DuplicatePolicy parameter options.
                        Parameter options : ( 'skip','overwrite','fail')
                        skip: Ignore the duplicates documents
                        overwrite: Update any existing documents with the same ID when adding documents.
                        fail: an error is raised if the document ID of the document being added already
                        exists.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

        :return: None
        """

        if index == None:
            index = self.index 

        if batch_size > 20:
            logger.warning(f"batch_size set to {batch_size}, but maximum batch_size for Astra when using the JSON API is 20. batch_size set to 20.")
            batch_size = 20

        def _convert_input_document(document: Union[dict, Document]):
            if not isinstance(document, dict):
                data = asdict(document)
            meta = data.pop("meta")
            document_dict = {**data, **meta}
            document_dict["_id"] = document_dict.pop("id")
            if "embedding" in document_dict:
                document_dict["$vector"] = list(document_dict.pop("embedding"))
            else:
                document_dict["$vector"] = [0]*self.embedding_dim

            return document_dict

        documents_to_write = [_convert_input_document(doc) for doc in documents]

        duplicate_documents = []
        i = 0
        while i < len(documents_to_write):
            doc = documents_to_write[i]
            id_exists = self.index.find_document(find_key="_id", find_value=doc["_id"])["exists"]
            if id_exists: 
                duplicate_documents.append(doc)
                del documents_to_write[i]
                i = i - 1
            i = i + 1

        def _batches(inputlist, batchsize):
            print(batchsize)
            l = len(inputlist)
            for ndx in range(0, l, batchsize):
                yield inputlist[ndx:min(ndx + batchsize, l)]

        if policy == DuplicatePolicy.SKIP:
            if len(documents_to_write) > 0:
                for batch in _batches(documents_to_write, batch_size):
                    inserted_ids = index.insert(batch, "_id")
                    logger.info(f"write_documents inserted documents with id {inserted_ids}")
            else:
                logger.warning("No new documents to write to astra. No documents written. Argument policy set to SKIP")
        
        elif policy == DuplicatePolicy.OVERWRITE:
            if len(documents_to_write) > 0:
                for batch in _batches(documents_to_write, batch_size):
                    inserted_ids = index.insert(batch, "_id")
                    logger.info(f"write_documents inserted documents with id {inserted_ids}")
            
            if len(duplicate_documents) > 0:
                updated_ids = []
                for duplicate_doc in duplicate_documents:
                    updated = index.update_document(duplicate_doc, "_id")
                    if updated:
                        updated_ids.append(duplicate_doc["_id"])
                logger.info(f"write_documents updated documents with id {updated_ids}")
        
        elif policy == DuplicatePolicy.FAIL:
            raise Exception(f"write documents called with duplicate ids {[x['_id'] for x in documents_to_write]}, but argument policy set to FAIL")


    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        count = requests.request(
            "POST",
            self.index.request_url,
            headers=self.index.request_header,
            data=json.dumps({"countDocuments":{}}),
        ).json()["status"]["count"]
        return count

        return self._index.describe_index_stats()["total_document_count"]

    # def count_vectors(self) -> int:
    #     """
    #     Returns how many vectors are present in the document store.
    #     """
    #     return self._index.describe_index_stats()["total_vector_count"]

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

        filter_string = self._convert_filters(filters)
        results = self._index.query(filter=filter_string, top_k=1000, include_values=True, include_metadata=True)
        documents = self._get_documents(results)
        return documents

    def _get_result_to_documents(self, results, return_embedding) -> List[Document]:
        documents = []
        for res in results:
            id = res.pop("_id")
            metadata = res.pop("$vector")
            val = res
            if return_embedding:
                document = Document(
                    id=id,
                    text=val,
                    metadata=metadata,
                    mime_type="",
                    score=1,
                )
            else:
                document = Document(
                    id=id,
                    text=val,
                    mime_type="",
                    score=1,
                )
            documents.append(document)
        return documents

    # def get_documents(self, ids: List[str]) -> List[Document]:
    #     query = {"find": {"filter": {"_id": ""}}}
    #     query["find"]["filter"] = {"_id": {"$in":ids}}
    #     documents = requests.request(
    #         "POST",
    #         self.index.request_url,
    #         headers=self.index.request_header,
    #         data=json.dumps(query),
    #     ).json()["data"]["documents"]
    #     return documents

    def get_documents(self, ids: List[str]) -> List[Document]:
        documents = []
        document_batch = []
        for id_batch in get_batches_from_generator(ids, self.batch_size):
            query = {"find": {"filter": {"_id": ""}}}
            query["find"]["filter"] = {"_id": {"$in":id_batch}}
            response = requests.request(
                "POST",
                self.index.request_url,
                headers=self.index.request_header,
                data=json.dumps(query),
            ).json()["data"]["documents"]
            document_batch.append(response)
        for docs in document_batch:
            for doc in docs:
                documents.append(doc)
        return documents

    def get_documents_by_id(self, ids: List[str], return_embedding: Optional[bool] = None) -> List[Document]:
        """
            Returns documents with given ids.
        """
        results = self.get_documents(ids=ids)
        ret = self._get_result_to_documents(results, return_embedding)
        return ret

    def get_document_by_id(self, id: str, return_embedding: Optional[bool] = None) -> Document:
        """
            Returns documents with given ids.
        """
        document = self.get_documents(ids=[id])
        ret = self._get_result_to_documents(document, return_embedding)
        return ret[0]

    # def search(
    #         self, queries: List[Union[str, Dict[str, float]]], top_k: int, filters: Optional[Dict[str, Any]] = None
    # ) -> List[List[Document]]:
    #     """Perform a search for a list of queries.
    #
    #     Args:
    #         queries (List[Union[str, Dict[str, float]]]): A list of queries.
    #         top_k (int): The number of results to return.
    #         filters (Optional[Dict[str, Any]], optional): Filters to apply during search. Defaults to None.
    #
    #     Returns:
    #         List[List[Document]]: A list of matching documents for each query.
    #     """
    #     results = []
    #     for query in queries:
    #         result = self._index.search(q=query, limit=top_k, filter_string=self._convert_filters(filters))
    #         results.append(result)
    #
    #     return self._query_result_to_documents(results)

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.
        Fails with `MissingDocumentError` if no document with this id is present in the store.

        :param document_ids: the document_ids to delete
        """
        self._index.delete(ids=document_ids)
