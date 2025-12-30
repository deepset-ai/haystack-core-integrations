# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError, MissingDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace

from .astra_client import AstraClient, QueryResponse
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

    Example Usage:
    ```python
    from haystack_integrations.document_stores.astra import AstraDocumentStore

    document_store = AstraDocumentStore(
        api_endpoint=api_endpoint,
        token=token,
        collection_name=collection_name,
        duplicates_policy=DuplicatePolicy.SKIP,
        embedding_dim=384,
    )
    ```
    """

    def __init__(
        self,
        api_endpoint: Secret = Secret.from_env_var("ASTRA_DB_API_ENDPOINT"),  # noqa: B008
        token: Secret = Secret.from_env_var("ASTRA_DB_APPLICATION_TOKEN"),  # noqa: B008
        collection_name: str = "documents",
        embedding_dimension: int = 768,
        duplicates_policy: DuplicatePolicy = DuplicatePolicy.NONE,
        similarity: str = "cosine",
        namespace: Optional[str] = None,
    ):
        """
        The connection to Astra DB is established and managed through the JSON API.
        The required credentials (api endpoint and application token) can be generated
        through the UI by clicking and the connect tab, and then selecting JSON API and
        Generate Configuration.

        :param api_endpoint: the Astra DB API endpoint.
        :param token: the Astra DB application token.
        :param collection_name: the current collection in the keyspace in the current Astra DB.
        :param embedding_dimension: dimension of embedding vector.
        :param duplicates_policy: handle duplicate documents based on DuplicatePolicy parameter options.
              Parameter options : (`SKIP`, `OVERWRITE`, `FAIL`, `NONE`)
              - `DuplicatePolicy.NONE`: Default policy, If a Document with the same ID already exists,
                    it is skipped and not written.
              - `DuplicatePolicy.SKIP`: if a Document with the same ID already exists, it is skipped and not written.
              - `DuplicatePolicy.OVERWRITE`: if a Document with the same ID already exists, it is overwritten.
              - `DuplicatePolicy.FAIL`: if a Document with the same ID already exists, an error is raised.
        :param similarity: the similarity function used to compare document vectors.

        :raises ValueError: if the API endpoint or token is not set.
        """
        resolved_api_endpoint = api_endpoint.resolve_value()
        if resolved_api_endpoint is None:
            msg = (
                "AstraDocumentStore expects the API endpoint. "
                "Set the ASTRA_DB_API_ENDPOINT environment variable (recommended) or pass it explicitly."
            )
            raise ValueError(msg)
        self.resolved_api_endpoint = resolved_api_endpoint

        resolved_token = token.resolve_value()
        if resolved_token is None:
            msg = (
                "AstraDocumentStore expects an authentication token. "
                "Set the ASTRA_DB_APPLICATION_TOKEN environment variable (recommended) or pass it explicitly."
            )
            raise ValueError(msg)
        self.resolved_token = resolved_token

        self.api_endpoint = api_endpoint
        self.token = token
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.duplicates_policy = duplicates_policy
        self.similarity = similarity
        self.namespace = namespace
        self._index: Optional[AstraClient] = None

    @property
    def index(self) -> AstraClient:
        if self._index is None:
            self._index = AstraClient(
                self.resolved_api_endpoint,
                self.resolved_token,
                self.collection_name,
                self.embedding_dimension,
                self.similarity,
                self.namespace,
            )
        return self._index

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AstraDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_endpoint", "token"])
        return default_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """

        return default_to_dict(
            self,
            api_endpoint=self.api_endpoint.to_dict(),
            token=self.token.to_dict(),
            collection_name=self.collection_name,
            embedding_dimension=self.embedding_dimension,
            duplicates_policy=self.duplicates_policy.name,
            similarity=self.similarity,
            namespace=self.namespace,
        )

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Indexes documents for later queries.

        :param documents: a list of Haystack Document objects.
        :param policy: handle duplicate documents based on DuplicatePolicy parameter options.
            Parameter options : (`SKIP`, `OVERWRITE`, `FAIL`, `NONE`)
            - `DuplicatePolicy.NONE`: Default policy, If a Document with the same ID already exists,
                it is skipped and not written.
            - `DuplicatePolicy.SKIP`: If a Document with the same ID already exists,
                it is skipped and not written.
            - `DuplicatePolicy.OVERWRITE`: If a Document with the same ID already exists, it is overwritten.
            - `DuplicatePolicy.FAIL`: If a Document with the same ID already exists, an error is raised.
        :returns: number of documents written.
        :raises ValueError: if the documents are not of type Document or dict.
        :raises DuplicateDocumentError: if a document with the same ID already exists and policy is set to FAIL.
        :raises Exception: if the document ID is not a string or if `id` and `_id` are both present in the document.
        """
        if policy is None or policy == DuplicatePolicy.NONE:
            if self.duplicates_policy is not None and self.duplicates_policy != DuplicatePolicy.NONE:
                policy = self.duplicates_policy
            else:
                policy = DuplicatePolicy.SKIP

        batch_size = MAX_BATCH_SIZE

        def _convert_input_document(document: Union[dict, Document]) -> dict[str, Any]:
            if isinstance(document, Document):
                document_dict = document.to_dict(flatten=False)
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

            if embedding := document_dict.pop("embedding", []):
                document_dict["$vector"] = embedding

            if "sparse_embedding" in document_dict:
                sparse_embedding = document_dict.pop("sparse_embedding", None)
                if sparse_embedding:
                    logger.warning(
                        "Document {id} has the `sparse_embedding` field set,"
                        "but storing sparse embeddings in Astra is not currently supported."
                        "The `sparse_embedding` field will be ignored.",
                        id=document_dict["_id"],
                    )

            return document_dict

        documents_to_write = [_convert_input_document(doc) for doc in documents]

        duplicate_documents = []
        new_documents: list[dict] = []
        i = 0
        while i < len(documents_to_write):
            doc = documents_to_write[i]
            # check to see if this ID already exists in our new_documents array
            exists = [d for d in new_documents if d["_id"] == doc["_id"]]
            # check to see if this ID is already in the DB
            response = self.index.find_documents({"filter": {"_id": doc["_id"]}})
            if response or exists:
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
                    inserted_ids = self.index.insert(batch)
                    insertion_counter += len(inserted_ids)
                    logger.info(f"write_documents inserted documents with id {inserted_ids}")
            else:
                logger.warning("No documents written. Argument policy set to SKIP")

        elif policy == DuplicatePolicy.OVERWRITE:
            if len(new_documents) > 0:
                for batch in _batches(new_documents, batch_size):
                    inserted_ids = self.index.insert(batch)
                    insertion_counter += len(inserted_ids)
                    logger.info(f"write_documents inserted documents with id {inserted_ids}")
            else:
                logger.warning("No documents written. Argument policy set to OVERWRITE")

            if len(duplicate_documents) > 0:
                updated_ids = []
                for duplicate_doc in duplicate_documents:
                    updated = self.index.update_document(duplicate_doc, "_id")
                    if updated:
                        updated_ids.append(duplicate_doc["_id"])
                insertion_counter = insertion_counter + len(updated_ids)
                logger.info(f"write_documents updated documents with id {updated_ids}")
            else:
                logger.info("No documents updated. Argument policy set to OVERWRITE")

        elif policy == DuplicatePolicy.FAIL:
            if len(new_documents) > 0:
                for batch in _batches(new_documents, batch_size):
                    inserted_ids = self.index.insert(batch)
                    insertion_counter = insertion_counter + len(inserted_ids)
                    logger.info(f"write_documents inserted documents with id {inserted_ids}")
            else:
                logger.warning("No documents written. Argument policy set to FAIL")

        return insertion_counter

    def count_documents(self) -> int:
        """
        Counts the number of documents in the document store.

        :returns: the number of documents in the document store.
        """
        return self.index.count_documents()

    def filter_documents(self, filters: Optional[dict[str, Any]] = None) -> list[Document]:
        """
        Returns at most 1000 documents that match the filter.

        :param filters: filters to apply.
        :returns: matching documents.
        :raises AstraDocumentStoreFilterError: if the filter is invalid or not supported by this class.
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
    def _get_result_to_documents(results: QueryResponse) -> list[Document]:
        documents = []
        for match in results.matches:
            metadata = match.metadata
            blob = metadata.pop("blob", None) if metadata else None
            meta = metadata.pop("meta", {}) if metadata else {}
            document = Document(
                content=match.text,
                id=match.document_id,
                embedding=match.values,
                blob=blob,
                meta=meta,
                score=match.score,
            )
            documents.append(document)
        return documents

    def get_documents_by_id(self, ids: list[str]) -> list[Document]:
        """
        Gets documents by their IDs.

        :param ids: the IDs of the documents to retrieve.
        :returns: the matching documents.
        """
        results = self.index.get_documents(ids=ids)
        ret = self._get_result_to_documents(results)
        return ret

    def get_document_by_id(self, document_id: str) -> Document:
        """
        Gets a document by its ID.

        :param document_id: the ID to filter by
        :returns: the found document
        :raises MissingDocumentError: if the document is not found
        """
        document = self.index.get_documents(ids=[document_id])
        ret = self._get_result_to_documents(document)
        if not ret:
            msg = f"Document {document_id} does not exist"
            raise MissingDocumentError(msg)
        return ret[0]

    def search(
        self, query_embedding: list[float], top_k: int, filters: Optional[dict[str, Any]] = None
    ) -> list[Document]:
        """
        Perform a search for a list of queries.

        :param query_embedding: a list of query embeddings.
        :param top_k: the number of results to return.
        :param filters: filters to apply during search.
        :returns: matching documents.
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

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes documents from the document store.

        :param document_ids: IDs of the documents to delete.
        :raises MissingDocumentError: if no document was deleted but document IDs were provided.
        """
        if self.index.find_one_document({"filter": {}}) is not None:
            deletion_counter = 0
            if document_ids is not None:
                for batch in _batches(document_ids, MAX_BATCH_SIZE):
                    deletion_counter += self.index.delete(ids=batch)
            logger.info(f"{deletion_counter} documents deleted")

            if document_ids is not None and deletion_counter == 0:
                msg = f"Document {document_ids} does not exist"
                raise MissingDocumentError(msg)
        else:
            logger.info("No documents in document store")

    def delete_all_documents(self) -> None:
        """
        Deletes all documents from the document store.
        """

        try:
            deletion_counter = self.index.delete_all_documents()
        except Exception as e:
            msg = f"Failed to delete all documents from Astra: {e!s}"
            raise DocumentStoreError(msg) from e

        if deletion_counter == -1:
            logger.info("All documents deleted")
        else:
            logger.error("Could not delete all documents")

    def delete_by_filter(self, filters: dict[str, Any]) -> int:
        """
        Deletes documents that match the provided filters.

        :param filters: The filters to apply to find documents to delete.
        :returns: The number of documents deleted.
        :raises AstraDocumentStoreFilterError: if the filter is invalid or not supported.
        """
        if not isinstance(filters, dict):
            msg = "Filters must be a dictionary"
            raise AstraDocumentStoreFilterError(msg)

        if "id" in filters:
            filters["_id"] = filters.pop("id")

        converted_filters = _convert_filters(filters)
        deletion_count = self.index.delete(filters=converted_filters)

        logger.info(f"{deletion_count} documents deleted by filter")
        return deletion_count

    def update_by_filter(self, filters: dict[str, Any], meta: dict[str, Any]) -> int:
        """
        Updates documents that match the provided filters with the given metadata.

        :param filters: The filters to apply to find documents to update.
        :param meta: The metadata fields to update. This will be merged with existing metadata.

        :returns:
            The number of documents updated.

        :raises:
            AstraDocumentStoreFilterError: if the filter is invalid or not supported.
        """
        if not isinstance(filters, dict):
            msg = "Filters must be a dictionary"
            raise AstraDocumentStoreFilterError(msg)

        if not isinstance(meta, dict):
            msg = "Meta must be a dictionary"
            raise AstraDocumentStoreFilterError(msg)

        if "id" in filters:
            filters["_id"] = filters.pop("id")

        converted_filters = _convert_filters(filters)

        # use dot notation to update nested fields in the meta-object - ensures fields are created if they don't exist
        update_fields = {f"meta.{key}": value for key, value in meta.items()}
        update_operation = {"$set": update_fields}
        update_count = self.index.update(filters=converted_filters, update=update_operation)  # type: ignore

        logger.info(f"{update_count} documents updated by filter")

        return update_count
