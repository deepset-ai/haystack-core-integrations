import logging
from typing import Any, Dict, List, Optional

from azure.cosmos import CosmosClient
from azure.cosmos.container import ContainerProxy
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

logger = logging.getLogger(__name__)


class AzureCosmosNoSqlDocumentStore:
    """
    AzureCosmosNoSqlDocumentStore is a DocumentStore implementation that uses [Azure CosmosDB NoSql]
    (https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search) service.

    To connect to Azure CosmosDB NoSql, you need to provide a cosmosClient. After providing the connection
    string, you need to provide `database_name`, `container_name`, `vector_embedding_policy`, `indexing_policy`,
    `cosmos_container_properties`.

    Please refer to README.md file for detailed setup.
    """
    def __init__(
        self,
        cosmos_client: CosmosClient,
        database_name: str,
        container_name: str,
        vector_embedding_policy: dict[str, Any] = None,
        indexing_policy: dict[str, Any] = None,
        cosmos_container_properties: dict[str, Any] = None,
    ):
        """
        Creates a new AzureCosmosNoSqlDocumentStore instance.

        :param cosmos_client: Azure CosmosClient for NoSql account.
        :param database_name: Name of the database to use.
        :param container_name: Name of the container to use.
        :param vector_embedding_policy: Dictionary of vector embeddings to use.
        :param indexing_policy: Dictionary of indexing policies to use.
        :param cosmos_container_properties: Dictionary of cosmos container properties to use.

        :raises ValueError: if the vectorIndexes or vectorEmbeddings are null in the indexing_policy
        or the vector_embedding_policy.
        """
        if indexing_policy["vectorIndexes"] is None or len(indexing_policy["vectorIndexes"]) == 0:
            raise ValueError("vectorIndexes cannot be null or empty in the indexing_policy.")
        if vector_embedding_policy is None or len(vector_embedding_policy["vectorEmbeddings"]) == 0:
            raise ValueError("vectorEmbeddings cannot be null or empty in the vector_embedding_policy.")

        self.cosmos_client = cosmos_client
        self.database_name = database_name
        self.container_name = container_name
        self.vector_embedding_policy = vector_embedding_policy
        self.indexing_policy = indexing_policy
        self.cosmos_container_properties = cosmos_container_properties
        self._container = Optional[ContainerProxy] = None

    @property
    def container(self) -> ContainerProxy:
        # Create the database if it already doesn't exist
        database = self.cosmos_client.create_database_if_not_exists(id=self.database_name)
        if database.get_container_client(self.container_name).read() is not None:
            # Create the collection if it already doesn't exist
            self._container = database.create_container(
                id=self.container_name,
                partition_key=self.cosmos_container_properties["partition_key"],
                indexing_policy=self.indexing_policy,
                vector_embedding_policy=self.vector_embedding_policy,
            )
        return self._container

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.

        :returns: The number of documents in the document store.
        """
        return len(list(self.container.read_all_items()))

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.

        :param document_ids: the document ids to delete
        """
        if not document_ids:
            return
        for document_id in document_ids:
            self.container.delete_item(document_id)

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int:
        """
        Writes documents into the MongoDB Atlas collection.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises DuplicateDocumentError: If a document with the same ID already exists in the document store
             and the policy is set to DuplicatePolicy.FAIL (or not specified).
        :raises ValueError: If the documents are not of type Document.
        :returns: The number of documents written to the document store.
        """

        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        for doc in documents:
            self.container.create_item(doc.to_dict())
        return len(documents)

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the Haystack [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

        :param filters: The filters to apply. It returns only the documents that match the filters.
        :returns: A list of Documents that match the given filters.
        """
        query = self._filtered_query(filters)
        items = list(self.container.query_items(query, enable_cross_partition_query=True))
        return [Document.from_dict(doc) for doc in items]

    def _embedding_retrieval(
            self,
            query_embedding: List[float],
            filters: Optional[Dict[str, Any]] = None,
            top_k: int = 10,
    ) -> List[Document]:
        """
        Find the documents that are most similar to the provided `query_embedding` by using a vector similarity metric.

        :param query_embedding: Embedding of the query
        :param filters: Optional filters.
        :param top_k: How many documents to return.
        :returns: A list of Documents that are most similar to the given `query_embedding`
        :raises ValueError: If `query_embedding` is empty.
        :raises DocumentStoreError: If the retrieval of documents from MongoDB Atlas fails.
        """
        embedding_key = self.vector_embedding_policy["vectorEmbeddings"][0]["path"][1:]

        query = "SELECT "
        # If limit_offset_clause is not specified, add TOP clause
        if filters is None or filters.get("limit_offset_clause") is None:
            query += "TOP @limit "
        query += (
            "c.id, c.@embeddingKey, c.content, c.meta, c.score, "
            "VectorDistance(c.@embeddingKey, @embeddings) AS SimilarityScore FROM c"
        )
        # Add where_clause if specified
        if filters is not None and filters.get("where_clause") is not None:
            query += " {}".format(filters["where_clause"])
        query += " ORDER BY VectorDistance(c.@embeddingKey, @embeddings)"
        # Add limit_offset_clause if specified
        if filters is not None and filters.get("limit_offset_clause") is not None:
            query += " {}".format(filters["limit_offset_clause"])
        parameters = [
            {"name": "@limit", "value": top_k},
            {"name": "@embeddingKey", "value": embedding_key},
            {"name": "@embeddings", "value": query_embedding},
        ]

        items = list(self.container.query_items(query, parameters=parameters, enable_cross_partition_query=True))
        nearest_results = [self._cosmos_doc_to_haystack_doc(item) for item in items]
        return nearest_results

    def _filtered_query(self, filters: Dict[str, Any]) -> str:
        if filters is None:
            return "SELECT * FROM c"
        query = "SELECT "

        if filters["top"] is not None:
            query += filters["top"]
        query += "* FROM c"
        if filters["where"] is not None:
            query += filters["where"]
        if filters["order_by"] is not None:
            query += filters["order_by"]
        if filters["limit_offset"] is not None and filters["top"] is None:
            query += filters["limit_offset"]
        return query

    def _cosmos_doc_to_haystack_doc(self, cosmos_doc: Dict[str, Any]) -> Document:
        """
        Converts the dictionary coming out of CosmosDB NoSql into a Haystack document

        :param cosmos_doc: A dictionary representing a document as stored in CosmosDB
        :returns: A Haystack Document object
        """
        cosmos_doc.pop("id", None)
        return Document.from_dict(cosmos_doc)


