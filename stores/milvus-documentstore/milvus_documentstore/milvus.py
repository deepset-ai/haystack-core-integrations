import json
import logging
from typing import Any, Dict, Generator, List, Optional, Set, Union

import numpy as np

try:
    from pymilvus_simple.simple_api import SimpleAPI
except (ImportError, ModuleNotFoundError) as ie:
    from haystack.utils.import_utils import _optional_component_not_installed

    _optional_component_not_installed(__name__, "milvus2", ie)

from haystack.document_stores import BaseDocumentStore
from haystack.errors import DuplicateDocumentError
from haystack.schema import Document, FilterType, Label

from .filter_utils import LogicalFilterClause

ID_FIELD = "id"
EMPTY_FIELD = "empty"
VECTOR_FIELD = "embedding"
META_FIELD = "$meta"

logger = logging.getLogger(__name__)


class MilvusDocumentStore(BaseDocumentStore):
    """
    Usage:
    1. Start a Milvus service via docker (see https://milvus.io/docs/install_standalone-docker.md)
    2. Run pip install pymilvus-haystack
    3. Init a MilvusDocumentStore() in Haystack

    Overview:
    Milvus (https://milvus.io/) is a highly reliable, scalable Document Store specialized on storing and processing vectors.
    Therefore, it is particularly suited for Haystack users that work with dense retrieval methods (like DPR).

    In contrast to FAISS, Milvus ...
     - runs as a separate service (e.g. a Docker container) and can scale easily in a distributed environment
     - allows dynamic data management (i.e. you can insert/delete vectors without recreating the whole index)
     - encapsulates multiple ANN libraries (FAISS, ANNOY ...)

    This class uses Milvus for all vector related storage, processing and querying.
    The meta-data (e.g. for filtering) and the document text are also stored within Milvus
    """

    def __init__(
        self,
        uri: str = "http://localhost:19530/default",
        api_key: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        index: str = "document",
        embedding_dim: int = 768,
        similarity: str = "dot_product",
        index_params: Optional[Dict[str, Any]] = None,
        return_embedding: bool = False,
        progress_bar: bool = False,
        consistency_level: str = "Session",
        replicas: int = 1,
        recreate_index: bool = False,
    ):
        """
        :param uri: The connection URI to use for Milvus.
        :param api_key: Optional API key to use instead of user and password.
        :param user: Optional username to use to connect to Milvus instance.
        :param password: Optional password to use to connect to Milvus instance.
        :param index: The name of the collection to use for storing the data.
        :param embedding_dim: The embedding vector size. Default: 768.
        :param similarity: The similarity function used to compare document vectors. 'dot_product' is the default and recommended for DPR embeddings.
            'cosine' is recommended for Sentence Transformers, but is not directly supported by Milvus.
            However, you can normalize your embeddings and use `dot_product` to get the same results.
            See https://milvus.io/docs/metric.md.
        :param index_params: Configuration parameters for the chose index_type needed at indexing time.
            For example: {"nlist": 16384} as the number of cluster units to create for index_type IVF_FLAT.
            See https://milvus.io/docs/index.md
        :param return_embedding: To return document embedding.
        :param progress_bar: Whether to show a tqdm progress bar or not.
                             Can be helpful to disable in production deployments to keep the logs clean.
        :param consistency_level: Which consistency level to use for Milvus index: Strong, Session, Bounded, Evenetually.
            See https://milvus.io/docs/v2.1.x/tune_consistency.md.
        :param replicas: How many in memory replicas to use for index.
            See https://milvus.io/docs/replica.md.
        :param recreate_index: If set to True, an existing Milvus index will be deleted and a new one will be
            created using the config you are using for initialization. Be aware that all data in the old index will be
            lost if you choose to recreate the index. Be aware that both the document_index and the label_index will
            be recreated.
        """
        self.collection_name = index
        self.consistency_level = consistency_level
        self.index_params = index_params
        self.progress_bar = progress_bar
        self.replicas = replicas
        self.dimension = embedding_dim
        self.dummy_data = [0] * embedding_dim
        self.return_embedding = return_embedding

        if similarity == "cosine":
            self.metric_type = "IP"
            self.normalize = True
        elif similarity == "dot_product":
            self.metric_type = "IP"
        elif similarity in ("l2", "euclidean"):
            self.metric_type = "L2"
        else:
            raise ValueError(
                "The Milvus document store can currently only support dot_product, cosine and euclidean metrics. "
            )

        self.client = SimpleAPI(uri=uri, api_key=api_key, user=user, password=password)
        self._create_collection(index, recreate_index)

    def _create_collection(self, name, recreate_index=False):
        self.client.create_collection(
            collection_name=name,
            dimension=self.dimension,
            primary_field=ID_FIELD,
            primary_type="str",
            primary_auto_id=False,
            vector_field=VECTOR_FIELD,
            metric_type=self.metric_type,
            partition_field={"name": EMPTY_FIELD, "type": "int"},
            index=self.index_params,
            overwrite=recreate_index,
            consistency_level=self.consistency_level,
            replicas=self.replicas,
        )

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
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
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip: Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)

        :return: None
        """
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        index = index or self.collection_name
        if index not in self.client.list_collections():
            self._create_collection(index)
        docs = self._handle_duplicate_documents(
            documents=documents,
            index=index,
            duplicate_documents=duplicate_documents,
            headers=headers,
        )
        docs_modified = []
        for doc in docs:
            doc = doc.to_dict()
            if doc.get("embedding") is None:
                doc["embedding"] = self.dummy_data
                doc[EMPTY_FIELD] = 1
            else:
                doc[EMPTY_FIELD] = 0
            doc.pop("score", None)
            meta = doc.pop("meta", None)
            doc.update(meta)
            docs_modified.append(doc)
        self.client.insert(
            collection_name=index,
            data=docs_modified,
            batch_size=batch_size,
            progress_bar=self.progress_bar,
        )

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Get documents from the document store.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                            __Example__:

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
                            ```

        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: Not supported.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        """
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        if batch_size is not None:
            logger.warn(
                "The parameter batch_size is not supported, loading all results."
            )

        index = index or self.collection_name
        return_embedding = return_embedding or self.return_embedding

        filters = (
            LogicalFilterClause.parse(filters).convert_to_milvus() if filters else None
        )
        res = self.client.query(
            collection_name=index,
            filter_expression=filters,
            include_vectors=return_embedding,
            partition_keys=[0, 1],
        )
        res_docs = []
        for hit in res:
            hit.pop(EMPTY_FIELD)
            res_docs.append(Document.from_dict(hit))
        return res_docs

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Generator[Document, None, None]:
        """
        Get documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the search space to documents whose metadata fulfill certain
                        conditions.
                        Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
                        operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
                        `"$gte"`, `"$lt"`, `"$lte"`) or a metadata field name.
                        Logical operator keys take a dictionary of metadata field names and/or logical operators as
                        value. Metadata field names take a dictionary of comparison operators as value. Comparison
                        operator keys take a single value or (in case of `"$in"`) a list of values as value.
                        If no logical operator is provided, `"$and"` is used as default operation. If no comparison
                        operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
                        operation.

                        __Example__:

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
                        ```

        :param return_embedding: Whether to return the document embeddings.
        :param batch_size: Not supported.
        """
        for doc in self.get_all_documents(
            index, filters, return_embedding, batch_size, headers
        ):
            yield doc

    def get_all_labels(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Label]:
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        index = index or self.collection_name

        filters = (
            LogicalFilterClause.parse(filters).convert_to_milvus() if filters else None
        )
        res = self.client.query(
            collection_name=index, filter_expression=filters, partition_keys=[2]
        )
        res_labels = []
        for hit in res:
            hit.pop(EMPTY_FIELD, None)
            hit.pop("embedding", None)
            res_labels.append(Label.from_dict(hit))
        return res_labels

    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        if filters is not None:
            filters = (
                LogicalFilterClause.parse(filters).convert_to_milvus()
                if filters
                else None
            )

        index = index or self.collection_name
        if index not in self.client.list_collections():
            self._create_collection(index)
            return 0

        if only_documents_without_embedding:
            only_documents_without_embedding = [1]
        else:
            only_documents_without_embedding = [0, 1]

        return len(
            self.client.query(
                index,
                filter_expression=filters,
                output_fields=[ID_FIELD],
                partition_keys=only_documents_without_embedding,
            )
        )

    def get_embedding_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        if filters is not None:
            filters = (
                LogicalFilterClause.parse(filters).convert_to_milvus()
                if filters
                else None
            )

        index = index or self.collection_name
        if index not in self.client.list_collections():
            self._create_collection(index)
            return 0

        return len(
            self.client.query(
                self.collection_name,
                filter_expression=filters,
                output_fields=[ID_FIELD],
                partition_keys=[0],
            )
        )

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
    ) -> List[Document]:
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        if self.similarity == "cosine":
            self.normalize_embedding(query_emb)

        index = index or self.collection_name
        return_embedding = return_embedding or self.return_embedding

        if index not in self.client.list_collections():
            self._create_collection(index)
            return []

        if filters is not None:
            filters = (
                LogicalFilterClause.parse(filters).convert_to_milvus()
                if filters
                else None
            )

        res = self.client.search(
            collection_name=index,
            data=[query_emb],
            top_k=top_k,
            include_vectors=return_embedding,
            partition_keys=[0],
            filter_expression=filters,
        )
        res_docs = []
        for hit in res[0]:
            hit.pop(EMPTY_FIELD)
            doc = Document.from_dict(hit)
            doc.score = (
                hit.score
                if not scale_score
                else self.scale_to_unit_interval(hit.score, self.similarity)
            )
            res_docs.append(doc)

        return res_docs

    def get_label_count(
        self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> int:
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        index = index or self.collection_name
        if index not in self.client.list_collections():
            self._create_collection(index)
            return 0

        return len(
            self.client.query(
                index,
                output_fields=[ID_FIELD],
                partition_keys=[2],
            )
        )

    def write_labels(
        self,
        labels: Union[List[Label], List[dict]],
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        batch_size=100,
    ):
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        index = index or self.collection_name
        if index not in self.client.list_collections():
            self._create_collection(index)

        insert_labels = []

        for label in labels:
            if isinstance(label, Label):
                label = label.to_json()
                label = json.loads(label)
            label["embedding"] = self.dummy_data
            label[EMPTY_FIELD] = 2
            insert_labels.append(label)
        self.client.insert(
            collection_name=index,
            data=insert_labels,
            batch_size=batch_size,
            progress_bar=self.progress_bar,
        )

    def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        index = index or self.collection_name
        if index not in self.client.list_collections():
            self._create_collection(index)
            return

        ids = [] if ids is None else ids

        if len(ids) != 0 and filters is not None:
            filters = (
                LogicalFilterClause.parse(filters).convert_to_milvus()
                if filters
                else None
            )
            res = self.client.query(
                collection_name=index,
                filter_expression=filters,
                output_fields=[ID_FIELD],
                partition_keys=[0, 1],
            )
            filter_ids = [hit[ID_FIELD] for hit in res]
            ids = list(set(ids).intersection(set(filter_ids)))

        elif filters is not None and len(ids) == 0:
            filters = (
                LogicalFilterClause.parse(filters).convert_to_milvus()
                if filters
                else None
            )
            res = self.client.query(
                collection_name=index,
                filter_expression=filters,
                output_fields=[ID_FIELD],
                partition_keys=[0, 1],
            )
            ids = [hit[ID_FIELD] for hit in res]

        elif filters is None and len(ids) != 0:
            pass
        else:
            self.delete_all_documents(index=index)

        self.client.delete(
            collection_name=index,
            field_name=ID_FIELD,
            values=ids,
            partition_keys=[0, 1],
        )

    def delete_labels(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        index = index or self.collection_name
        if index not in self.client.list_collections():
            self._create_collection(index)
            return

        ids = [] if ids is None else ids

        if len(ids) != 0 and filters is not None:
            filters = (
                LogicalFilterClause.parse(filters).convert_to_milvus()
                if filters
                else None
            )
            res = self.client.query(
                collection_name=index,
                filter_expression=filters,
                output_fields=[ID_FIELD],
                partition_keys=[2],
            )
            filter_ids = [hit[ID_FIELD] for hit in res]
            ids = list(set(ids).intersection(set(filter_ids)))

        elif filters is not None and len(ids) == 0:
            filters = (
                LogicalFilterClause.parse(filters).convert_to_milvus()
                if filters
                else None
            )
            res = self.client.query(
                collection_name=index,
                filter_expression=filters,
                output_fields=[ID_FIELD],
                partition_keys=[2],
            )
            ids = [hit[ID_FIELD] for hit in res]

        elif filters is None and len(ids) != 0:
            pass
        else:
            self.delete_all_labels(index=index)

        self.client.delete(
            collection_name=index, field_name=ID_FIELD, values=ids, partition_keys=[2]
        )

    def delete_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        index = index or self.collection_name
        if index not in self.client.list_collections():
            self._create_collection(index)
            return
        if filters is not None:
            filters = (
                LogicalFilterClause.parse(filters).convert_to_milvus()
                if filters
                else None
            )
        else:
            filters = f'{ID_FIELD} >= ""'

        res = self.client.query(
            collection_name=index,
            filter_expression=filters,
            output_fields=[ID_FIELD],
            partition_keys=[0, 1],
        )

        ids = [hit[ID_FIELD] for hit in res]

        self.client.delete(
            collection_name=index,
            field_name=ID_FIELD,
            values=ids,
            partition_keys=[0, 1],
        )

    def delete_all_labels(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        index = index or self.collection_name
        if index not in self.client.list_collections():
            self._create_collection(index)
            return
        if filters is not None:
            filters = (
                LogicalFilterClause.parse(filters).convert_to_milvus()
                if filters
                else None
            )
        else:
            filters = f'{ID_FIELD} >= ""'

        res = self.client.query(
            collection_name=index,
            filter_expression=filters,
            output_fields=[ID_FIELD],
            partition_keys=[2],
        )

        ids = [hit[ID_FIELD] for hit in res]

        self.client.delete(
            collection_name=index, field_name=ID_FIELD, values=ids, partition_keys=[2]
        )

    def delete_index(self, index: Optional[str] = None):
        """
        Delete an existing index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        index = index or self.collection_name
        self.client.drop_collection(index)

    def _create_document_field_map(self) -> Dict:
        pass

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = 10_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        logger.warn("Batch size ignored for queries")

        index = index or self.collection_name
        if index not in self.client.list_collections():
            self._create_collection(index)
            return []
        res = self.client.fetch(
            collection_name=index,
            field_name=ID_FIELD,
            values=ids,
            partition_keys=[0, 1],
        )
        res_docs = []
        for hit in res:
            hit.pop(EMPTY_FIELD)
            res_docs.append(Document.from_dict(hit))
        return res_docs

    def get_document_by_id(
        self,
        id: str,
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Document]:
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        docs = self.get_documents_by_id(ids=[id], index=index, headers=headers)
        if len(docs) == 0:
            return None
        else:
            return docs[0]

    def update_document_meta(
        self, id: str, meta: Dict[str, Any], index: Optional[str] = None
    ):
        index = index or self.collection_name
        if index not in self.client.list_collections():
            self._create_collection(index)
            return
        res = self.client.fetch(
            collection_name=index,
            field_name=ID_FIELD,
            values=[id],
            partition_keys=[0, 1],
            include_vectors=True,
        )
        if len(res) != 0:
            res[0].update(meta)
            self.client.delete(
                collection_name=index,
                field_name=ID_FIELD,
                values=[id],
                partition_keys=[0, 1],
            )
            self.client.insert(collection_name=index, data=[res[0]])

    def _handle_duplicate_documents(
        self,
        documents: List[Document],
        index: Optional[str] = None,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Checks whether any of the passed documents is already existing in the chosen index and returns a list of
        documents that are not in the index yet.

        :param documents: A list of Haystack Document objects.
        :param index: name of the index
        :param duplicate_documents: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip (default option): Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :param headers: Custom HTTP headers to pass to document store client if supported (e.g. {'Authorization': 'Basic YWRtaW46cm9vdA=='} for basic authentication)
        :return: A list of Haystack Document objects.
        """
        if headers:
            raise NotImplementedError("MilvusDocumentStore does not support headers.")

        index = index or self.collection_name

        documents = self._drop_duplicate_documents(documents, index)
        documents_found = self.get_documents_by_id(
            ids=[doc.id for doc in documents], index=index, headers=headers
        )
        ids_exist_in_db: List[str] = [doc.id for doc in documents_found]

        if len(ids_exist_in_db) > 0 and duplicate_documents == "fail":
            raise DuplicateDocumentError(
                f"Document with ids '{', '.join(ids_exist_in_db)} already exists"
                f" in index = '{index}'."
            )

        if len(ids_exist_in_db) > 0 and duplicate_documents == "overwrite":
            self.delete_documents(ids=ids_exist_in_db, index=index)
            return documents

        documents = list(filter(lambda doc: doc.id not in ids_exist_in_db, documents))

        return documents

    def _drop_duplicate_documents(
        self, documents: List[Document], index: Optional[str] = None
    ) -> List[Document]:
        """
        Drop duplicates documents based on same hash ID

        :param documents: A list of Haystack Document objects.
        :param index: name of the index
        :return: A list of Haystack Document objects.
        """
        _hash_ids: Set = set([])
        _documents: List[Document] = []

        for document in documents:
            if isinstance(document, dict):
                document = Document.from_dict(document)
            if document.id in _hash_ids:
                logger.info(
                    "Duplicate Documents: Document with id '%s' already exists in collection '%s'",
                    document.id,
                    index or self.collection_name,
                )
                continue
            _documents.append(document)
            _hash_ids.add(document.id)

        return _documents
