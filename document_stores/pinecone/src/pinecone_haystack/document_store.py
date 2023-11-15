# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import logging
import operator
from functools import reduce
from itertools import islice
from typing import Any, Dict, Generator, List, Literal, Optional, Set, Union

import numpy as np
from tqdm import tqdm
import pinecone

from haystack.preview.dataclasses import Document
from haystack.preview.document_stores.decorator import document_store
from haystack.preview.document_stores.errors import (
    DuplicateDocumentError,
    MissingDocumentError,
)
from haystack.preview.document_stores.protocols import DuplicatePolicy

from pinecone_haystack.errors import (
    PineconeDocumentStoreError,
    PineconeDocumentStoreFilterError,
)
from pinecone_haystack.filter_utils import LogicalFilterClause


logger = logging.getLogger(__name__)


TYPE_METADATA_FIELD = "doc_type"
DOCUMENT_WITH_EMBEDDING = "vector"
DOCUMENT_WITHOUT_EMBEDDING = "no-vector"
LABEL = "label"

AND_OPERATOR = "$and"
IN_OPERATOR = "$in"
EQ_OPERATOR = "$eq"

DocTypeMetadata = Literal["vector", "no-vector", "label"]


def _sanitize_index(index: Optional[str]) -> Optional[str]:
    if index:
        return index.replace("_", "-").lower()
    return None


def _get_by_path(root, items):
    """Access a nested object in root by item sequence."""
    return reduce(operator.getitem, items, root)


def _set_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    _get_by_path(root, items[:-1])[items[-1]] = value


@document_store
class PineconeDocumentStore:
    """
    It implements the Pinecone vector database ([https://www.pinecone.io](https://www.pinecone.io))
    to perform similarity search on vectors. In order to use this document store, you need an API key that you can
    obtain by creating an account on the [Pinecone website](https://www.pinecone.io).

    This is a hosted document store,
    this means that your vectors will not be stored locally but in the cloud. This means that the similarity
    search will be run on the cloud as well.
    """

    top_k_limit = 10_000
    top_k_limit_vectors = 1_000

    def __init__(
        self,
        api_key: str,
        environment: str = "us-west1-gcp",
        pinecone_index: Optional["pinecone.Index"] = None,
        embedding_dim: int = 768,
        batch_size: int = 100,
        return_embedding: bool = False,
        index: str = "document",
        similarity: str = "cosine",
        replicas: int = 1,
        shards: int = 1,
        namespace: Optional[str] = None,
        embedding_field: str = "embedding",
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        recreate_index: bool = False,
        metadata_config: Optional[Dict] = None,
        validate_index_sync: bool = True,
    ):
        """
        :param api_key: Pinecone vector database API key ([https://app.pinecone.io](https://app.pinecone.io)).
        :param environment: Pinecone cloud environment uses `"us-west1-gcp"` by default. Other GCP and AWS
            regions are supported, contact Pinecone [here](https://www.pinecone.io/contact/) if required.
        :param pinecone_index: pinecone-client Index object, an index will be initialized or loaded if not specified.
        :param embedding_dim: The embedding vector size.
        :param batch_size: The batch size to be used when writing documents to the document store.
        :param return_embedding: Whether to return document embeddings.
        :param index: Name of index in document store to use.
        :param similarity: The similarity function used to compare document vectors. `"cosine"` is the default
            and is recommended if you are using a Sentence-Transformer model. `"dot_product"` is more performant
            with DPR embeddings.
            In both cases, the returned values in Document.score are normalized to be in range [0,1]:
                - For `"dot_product"`: `expit(np.asarray(raw_score / 100))`
                - For `"cosine"`: `(raw_score + 1) / 2`
        :param replicas: The number of replicas. Replicas duplicate the index. They provide higher availability and
            throughput.
        :param shards: The number of shards to be used in the index. We recommend to use 1 shard per 1GB of data.
        :param namespace: Optional namespace. If not specified, None is default.
        :param embedding_field: Name of field containing an embedding vector.
        :param progress_bar: Whether to show a tqdm progress bar or not.
            Can be helpful to disable in production deployments to keep the logs clean.
        :param duplicate_documents: Handle duplicate documents based on parameter options.\
            Parameter options:
                - `"skip"`: Ignore the duplicate documents.
                - `"overwrite"`: Update any existing documents with the same ID when adding documents.
                - `"fail"`: An error is raised if the document ID of the document being added already exists.
        :param recreate_index: If set to True, an existing Pinecone index will be deleted and a new one will be
            created using the config you are using for initialization. Be aware that all data in the old index will be
            lost if you choose to recreate the index. Be aware that both the document_index and the label_index will
            be recreated.
        :param metadata_config: Which metadata fields should be indexed, part of the
            [selective metadata filtering](https://www.pinecone.io/docs/manage-indexes/#selective-metadata-indexing) feature.
            Should be in the format `{"indexed": ["metadata-field-1", "metadata-field-2", "metadata-field-n"]}`. By default,
            no fields are indexed.
        """

        if metadata_config is None:
            metadata_config = {"indexed": []}
        # Connect to Pinecone server using python client binding
        if not api_key:
            raise PineconeDocumentStoreError(
                "Pinecone requires an API key, please provide one. https://app.pinecone.io"
            )

        pinecone.init(api_key=api_key, environment=environment)
        self._api_key = api_key

        # Format similarity string
        self._set_similarity_metric(similarity)

        self.similarity = similarity
        self.index: str = self._index(index)
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents

        # Pinecone index params
        self.replicas = replicas
        self.shards = shards
        self.namespace = namespace

        # Add necessary metadata fields to metadata_config
        fields = ["label-id", "query", TYPE_METADATA_FIELD]
        metadata_config["indexed"] += fields
        self.metadata_config = metadata_config

        # Initialize dictionary of index connections
        self.pinecone_indexes: Dict[str, pinecone.Index] = {}
        self.return_embedding = return_embedding
        self.embedding_field = embedding_field

        # Initialize dictionary to store temporary set of document IDs
        self.all_ids: dict = {}

        # Dummy query to be used during searches
        self.dummy_query = [0.0] * self.embedding_dim

        if pinecone_index:
            if not isinstance(pinecone_index, pinecone.Index):
                raise PineconeDocumentStoreError(
                    f"The parameter `pinecone_index` needs to be a "
                    f"`pinecone.Index` object. You provided an object of "
                    f"type `{type(pinecone_index)}`."
                )
            self.pinecone_indexes[self.index] = pinecone_index
        else:
            self.pinecone_indexes[self.index] = self._create_index(
                embedding_dim=self.embedding_dim,
                index=self.index,
                metric_type=self.metric_type,
                replicas=self.replicas,
                shards=self.shards,
                recreate_index=recreate_index,
                metadata_config=self.metadata_config,
            )

        super().__init__()

    def _index(self, index) -> str:
        index = _sanitize_index(index) or self.index
        return index

    def _create_index(
        self,
        embedding_dim: int,
        index: Optional[str] = None,
        metric_type: Optional[str] = "cosine",
        replicas: Optional[int] = 1,
        shards: Optional[int] = 1,
        recreate_index: bool = False,
        metadata_config: Optional[Dict] = None,
    ) -> "pinecone.Index":
        """
        Create a new index for storing documents in case an index with the name
        doesn't exist already.
        """
        if metadata_config is None:
            metadata_config = {"indexed": []}

        if recreate_index:
            self.delete_index(index)

        # Skip if already exists
        if index in self.pinecone_indexes:
            index_connection = self.pinecone_indexes[index]
        else:
            # Search pinecone hosted indexes and create an index if it does not exist
            if index not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index,
                    dimension=embedding_dim,
                    metric=metric_type,
                    replicas=replicas,
                    shards=shards,
                    metadata_config=metadata_config,
                )
            index_connection = pinecone.Index(index)

        # return index connection
        return index_connection

    def get_index_stats(self):
        stats = self.pinecone_indexes[self.index]
        self.index_stats = stats
        # Get index statistics
        dims = stats["dimension"]
        count = stats["namespaces"][""]["vector_count"] if stats["namespaces"].get("") else 0
        logger.info(
            "Index statistics: name: %s embedding dimensions: %s, record count: %s",
            self.index,
            dims,
            count,
        )

        return stats, dims, count

    def _index_connection_exists(self, index: str, create: bool = False) -> Optional["pinecone.Index"]:
        """
        Check if the index connection exists. If specified, create an index if it does not exist yet.

        :param index: Index name.
        :param create: Indicates if an index needs to be created or not. If set to `True`, create an index
            and return connection to it, otherwise raise `PineconeDocumentStoreError` error.
        :raises PineconeDocumentStoreError: Exception trigger when index connection not found.
        """
        if index not in self.pinecone_indexes:
            if create:
                return self._create_index(
                    embedding_dim=self.embedding_dim,
                    index=index,
                    metric_type=self.metric_type,
                    replicas=self.replicas,
                    shards=self.shards,
                    recreate_index=False,
                    metadata_config=self.metadata_config,
                )
            raise PineconeDocumentStoreError(
                f"Index named '{index}' does not exist. Try reinitializing PineconeDocumentStore() and running "
                f"'update_embeddings()' to create and populate an index."
            )
        return None

    def _set_similarity_metric(self, similarity: str):
        """
        Set vector similarity metric.
        """
        if similarity == "cosine":
            self.metric_type = similarity
        elif similarity == "dot_product":
            self.metric_type = "dotproduct"
        elif similarity in ["l2", "euclidean"]:
            self.metric_type = "euclidean"
        else:
            raise ValueError(
                "The Pinecone document store can currently only support dot_product, cosine and euclidean metrics. "
                "Please set similarity to one of the above."
            )

    def _add_local_ids(self, index: str, ids: List[str]):
        """
        Add all document IDs to the set of all IDs.
        """
        if index not in self.all_ids:
            self.all_ids[index] = set()
        self.all_ids[index] = self.all_ids[index].union(set(ids))

    def _add_type_metadata_filter(
        self, filters: Dict[str, Any], type_value: Optional[DocTypeMetadata]
    ) -> Dict[str, Any]:
        """
        Add new filter for `doc_type` metadata field.
        """
        if type_value:
            new_type_filter = {TYPE_METADATA_FIELD: {EQ_OPERATOR: type_value}}
            if AND_OPERATOR not in filters and TYPE_METADATA_FIELD not in filters:
                # extend filters with new `doc_type` filter and add $and operator
                filters.update(new_type_filter)
                all_filters = filters
                return {AND_OPERATOR: all_filters}

            filters_content = filters[AND_OPERATOR] if AND_OPERATOR in filters else filters
            if TYPE_METADATA_FIELD in filters_content:  # type: ignore
                current_type_filter = filters_content[TYPE_METADATA_FIELD]  # type: ignore
                type_values = {type_value}
                if isinstance(current_type_filter, str):
                    type_values.add(current_type_filter)  # type: ignore
                elif isinstance(current_type_filter, dict):
                    if EQ_OPERATOR in current_type_filter:
                        # current `doc_type` filter has single value
                        type_values.add(current_type_filter[EQ_OPERATOR])
                    else:
                        # current `doc_type` filter has multiple values
                        type_values.update(set(current_type_filter[IN_OPERATOR]))
                new_type_filter = {TYPE_METADATA_FIELD: {IN_OPERATOR: list(type_values)}}  # type: ignore
            filters_content.update(new_type_filter)  # type: ignore

        return filters

    def _get_default_type_metadata(self, index: Optional[str], namespace: Optional[str] = None) -> str:
        """
        Get default value for `doc_type` metadata filed. If there is at least one embedding, default value
        will be `vector`, otherwise it will be `no-vector`.
        """
        if self.get_embedding_count(index=index, namespace=namespace) > 0:
            return DOCUMENT_WITH_EMBEDDING
        return DOCUMENT_WITHOUT_EMBEDDING

    def _get_vector_count(
        self,
        index: str,
        filters: Optional[Dict[str, Any]],
        namespace: Optional[str],
    ) -> int:
        res = self.pinecone_indexes[index].query(
            self.dummy_query,
            top_k=self.top_k_limit,
            include_values=False,
            include_metadata=False,
            filter=filters,
            namespace=namespace,
        )
        return len(res["matches"])

    def get_document_count(
        self,
        filters: Dict[str, Any] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
        namespace: Optional[str] = None,
        type_metadata: Optional[DocTypeMetadata] = None,
    ) -> int:
        """
        Return the count of documents in the document store.

        :param filters: Optional filters to narrow down the documents which will be counted.
            Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
            operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`,
            `"$gte"`, `"$lt"`, `"$lte"`), or a metadata field name.
            Logical operator keys take a dictionary of metadata field names or logical operators as
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
        :param index: Optional index name to use for the query. If not provided, the default index name is used.
        :param only_documents_without_embedding: If set to `True`, only documents without embeddings are counted.
        :param headers: PineconeDocumentStore does not support headers.
        :param namespace: Optional namespace to count documents from. If not specified, None is default.
        :param type_metadata: Optional value for `doc_type` metadata to reference documents that need to be counted.
            Parameter options:
                - `"vector"`: Documents with embedding.
                - `"no-vector"`: Documents without embedding (dummy embedding only).
                - `"label"`: Labels.
        """
        if headers:
            raise NotImplementedError("PineconeDocumentStore does not support headers.")

        index = self._index(index)
        self._index_connection_exists(index)

        filters = filters or {}
        if not type_metadata:
            # add filter for `doc_type` metadata related to documents without embeddings
            filters = self._add_type_metadata_filter(filters, type_value=DOCUMENT_WITHOUT_EMBEDDING)  # type: ignore
            if not only_documents_without_embedding:
                # add filter for `doc_type` metadata related to documents with embeddings
                filters = self._add_type_metadata_filter(filters, type_value=DOCUMENT_WITH_EMBEDDING)  # type: ignore
        else:
            # if value for `doc_type` metadata is specified, add filter with given value
            filters = self._add_type_metadata_filter(filters, type_value=type_metadata)

        pinecone_syntax_filter = LogicalFilterClause.parse(filters).convert_to_pinecone() if filters else None
        return self._get_vector_count(index, filters=pinecone_syntax_filter, namespace=namespace)

    def get_embedding_count(
        self,
        filters: Optional[Dict[str, Any]] = None,
        index: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> int:
        """
        Return the count of embeddings in the document store.

        :param index: Optional index name to retrieve all documents from.
        :param filters: Filters are not supported for `get_embedding_count` in Pinecone.
        :param namespace: Optional namespace to count embeddings from. If not specified, None is default.
        """
        if filters:
            raise NotImplementedError("Filters are not supported for get_embedding_count in PineconeDocumentStore")

        index = self._index(index)
        self._index_connection_exists(index)

        pinecone_filters = self._meta_for_pinecone({TYPE_METADATA_FIELD: DOCUMENT_WITH_EMBEDDING})
        return self._get_vector_count(index, filters=pinecone_filters, namespace=namespace)

    def _meta_for_pinecone(self, meta: Dict[str, Any], parent_key: str = "", labels: bool = False) -> Dict[str, Any]:
        """
        Converts the meta dictionary to a format that can be stored in Pinecone.
        :param meta: Metadata dictionary to be converted.
        :param parent_key: Optional, used for recursive calls to keep track of parent keys, for example:
            ```
            {"parent1": {"parent2": {"child": "value"}}}
            ```
            On the second recursive call, parent_key would be "parent1", and the final key would be "parent1.parent2.child".
        :param labels: Optional, used to indicate whether the metadata is being stored as a label or not. If True the
            the flattening of dictionaries is not required.
        """
        items: list = []
        if labels:
            # Replace any None values with empty strings
            for key, value in meta.items():
                if value is None:
                    meta[key] = ""
        else:
            # Explode dict of dicts into single flattened dict
            for key, value in meta.items():
                # Replace any None values with empty strings
                if value is None:
                    value = ""
                if key == "_split_overlap":
                    value = json.dumps(value)
                # format key
                new_key = f"{parent_key}.{key}" if parent_key else key
                # if value is dict, expand
                if isinstance(value, dict):
                    items.extend(self._meta_for_pinecone(value, parent_key=new_key).items())
                else:
                    items.append((new_key, value))
            # Create new flattened dictionary
            meta = dict(items)
        return meta

    def _pinecone_meta_format(self, meta: Dict[str, Any], labels: bool = False) -> Dict[str, Any]:
        """
        Converts the meta extracted from Pinecone into a better format for Python.
        :param meta: Metadata dictionary to be converted.
        :param labels: Optional, used to indicate whether the metadata is being stored as a label or not. If True the
            the flattening of dictionaries is not required.
        """
        new_meta: Dict[str, Any] = {}

        if labels:
            # Replace any empty strings with None values
            for key, value in meta.items():
                if value == "":
                    meta[key] = None
            return meta
        else:
            for key, value in meta.items():
                # Replace any empty strings with None values
                if value == "":
                    value = None
                if "." in key:
                    # We must split into nested dictionary
                    keys = key.split(".")
                    # Iterate through each dictionary level
                    for i in range(len(keys)):
                        path = keys[: i + 1]
                        # Check if path exists
                        try:
                            _get_by_path(new_meta, path)
                        except KeyError:
                            # Create path
                            if i == len(keys) - 1:
                                _set_by_path(new_meta, path, value)
                            else:
                                _set_by_path(new_meta, path, {})
                else:
                    new_meta[key] = value
            return new_meta

    def _validate_index_sync(self, index: Optional[str] = None):
        """
        This check ensures the correct number of documents with embeddings and embeddings are found in the
        Pinecone database.
        """
        if self.get_document_count(
            index=index, type_metadata=DOCUMENT_WITH_EMBEDDING  # type: ignore
        ) != self.get_embedding_count(index=index):
            raise PineconeDocumentStoreError(
                f"The number of documents present in Pinecone ({self.get_document_count(index=index)}) "
                "does not match the number of embeddings in Pinecone "
                f" ({self.get_embedding_count(index=index)}). This can happen if a document store "
                "instance is deleted during write operations. Call "
                "the `update_documents` method to fix it."
            )

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        count = self.index_stats["namespaces"][""]["vector_count"] if self.index_stats["namespaces"].get("") else 0
        return count

    def write_documents(
        self,
        documents: List[Document],
        policy: DuplicatePolicy = "fail",
    ) -> None:
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
        if not isinstance(documents, list):
            msg = "Documents must be a list"
            raise ValueError(msg)

        index = self._index(self.index)
        index_connection = self._index_connection_exists(index, create=True)
        if index_connection:
            self.pinecone_indexes[index] = index_connection

        duplicate_documents = policy or self.duplicate_documents
        policy_options = ["skip", "overwrite", "fail"]
        assert (
            duplicate_documents in policy_options
        ), f"duplicate_documents parameter must be {', '.join(policy_options)}"

        add_vectors = documents[0].embedding is not None
        type_metadata = DOCUMENT_WITH_EMBEDDING if add_vectors else DOCUMENT_WITHOUT_EMBEDDING

        if not add_vectors:
            # To store documents in Pinecone, we use dummy embeddings (to be replaced with real embeddings later)
            embeddings_to_index = np.zeros((self.batch_size, self.embedding_dim), dtype="float32")
            # Convert embeddings to list objects
            embeddings = [embed.tolist() if embed is not None else None for embed in embeddings_to_index]

        with tqdm(
            total=len(documents),
            disable=not self.progress_bar,
            position=0,
            desc="Writing Documents",
        ) as progress_bar:
            for i in range(0, len(documents), self.batch_size):
                document_batch = documents[i : i + self.batch_size]
                ids = [doc.id for doc in document_batch]
                # If duplicate_documents set to `skip` or `fail`, we need to check for existing documents
                if duplicate_documents in ["skip", "fail"]:
                    existing_documents = self.get_documents_by_id(
                        ids=ids,
                        index=index,
                        namespace=self.namespace,
                        include_type_metadata=True,
                    )
                    # First check for documents in current batch that exist in the index
                    if existing_documents:
                        if duplicate_documents == "skip":
                            # If we should skip existing documents, we drop the ids that already exist
                            skip_ids = [doc.id for doc in existing_documents]
                            # We need to drop the affected document objects from the batch
                            document_batch = [doc for doc in document_batch if doc.id not in skip_ids]
                            # Now rebuild the ID list
                            ids = [doc.id for doc in document_batch]
                            progress_bar.update(len(skip_ids))
                        elif duplicate_documents == "fail":
                            # Otherwise, we raise an error
                            raise DuplicateDocumentError(
                                f"Document ID {existing_documents[0].id} already exists in index {index}"
                            )
                    # Now check for duplicate documents within the batch itself
                    if len(ids) != len(set(ids)):
                        if duplicate_documents == "skip":
                            # We just keep the first instance of each duplicate document
                            ids = []
                            temp_document_batch = []
                            for doc in document_batch:
                                if doc.id not in ids:
                                    ids.append(doc.id)
                                    temp_document_batch.append(doc)
                            document_batch = temp_document_batch
                        elif duplicate_documents == "fail":
                            # Otherwise, we raise an error
                            raise DuplicateDocumentError(f"Duplicate document IDs found in batch: {ids}")
                metadata = [
                    self._meta_for_pinecone(
                        {
                            TYPE_METADATA_FIELD: type_metadata,  # add `doc_type` in metadata
                            "text": doc.content,
                            "content_type": doc.meta,
                        }
                    )
                    for doc in documents[i : i + self.batch_size]
                ]
                if add_vectors:
                    embeddings = [doc.embedding for doc in documents[i : i + self.batch_size]]
                    embeddings_to_index = np.array(embeddings, dtype="float32")

                    # Convert embeddings to list objects
                    embeddings = [embed.tolist() if embed is not None else None for embed in embeddings_to_index]
                data_to_write_to_pinecone = zip(ids, embeddings, metadata)
                # Metadata fields and embeddings are stored in Pinecone
                self.pinecone_indexes[index].upsert(vectors=data_to_write_to_pinecone, namespace=self.namespace)
                # Add IDs to ID list
                self._add_local_ids(index, ids)
                progress_bar.update(self.batch_size)
        progress_bar.close()

    def _limit_check(self, top_k: int, include_values: Optional[bool] = None):
        """
        Confirms the top_k value does not exceed Pinecone vector database limits.
        """
        if include_values:
            if top_k > self.top_k_limit_vectors:
                raise PineconeDocumentStoreError(
                    f"PineconeDocumentStore allows requests of no more than {self.top_k_limit_vectors} records "
                    f"when returning embedding values. This request is attempting to return {top_k} records."
                )
        else:
            if top_k > self.top_k_limit:
                raise PineconeDocumentStoreError(
                    f"PineconeDocumentStore allows requests of no more than {self.top_k_limit} records. "
                    f"This request is attempting to return {top_k} records."
                )

    def query_by_embedding(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = True,
        return_embedding: Optional[bool] = None,
    ) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_embedding` by using a vector similarity metric.

        :param query_embedding: Embedding of the query.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :param scale_score: Whether to scale the scores of the retrieved documents or not.
        :param return_embedding: Whether to return the embedding of the retrieved Documents.
        :return: The retrieved documents.
        """
        if return_embedding is None:
            return_embedding = self.return_embedding

        self._limit_check(top_k, include_values=return_embedding)

        index = self._index(self.index)
        self._index_connection_exists(index)

        type_metadata = DOCUMENT_WITH_EMBEDDING  # type: ignore

        filters = filters or {}
        filters = self._add_type_metadata_filter(filters, type_metadata)

        pinecone_syntax_filter = LogicalFilterClause.parse(filters).convert_to_pinecone() if filters else None

        res = self.pinecone_indexes[index].query(
            query_embedding,
            namespace=self.namespace,
            top_k=top_k,
            include_values=return_embedding,
            include_metadata=True,
            filter=pinecone_syntax_filter,
        )

        score_matrix = []
        vector_id_matrix = []
        meta_matrix = []
        embedding_matrix = []
        for match in res["matches"]:
            score_matrix.append(match["score"])
            vector_id_matrix.append(match["id"])
            meta_matrix.append(match["metadata"])
            if return_embedding:
                embedding_matrix.append(match["values"])
        if return_embedding:
            values = embedding_matrix
        else:
            values = None
        documents = self._get_documents_by_meta(
            vector_id_matrix,
            meta_matrix,
            values=values,
            index=index,
            return_embedding=return_embedding,
        )

        # assign query score to each document
        scores_for_vector_ids: Dict[str, float] = {str(v_id): s for v_id, s in zip(vector_id_matrix, score_matrix)}
        return_documents = []
        for doc in documents:
            score = scores_for_vector_ids[doc.id]
            if scale_score:
                if self.similarity == "cosine":
                    score = (score + 1) / 2
                else:
                    score = float(1 / (1 + np.exp(-score / 100)))
            doc.score = score
            return_document = copy.copy(doc)
            return_documents.append(return_document)

        return return_documents

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
        docs = self.query_by_embedding(
            query_embedding=self.dummy_query,
            filters=filters,
            top_k=10,
            scale_score=True,
            return_embedding=True,
        )

        return docs

    def _attach_embedding_to_document(self, document: Document, index: str):
        """
        Fetches the Document's embedding from the specified Pinecone index and attaches it to the Document's
        embedding field.
        """
        result = self.pinecone_indexes[index].fetch(ids=[document.id])
        if result["vectors"].get(document.id, False):
            embedding = result["vectors"][document.id].get("values", None)
            document.embedding = np.asarray(embedding, dtype=np.float32)

    def _get_documents_by_meta(
        self,
        ids: List[str],
        metadata: List[dict],
        values: Optional[List[List[float]]] = None,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
    ) -> List[Document]:
        if return_embedding is None:
            return_embedding = self.return_embedding

        index = self._index(index)

        # extract ID, content, and metadata to create Documents
        documents = []
        for _id, meta in zip(ids, metadata):
            content = meta.pop("content")
            content_type = meta.pop("content_type")
            if "_split_overlap" in meta:
                meta["_split_overlap"] = json.loads(meta["_split_overlap"])
            doc = Document(id=_id, content=content, content_type=content_type, meta=meta)
            documents.append(doc)
        if return_embedding:
            if values is None:
                # If no embedding values are provided, we must request the embeddings from Pinecone
                for doc in documents:
                    self._attach_embedding_to_document(document=doc, index=index)
            else:
                # If embedding values are given, we just add
                for doc, embedding in zip(documents, values):
                    doc.embedding = np.asarray(embedding, dtype=np.float32)

        return documents

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = 100,
        return_embedding: Optional[bool] = None,
        namespace: Optional[str] = None,
        include_type_metadata: Optional[bool] = False,
    ) -> List[Document]:
        """
        Retrieves all documents in the index using their IDs.

        :param ids: List of IDs to retrieve.
        :param index: Optional index name to retrieve all documents from.
        :param batch_size: Number of documents to retrieve at a time. When working with large number of documents,
            batching can help reduce memory footprint.
        :param headers: Pinecone does not support headers.
        :param return_embedding: Optional flag to return the embedding of the document.
        :param namespace: Optional namespace to retrieve document from. If not specified, None is default.
        :param include_type_metadata: Indicates if `doc_type` value will be included in document metadata or not.
            If not specified, `doc_type` field will be dropped from document metadata.
        """

        if return_embedding is None:
            return_embedding = self.return_embedding

        index = self._index(index)
        self._index_connection_exists(index)

        documents = []
        for i in range(0, len(ids), batch_size):
            i_end = min(len(ids), i + batch_size)
            id_batch = ids[i:i_end]
            result = self.pinecone_indexes[index].fetch(ids=id_batch, namespace=namespace)

            vector_id_matrix = []
            meta_matrix = []
            embedding_matrix = []
            for _id in result["vectors"]:
                vector_id_matrix.append(_id)
                metadata = result["vectors"][_id]["metadata"]
                if not include_type_metadata and TYPE_METADATA_FIELD in metadata:
                    metadata.pop(TYPE_METADATA_FIELD)
                meta_matrix.append(self._pinecone_meta_format(metadata))
                if return_embedding:
                    embedding_matrix.append(result["vectors"][_id]["values"])
            if return_embedding:
                values = embedding_matrix
            else:
                values = None
            document_batch = self._get_documents_by_meta(
                vector_id_matrix,
                meta_matrix,
                values=values,
                index=index,
                return_embedding=return_embedding,
            )
            documents.extend(document_batch)

        return documents

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.
        Fails with `MissingDocumentError` if no document with this id is present in the store.

        :param document_ids: the document_ids to delete
        """
        for doc_id in document_ids:
            msg = f"ID '{doc_id}' not found, cannot delete it."
            document_ids.remove(doc_id)
            raise MissingDocumentError(msg)

        index = self._index(self.index)
        self._index_connection_exists(index)

        if index not in self.all_ids:
            self.all_ids[index] = set()
        if document_ids is None:
            # If no IDs we delete everything
            self.pinecone_indexes[index].delete(delete_all=True, namespace=self.namespace)
            id_values = list(self.all_ids[index])
        else:
            id_values = document_ids
            self.pinecone_indexes[index].delete(ids=document_ids, namespace=self.namespace)

        # Remove deleted ids from all_ids
        self.all_ids[index] = self.all_ids[index].difference(set(id_values))

    def delete_index(self, index: Optional[str]):
        """
        Delete an existing index. The index including all data will be removed.

        :param index: The name of the index to delete.
        :return: None
        """
        index = self._index(index)

        if index in pinecone.list_indexes():
            pinecone.delete_index(index)
            logger.info("Index '%s' deleted.", index)
        if index in self.pinecone_indexes:
            del self.pinecone_indexes[index]
        if index in self.all_ids:
            self.all_ids[index] = set()
