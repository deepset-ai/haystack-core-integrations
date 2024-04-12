import inspect
import logging
from itertools import islice
from typing import Any, ClassVar, Dict, Generator, List, Optional, Set, Union

import numpy as np
import qdrant_client
from grpc import RpcError
from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.filters import convert as convert_legacy_filters
from qdrant_client import grpc
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from tqdm import tqdm

from .converters import (
    DENSE_VECTORS_NAME,
    SPARSE_VECTORS_NAME,
    convert_haystack_documents_to_qdrant_points,
    convert_id,
    convert_qdrant_point_to_haystack_document,
)
from .filters import convert_filters_to_qdrant

logger = logging.getLogger(__name__)


class QdrantStoreError(DocumentStoreError):
    pass


FilterType = Dict[str, Union[Dict[str, Any], List[Any], str, int, float, bool]]


def get_batches_from_generator(iterable, n):
    """
    Batch elements of an iterable into fixed-length chunks or blocks.
    """
    it = iter(iterable)
    x = tuple(islice(it, n))
    while x:
        yield x
        x = tuple(islice(it, n))


class QdrantDocumentStore:
    SIMILARITY: ClassVar[Dict[str, str]] = {
        "cosine": rest.Distance.COSINE,
        "dot_product": rest.Distance.DOT,
        "l2": rest.Distance.EUCLID,
    }

    def __init__(
        self,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[Secret] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        index: str = "Document",
        embedding_dim: int = 768,
        on_disk: bool = False,
        content_field: str = "content",
        name_field: str = "name",
        embedding_field: str = "embedding",
        use_sparse_embeddings: bool = False,
        similarity: str = "cosine",
        return_embedding: bool = False,
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        recreate_index: bool = False,
        shard_number: Optional[int] = None,
        replication_factor: Optional[int] = None,
        write_consistency_factor: Optional[int] = None,
        on_disk_payload: Optional[bool] = None,
        hnsw_config: Optional[dict] = None,
        optimizers_config: Optional[dict] = None,
        wal_config: Optional[dict] = None,
        quantization_config: Optional[dict] = None,
        init_from: Optional[dict] = None,
        wait_result_from_api: bool = True,
        metadata: Optional[dict] = None,
        write_batch_size: int = 100,
        scroll_size: int = 10_000,
        payload_fields_to_index: Optional[List[dict]] = None,
    ):
        super().__init__()

        metadata = metadata or {}
        self.client = qdrant_client.QdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key.resolve_value() if api_key else None,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            metadata=metadata,
        )

        # Store the Qdrant client specific attributes
        self.location = location
        self.url = url
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.https = https
        self.api_key = api_key
        self.prefix = prefix
        self.timeout = timeout
        self.host = host
        self.path = path
        self.metadata = metadata
        self.api_key = api_key

        # Store the Qdrant collection specific attributes
        self.shard_number = shard_number
        self.replication_factor = replication_factor
        self.write_consistency_factor = write_consistency_factor
        self.on_disk_payload = on_disk_payload
        self.hnsw_config = hnsw_config
        self.optimizers_config = optimizers_config
        self.wal_config = wal_config
        self.quantization_config = quantization_config
        self.init_from = init_from
        self.wait_result_from_api = wait_result_from_api
        self.recreate_index = recreate_index
        self.payload_fields_to_index = payload_fields_to_index
        self.use_sparse_embeddings = use_sparse_embeddings

        # Make sure the collection is properly set up
        self._set_up_collection(
            index, embedding_dim, recreate_index, similarity, use_sparse_embeddings, on_disk, payload_fields_to_index
        )

        self.embedding_dim = embedding_dim
        self.on_disk = on_disk
        self.content_field = content_field
        self.name_field = name_field
        self.embedding_field = embedding_field
        self.similarity = similarity
        self.index = index
        self.return_embedding = return_embedding
        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents
        self.write_batch_size = write_batch_size
        self.scroll_size = scroll_size

    def count_documents(self) -> int:
        try:
            response = self.client.count(
                collection_name=self.index,
            )
            return response.count
        except (UnexpectedResponse, ValueError):
            # Qdrant local raises ValueError if the collection is not found, but
            # with the remote server UnexpectedResponse is raised. Until that's unified,
            # we need to catch both.
            return 0

    def filter_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        if filters and not isinstance(filters, dict):
            msg = "Filter must be a dictionary"
            raise ValueError(msg)

        if filters and "operator" not in filters:
            filters = convert_legacy_filters(filters)
        return list(
            self.get_documents_generator(
                filters,
            )
        )

    def write_documents(
        self,
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL,
    ):
        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"DocumentStore.write_documents() expects a list of Documents but got an element of {type(doc)}."
                raise ValueError(msg)
        self._set_up_collection(self.index, self.embedding_dim, False, self.similarity, self.use_sparse_embeddings)

        if len(documents) == 0:
            logger.warning("Calling QdrantDocumentStore.write_documents() with empty list")
            return

        document_objects = self._handle_duplicate_documents(
            documents=documents,
            index=self.index,
            policy=policy,
        )

        batched_documents = get_batches_from_generator(document_objects, self.write_batch_size)
        with tqdm(total=len(document_objects), disable=not self.progress_bar) as progress_bar:
            for document_batch in batched_documents:
                batch = convert_haystack_documents_to_qdrant_points(
                    document_batch,
                    embedding_field=self.embedding_field,
                    use_sparse_embeddings=self.use_sparse_embeddings,
                )

                self.client.upsert(
                    collection_name=self.index,
                    points=batch,
                    wait=self.wait_result_from_api,
                )

                progress_bar.update(self.write_batch_size)
        return len(document_objects)

    def delete_documents(self, ids: List[str]):
        ids = [convert_id(_id) for _id in ids]
        try:
            self.client.delete(
                collection_name=self.index,
                points_selector=ids,
                wait=self.wait_result_from_api,
            )
        except KeyError:
            logger.warning(
                "Called QdrantDocumentStore.delete_documents() on a non-existing ID",
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QdrantDocumentStore":
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        params = inspect.signature(self.__init__).parameters  # type: ignore
        # All the __init__ params must be set as attributes
        # Set as init_parms without default values
        init_params = {k: getattr(self, k) for k in params}
        init_params["api_key"] = self.api_key.to_dict() if self.api_key else None
        return default_to_dict(
            self,
            **init_params,
        )

    def get_documents_generator(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Generator[Document, None, None]:
        index = self.index
        qdrant_filters = convert_filters_to_qdrant(filters)

        next_offset = None
        stop_scrolling = False
        while not stop_scrolling:
            records, next_offset = self.client.scroll(
                collection_name=index,
                scroll_filter=qdrant_filters,
                limit=self.scroll_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            stop_scrolling = next_offset is None or (
                isinstance(next_offset, grpc.PointId) and next_offset.num == 0 and next_offset.uuid == ""
            )

            for record in records:
                yield convert_qdrant_point_to_haystack_document(
                    record, use_sparse_embeddings=self.use_sparse_embeddings
                )

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
    ) -> List[Document]:
        index = index or self.index

        documents: List[Document] = []

        ids = [convert_id(_id) for _id in ids]
        records = self.client.retrieve(
            collection_name=index,
            ids=ids,
            with_payload=True,
            with_vectors=True,
        )

        for record in records:
            documents.append(
                convert_qdrant_point_to_haystack_document(record, use_sparse_embeddings=self.use_sparse_embeddings)
            )
        return documents

    def query_by_sparse(
        self,
        query_sparse_embedding: SparseEmbedding,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = True,
        return_embedding: bool = False,
    ) -> List[Document]:
        if not self.use_sparse_embeddings:
            message = (
                "You are trying to query using sparse embeddings, but the Document Store "
                "was initialized with `use_sparse_embeddings=False`. "
            )
            raise QdrantStoreError(message)

        qdrant_filters = convert_filters_to_qdrant(filters)
        query_indices = query_sparse_embedding.indices
        query_values = query_sparse_embedding.values
        points = self.client.search(
            collection_name=self.index,
            query_vector=rest.NamedSparseVector(
                name=SPARSE_VECTORS_NAME,
                vector=rest.SparseVector(
                    indices=query_indices,
                    values=query_values,
                ),
            ),
            query_filter=qdrant_filters,
            limit=top_k,
            with_vectors=return_embedding,
        )
        results = [
            convert_qdrant_point_to_haystack_document(point, use_sparse_embeddings=self.use_sparse_embeddings)
            for point in points
        ]
        if scale_score:
            for document in results:
                score = document.score
                score = float(1 / (1 + np.exp(-score / 100)))
                document.score = score
        return results

    def query_by_embedding(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = True,
        return_embedding: bool = False,
    ) -> List[Document]:
        qdrant_filters = convert_filters_to_qdrant(filters)

        points = self.client.search(
            collection_name=self.index,
            query_vector=rest.NamedVector(
                name=DENSE_VECTORS_NAME if self.use_sparse_embeddings else "",
                vector=query_embedding,
            ),
            query_filter=qdrant_filters,
            limit=top_k,
            with_vectors=return_embedding,
        )
        results = [
            convert_qdrant_point_to_haystack_document(point, use_sparse_embeddings=self.use_sparse_embeddings)
            for point in points
        ]
        if scale_score:
            for document in results:
                score = document.score
                if self.similarity == "cosine":
                    score = (score + 1) / 2
                else:
                    score = float(1 / (1 + np.exp(-score / 100)))
                document.score = score
        return results

    def _get_distance(self, similarity: str) -> rest.Distance:
        try:
            return self.SIMILARITY[similarity]
        except KeyError as ke:
            msg = (
                f"Provided similarity '{similarity}' is not supported by Qdrant "
                f"document store. Please choose one of the options: "
                f"{', '.join(self.SIMILARITY.keys())}"
            )
            raise QdrantStoreError(msg) from ke

    def _create_payload_index(self, collection_name: str, payload_fields_to_index: Optional[List[dict]] = None):
        """
        Create payload index for the collection if payload_fields_to_index is provided
        See: https://qdrant.tech/documentation/concepts/indexing/#payload-index
        """
        if payload_fields_to_index is not None:
            for payload_index in payload_fields_to_index:
                self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=payload_index["field_name"],
                    field_schema=payload_index["field_schema"],
                )

    def _set_up_collection(
        self,
        collection_name: str,
        embedding_dim: int,
        recreate_collection: bool,
        similarity: str,
        use_sparse_embeddings: bool,
        on_disk: bool = False,
        payload_fields_to_index: Optional[List[dict]] = None,
    ):
        distance = self._get_distance(similarity)

        if recreate_collection:
            # There is no need to verify the current configuration of that
            # collection. It might be just recreated again.
            self._recreate_collection(collection_name, distance, embedding_dim, on_disk, use_sparse_embeddings)
            # Create Payload index if payload_fields_to_index is provided
            self._create_payload_index(collection_name, payload_fields_to_index)
            return

        try:
            # Check if the collection already exists and validate its
            # current configuration with the parameters.
            collection_info = self.client.get_collection(collection_name)
        except (UnexpectedResponse, RpcError, ValueError):
            # That indicates the collection does not exist, so it can be
            # safely created with any configuration.
            #
            # Qdrant local raises ValueError if the collection is not found, but
            # with the remote server UnexpectedResponse / RpcError is raised.
            # Until that's unified, we need to catch both.
            self._recreate_collection(collection_name, distance, embedding_dim, on_disk, use_sparse_embeddings)
            # Create Payload index if payload_fields_to_index is provided
            self._create_payload_index(collection_name, payload_fields_to_index)
            return

        has_named_vectors = (
            isinstance(collection_info.config.params.vectors, dict)
            and DENSE_VECTORS_NAME in collection_info.config.params.vectors
        )

        if self.use_sparse_embeddings and not has_named_vectors:
            msg = (
                f"Collection '{collection_name}' already exists in Qdrant, "
                f"but it has been originally created without sparse embedding vectors. "
                f"If you want to use that collection, you can set `use_sparse_embeddings=False`. "
                f"To use sparse embeddings, you need to recreate the collection or migrate the existing one."
            )
            raise QdrantStoreError(msg)

        elif not self.use_sparse_embeddings and has_named_vectors:
            msg = (
                f"Collection '{collection_name}' already exists in Qdrant, "
                f"but it has been originally created with sparse embedding vectors."
                f"If you want to use that collection, please set `use_sparse_embeddings=True`."
            )
            raise QdrantStoreError(msg)

        if self.use_sparse_embeddings:
            current_distance = collection_info.config.params.vectors[DENSE_VECTORS_NAME].distance
            current_vector_size = collection_info.config.params.vectors[DENSE_VECTORS_NAME].size
        else:
            current_distance = collection_info.config.params.vectors.distance
            current_vector_size = collection_info.config.params.vectors.size

        if current_distance != distance:
            msg = (
                f"Collection '{collection_name}' already exists in Qdrant, "
                f"but it is configured with a similarity '{current_distance.name}'. "
                f"If you want to use that collection, but with a different "
                f"similarity, please set `recreate_collection=True` argument."
            )
            raise ValueError(msg)

        if current_vector_size != embedding_dim:
            msg = (
                f"Collection '{collection_name}' already exists in Qdrant, "
                f"but it is configured with a vector size '{current_vector_size}'. "
                f"If you want to use that collection, but with a different "
                f"vector size, please set `recreate_collection=True` argument."
            )
            raise ValueError(msg)

    def _recreate_collection(
        self,
        collection_name: str,
        distance,
        embedding_dim: int,
        on_disk: bool,
        use_sparse_embeddings: bool,
    ):
        # dense vectors configuration
        vectors_config = rest.VectorParams(size=embedding_dim, on_disk=on_disk, distance=distance)

        if use_sparse_embeddings:
            # in this case, we need to define named vectors
            vectors_config = {DENSE_VECTORS_NAME: vectors_config}

            sparse_vectors_config = {
                SPARSE_VECTORS_NAME: rest.SparseVectorParams(
                    index=rest.SparseIndexParams(
                        on_disk=on_disk,
                    )
                ),
            }

        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config if use_sparse_embeddings else None,
            shard_number=self.shard_number,
            replication_factor=self.replication_factor,
            write_consistency_factor=self.write_consistency_factor,
            on_disk_payload=self.on_disk_payload,
            hnsw_config=self.hnsw_config,
            optimizers_config=self.optimizers_config,
            wal_config=self.wal_config,
            quantization_config=self.quantization_config,
            init_from=self.init_from,
        )

    def _handle_duplicate_documents(
        self,
        documents: List[Document],
        index: Optional[str] = None,
        policy: DuplicatePolicy = None,
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
        :returns: A list of Haystack Document objects.
        """

        index = index or self.index
        if policy in (DuplicatePolicy.SKIP, DuplicatePolicy.FAIL):
            documents = self._drop_duplicate_documents(documents, index)
            documents_found = self.get_documents_by_id(ids=[doc.id for doc in documents], index=index)
            ids_exist_in_db: List[str] = [doc.id for doc in documents_found]

            if len(ids_exist_in_db) > 0 and policy == DuplicatePolicy.FAIL:
                msg = f"Document with ids '{', '.join(ids_exist_in_db)} already exists in index = '{index}'."
                raise DuplicateDocumentError(msg)

            documents = list(filter(lambda doc: doc.id not in ids_exist_in_db, documents))

        return documents

    def _drop_duplicate_documents(self, documents: List[Document], index: Optional[str] = None) -> List[Document]:
        """
        Drop duplicates documents based on same hash ID

        :param documents: A list of Haystack Document objects.
        :param index: name of the index
        :returns: A list of Haystack Document objects.
        """
        _hash_ids: Set = set()
        _documents: List[Document] = []

        for document in documents:
            if document.id in _hash_ids:
                logger.info(
                    "Duplicate Documents: Document with id '%s' already exists in index '%s'",
                    document.id,
                    index or self.index,
                )
                continue
            _documents.append(document)
            _hash_ids.add(document.id)

        return _documents
