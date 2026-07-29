import logging
from typing import Any

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from upstash_vector import Index

from .filters import _normalize_filters

logger = logging.getLogger(__name__)

TOP_K_LIMIT = 10_000


class UpstashDocumentStore:
    """
    A Document Store using Upstash Vector as the backend.
    """

    def __init__(
        self,
        url: Secret = Secret.from_env_var("UPSTASH_VECTOR_REST_URL"),
        token: Secret = Secret.from_env_var("UPSTASH_VECTOR_REST_TOKEN"),
    ) -> None:
        """
        Initializes the UpstashDocumentStore.

        :param url: The URL of the Upstash Vector index.
        :param token: The REST token for the Upstash Vector index.
        """
        self.url = url
        self.token = token
        url_val = url.resolve_value()
        token_val = token.resolve_value()
        if not isinstance(url_val, str) or not isinstance(token_val, str):
            msg = "Upstash Vector URL and Token must be valid strings."
            raise ValueError(msg)
        self._index = Index(url=url_val, token=token_val)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this Document Store to a dictionary.

        :returns: The serialized Document Store.
        """
        return default_to_dict(
            self,
            url=self.url.to_dict(),
            token=self.token.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UpstashDocumentStore":
        """
        Deserializes a dictionary into a Document Store.

        :param data: The serialized Document Store.
        :returns: The deserialized Document Store.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["url", "token"])
        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Returns the number of documents in the Document Store.

        :returns: The total number of documents in the Upstash Vector index.
        """
        return self._index.info().vector_count

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Writes documents to the Upstash Vector index.

        :param documents: A list of Documents to be written to the Document Store.
        :param policy: The duplicate policy to apply when a document with the same ID already exists.
            If `DuplicatePolicy.NONE`, it defaults to `DuplicatePolicy.FAIL`.
            `DuplicatePolicy.FAIL` will raise a `DuplicateDocumentError`.
            `DuplicatePolicy.SKIP` will skip existing documents.
            `DuplicatePolicy.OVERWRITE` will replace existing documents.
        :returns: The number of documents written.
        :raises DuplicateDocumentError: If a document with the same ID already exists and `policy` is `FAIL`.
        :raises DocumentStoreError: If a document does not have an embedding.
        """
        if len(documents) == 0:
            return 0

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.OVERWRITE

        if policy in [DuplicatePolicy.SKIP, DuplicatePolicy.FAIL]:
            # Fetch existing to handle SKIP/FAIL
            existing = self._index.fetch([doc.id for doc in documents])
            existing_ids = [res.id for res in existing if res is not None]
            if existing_ids:
                if policy == DuplicatePolicy.FAIL:
                    msg = f"Documents {existing_ids} already exist."
                    raise DuplicateDocumentError(msg)
                # If SKIP, filter them out
                documents = [doc for doc in documents if doc.id not in existing_ids]
                if not documents:
                    return 0

        vectors = []
        for doc in documents:
            if doc.embedding is None:
                msg = f"Document {doc.id} must have an embedding."
                raise DocumentStoreError(msg)

            metadata = doc.meta.copy() if doc.meta else {}
            vector_dict = {"id": doc.id, "vector": doc.embedding, "metadata": metadata}
            if doc.content is not None:
                vector_dict["data"] = doc.content
            if hasattr(doc, "sparse_embedding") and doc.sparse_embedding is not None:
                vector_dict["sparse_vector"] = (list(doc.sparse_embedding.indices), list(doc.sparse_embedding.values))

            vectors.append(vector_dict)

        # Upsert in batches of 1000
        for i in range(0, len(vectors), 1000):
            self._index.upsert(vectors=vectors[i : i + 1000])

        return len(documents)

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes documents from the Document Store by their IDs.

        :param document_ids: A list of document IDs to delete.
        """
        if not document_ids:
            return

        # Delete in batches
        for i in range(0, len(document_ids), 1000):
            self._index.delete(document_ids[i : i + 1000])

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """
        Retrieves documents from the Document Store that match the given filters.

        Note: Due to backend limitations, this method retrieves a maximum of 10,000 documents.

        :param filters: The filters to apply.
        :returns: A list of Documents that match the filters.
        """
        filter_str = ""
        if filters:
            filter_str = _normalize_filters(filters)

        dim = self._index.info().dimension
        dummy_vector = [1.0] + [0.0] * (dim - 1)

        results = self._index.query(
            vector=dummy_vector,
            top_k=TOP_K_LIMIT,
            filter=filter_str,
            include_metadata=True,
            include_vectors=True,
            include_data=True,
        )

        if len(results) == TOP_K_LIMIT:
            logger.warning(
                "Upstash Vector allows a maximum of 10,000 documents to be retrieved by a filter. "
                "The result might be truncated."
            )

        documents = []
        for res in results:
            documents.append(
                Document(
                    id=res.id,
                    content=res.data,
                    embedding=res.vector,
                    meta=res.metadata or {},
                )
            )

        return documents
