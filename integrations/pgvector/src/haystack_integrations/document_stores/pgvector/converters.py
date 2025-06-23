from typing import Any, Dict, List

from haystack import logging
from haystack.dataclasses import ByteStream, Document
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)


def _from_haystack_to_pg_documents(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Internal method to convert a list of Haystack Documents to a list of dictionaries that can be used to insert
    documents into the PgvectorDocumentStore.
    """

    db_documents = []
    for document in documents:
        db_document = {k: v for k, v in document.to_dict(flatten=False).items() if k not in ["score", "blob"]}

        blob = document.blob
        db_document["blob_data"] = blob.data if blob else None
        db_document["blob_meta"] = Jsonb(blob.meta) if blob and blob.meta else None
        db_document["blob_mime_type"] = blob.mime_type if blob and blob.mime_type else None
        db_document["meta"] = Jsonb(db_document["meta"])

        if "sparse_embedding" in db_document:
            sparse_embedding = db_document.pop("sparse_embedding", None)
            if sparse_embedding:
                logger.warning(
                    "Document {doc_id} has the `sparse_embedding` field set,"
                    "but storing sparse embeddings in Pgvector is not currently supported."
                    "The `sparse_embedding` field will be ignored.",
                    doc_id=db_document["id"],
                )

        db_documents.append(db_document)

    return db_documents


def _from_pg_to_haystack_documents(documents: List[Dict[str, Any]]) -> List[Document]:
    """
    Internal method to convert a list of dictionaries from pgvector to a list of Haystack Documents.
    """

    haystack_documents = []
    for document in documents:
        haystack_dict = dict(document)
        blob_data = haystack_dict.pop("blob_data")
        blob_meta = haystack_dict.pop("blob_meta")
        blob_mime_type = haystack_dict.pop("blob_mime_type")

        # convert the embedding to a list of floats
        # for strange reasons, halfvec and vector have different methods to convert the embedding to a list
        if document.get("embedding") is not None:
            if hasattr(document["embedding"], "tolist"):  # vector
                haystack_dict["embedding"] = document["embedding"].tolist()
            else:  # halfvec
                haystack_dict["embedding"] = document["embedding"].to_list()
        # Document.from_dict expects the meta field to be a a dict or not be present (not None)
        if "meta" in haystack_dict and haystack_dict["meta"] is None:
            haystack_dict.pop("meta")

        haystack_document = Document.from_dict(haystack_dict)

        if blob_data:
            blob = ByteStream(data=blob_data, meta=blob_meta, mime_type=blob_mime_type)
            haystack_document.blob = blob

        haystack_documents.append(haystack_document)

    return haystack_documents
