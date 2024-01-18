# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, List, Literal, Optional

import sqlalchemy
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from sqlalchemy import create_engine, delete, text
from sqlalchemy.dialects.postgresql import BYTEA, JSON, TEXT, VARCHAR, insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, mapped_column
from sqlalchemy.schema import Index

from pgvector.sqlalchemy import Vector

logger = logging.getLogger(__name__)

HNSW_INDEX_CREATION_VALID_KWARGS = ["m", "ef_construction"]

SIMILARITY_FUNCTION_TO_POSTGRESQL_OPS = {
    "cosine_distance": "vector_cosine_ops",
    "max_inner_product": "vector_ip_ops",
    "l2_distance": "vector_l2_ops",
}


class _AbstractDBDocument(DeclarativeBase):
    # __abstract__ = True means that this class does not correspond to a table in the database
    # this allows setting dinamically the table name
    __abstract__ = True

    id = mapped_column(VARCHAR(64), primary_key=True)
    embedding = mapped_column(Vector(None), nullable=True)
    content = mapped_column(TEXT, nullable=True)
    dataframe = mapped_column(JSON, nullable=True)
    blob = mapped_column(BYTEA, nullable=True)
    blob_meta = mapped_column(JSON, nullable=True)
    blob_mime_type = mapped_column(VARCHAR(255), nullable=True)
    meta = mapped_column(JSON, nullable=True)


def _get_db_document(table_name):
    return type("DBDocument", (_AbstractDBDocument,), {"__tablename__": table_name})


class PgvectorDocumentStore:
    def __init__(
        self,
        *,
        connection_string: str,
        table_name: str = "haystack_documents",
        embedding_similarity_function: Literal[
            "cosine_distance", "max_inner_product", "l2_distance"
        ] = "cosine_distance",
        recreate_table: bool = False,
        search_strategy: Literal["exact_nearest_neighbor", "hnsw"] = "exact_nearest_neighbor",
        hnsw_index_creation_kwargs: Optional[Dict[str, Any]] = None,
        hnsw_ef_search: Optional[int] = None,
    ):
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

        self._DBDocument = _get_db_document(table_name)

        self._session = Session(engine)

        if recreate_table:
            self._DBDocument.__table__.drop(engine, checkfirst=True)
            self._DBDocument.__table__.create(engine)

        self._distance = getattr(self._DBDocument.embedding, embedding_similarity_function)

        hnsw_index_creation_kwargs = hnsw_index_creation_kwargs or {}

        if search_strategy == "hnsw":
            effective_hnsw_index_creation_kwargs = {}
            for key, value in hnsw_index_creation_kwargs.items():
                if key in HNSW_INDEX_CREATION_VALID_KWARGS:
                    effective_hnsw_index_creation_kwargs[key] = value
                else:
                    logger.warning(
                        "Invalid HNSW index creation keyword argument: %sValid arguments are: %s",
                        key,
                        HNSW_INDEX_CREATION_VALID_KWARGS,
                    )

            index = Index(
                "hnsw_index",
                self._DBDocument.embedding,
                postgresql_using="hnsw",
                postgresql_with=effective_hnsw_index_creation_kwargs,
                postgresql_ops={"embedding": SIMILARITY_FUNCTION_TO_POSTGRESQL_OPS[embedding_similarity_function]},
            )
            index.create(engine)

            if hnsw_ef_search:
                conn.execute(text("SET hnsw.ef_search = :ef_search"), ef_search=hnsw_ef_search)

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        return self._session.query(self._DBDocument).count()

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:  # noqa: ARG002
        return []

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes documents into to PgvectorDocumentStore.

        :param documents: A list of Documents to write to the document store.
        :param policy: The duplicate policy to use when writing documents.
        :raises DuplicateDocumentError: If a document with the same id already exists in the document store
             and the policy is set to DuplicatePolicy.FAIL (or not specified).
        :return: The number of documents written to the document store.
        """
        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        db_documents = []
        for document in documents:
            db_document = document.to_dict(flatten=False)
            db_document.pop("score")
            blob = document.blob
            if blob:
                if blob.meta:
                    db_document["blob_meta"] = blob.meta
                if blob.mime_type:
                    db_document["blob_mime_type"] = blob.mime_type
            db_documents.append(db_document)

        # we use Postgresql insert statements to properly handle the different policies
        insert_statement = insert(self._DBDocument).values(db_documents)
        if policy == DuplicatePolicy.OVERWRITE:
            insert_statement = insert_statement.on_conflict_do_update(
                constraint=self._DBDocument.__table__.primary_key,
                set_={k: getattr(insert_statement.excluded, k) for k in db_documents[0].keys() if k != "id"},
            )
        elif policy == DuplicatePolicy.SKIP:
            insert_statement = insert_statement.on_conflict_do_nothing()

        try:
            result = self._session.execute(insert_statement)
            self._session.commit()
        except sqlalchemy.exc.IntegrityError as e:
            self._session.rollback()
            raise DuplicateDocumentError from e

        return result.rowcount

    def delete_documents(self, document_ids: List[str]) -> None:
        statement = delete(self._DBDocument).where(self._DBDocument.id.in_(document_ids))

        try:
            self._session.execute(statement)
            self._session.commit()
        except SQLAlchemyError as e:
            self._session.rollback()
            msg = "Could not delete documents from PgvectorDocumentStore"
            raise DocumentStoreError(msg) from e
