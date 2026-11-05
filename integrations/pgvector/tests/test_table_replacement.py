# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from haystack.dataclasses.document import Document

from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore


def _make_staging_store(active_store: PgvectorDocumentStore) -> PgvectorDocumentStore:
    staging_table_name = f"{active_store.table_name}_stage"
    index_suffix = uuid4().hex[:8]
    return PgvectorDocumentStore(
        connection_string=active_store.connection_string,
        schema_name=active_store.schema_name,
        table_name=staging_table_name,
        embedding_dimension=active_store.embedding_dimension,
        vector_type=active_store.vector_type,
        vector_function=active_store.vector_function,
        language=active_store.language,
        search_strategy=active_store.search_strategy,
        hnsw_index_creation_kwargs=active_store.hnsw_index_creation_kwargs,
        hnsw_index_name=f"stage_hnsw_{index_suffix}",
        keyword_index_name=f"stage_kw_{index_suffix}",
    )


@pytest.mark.integration
def test_replace_sync(document_store: PgvectorDocumentStore) -> None:
    document_store.write_documents([Document(id="old", content="old document")])
    staging_store = _make_staging_store(document_store)
    try:
        staging_store.write_documents([Document(id="new", content="new document")])
        backup_name = document_store.replace_table(
            staging_store, backup_table_name=f"{document_store.table_name}_backup"
        )

        assert backup_name == f"{document_store.table_name}_backup"
        assert [document.id for document in document_store.filter_documents()] == ["new"]
        assert [document.id for document in staging_store.filter_documents()] == ["old"]
        assert document_store.recreate_table is False
        assert staging_store.recreate_table is False
    finally:
        staging_store.delete_table()
        staging_store.close()


@pytest.mark.integration
def test_replace_existing_backup(document_store: PgvectorDocumentStore) -> None:
    document_store.write_documents([Document(id="old", content="old document")])
    staging_store = _make_staging_store(document_store)
    try:
        staging_store.write_documents([Document(id="new", content="new document")])

        with pytest.raises(ValueError, match="Backup relation"):
            document_store.replace_table(staging_store, backup_table_name=document_store.keyword_index_name)

        assert [document.id for document in document_store.filter_documents()] == ["old"]
        assert [document.id for document in staging_store.filter_documents()] == ["new"]
    finally:
        staging_store.delete_table()
        staging_store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_replace_async(document_store: PgvectorDocumentStore) -> None:
    await document_store.write_documents_async([Document(id="old", content="old document")])
    staging_store = _make_staging_store(document_store)
    try:
        await staging_store.write_documents_async([Document(id="new", content="new document")])
        backup_name = await document_store.replace_table_async(
            staging_store, backup_table_name=f"{document_store.table_name}_backup"
        )

        assert backup_name == f"{document_store.table_name}_backup"
        assert [document.id for document in await document_store.filter_documents_async()] == ["new"]
        assert [document.id for document in await staging_store.filter_documents_async()] == ["old"]
        assert document_store.recreate_table is False
        assert staging_store.recreate_table is False
    finally:
        await staging_store.delete_table_async()
        await staging_store.close_async()


@pytest.mark.integration
def test_replace_missing_staging(document_store: PgvectorDocumentStore) -> None:
    document_store.write_documents([Document(id="old", content="old document")])
    staging_store = _make_staging_store(document_store)
    try:
        with pytest.raises(ValueError, match="Staging table"):
            document_store.replace_table(staging_store)

        assert [document.id for document in document_store.filter_documents()] == ["old"]
        assert staging_store._table_initialized is False
    finally:
        staging_store.close()


@pytest.mark.integration
def test_replace_rejects_incompatible_stores(document_store: PgvectorDocumentStore) -> None:
    staging_store = PgvectorDocumentStore(
        connection_string=document_store.connection_string,
        schema_name=document_store.schema_name,
        table_name=f"{document_store.table_name}_stage",
        embedding_dimension=document_store.embedding_dimension + 1,
        keyword_index_name="incompatible_keyword_idx",
    )

    with pytest.raises(ValueError, match="same embedding_dimension"):
        document_store.replace_table(staging_store)
