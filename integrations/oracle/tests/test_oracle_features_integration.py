# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import oracledb
import pytest
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.embedders.oracle import OracleDocumentEmbedder, OracleTextEmbedder
from haystack_integrations.components.retrievers.oracle import OracleEmbeddingRetriever, OracleHybridRetriever
from haystack_integrations.document_stores.oracle import (
    OracleDocumentStore,
    OracleVectorizerPreference,
)

pytestmark = pytest.mark.integration

_DEFAULT_EMBEDDING_MODEL = "ALL_MINILM_L12_V2"


def _env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _embedding_params() -> dict:
    if params := _env_value("ORACLE_EMBEDDING_PARAMS"):
        return json.loads(params)
    return {
        "provider": _env_value("ORACLE_EMBEDDING_PROVIDER", default="database"),
        "model": _env_value("ORACLE_EMBEDDING_MODEL", default=_DEFAULT_EMBEDDING_MODEL),
    }


def _proxy() -> str | None:
    return _env_value("ORACLE_EMBEDDING_PROXY", "ORACLE_PROXY")


def _table_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}".upper()


def _drop_table(store: OracleDocumentStore) -> None:
    store.delete_table()


def _drop_sql_index_if_exists(store: OracleDocumentStore, index_name: str) -> None:
    if not index_name.replace("_", "").isalnum():
        msg = f"Invalid test index name: {index_name}"
        raise ValueError(msg)
    try:
        with store._get_connection() as conn, conn.cursor() as cur:
            cur.execute(f"DROP INDEX {index_name}")
            conn.commit()
    except oracledb.DatabaseError as exc:
        message = str(exc)
        if "ORA-01418" in message or "ORA-00942" in message:
            return
        raise


@contextmanager
def _temporary_store(
    connection_config, embedding_dim: int = 4, *, prefix: str = "HS_IT"
) -> Iterator[OracleDocumentStore]:
    store = OracleDocumentStore(
        connection_config=connection_config(),
        table_name=_table_name(prefix),
        embedding_dim=embedding_dim,
        distance_metric="COSINE",
        create_table_if_not_exists=True,
    )
    try:
        yield store
    finally:
        try:
            _drop_table(store)
        finally:
            store.close()


def _text_embedder(connection_config) -> OracleTextEmbedder:
    return OracleTextEmbedder(
        connection_config=connection_config(),
        embedding_params=_embedding_params(),
        proxy=_proxy(),
    )


def _document_embedder(connection_config) -> OracleDocumentEmbedder:
    return OracleDocumentEmbedder(
        connection_config=connection_config(),
        embedding_params=_embedding_params(),
        proxy=_proxy(),
        meta_fields_to_embed=["title"],
    )


def test_contains_and_not_contains_filters_live(connection_config) -> None:
    run_id = uuid.uuid4().hex
    with _temporary_store(connection_config, prefix="HS_FLT") as store:
        store.write_documents(
            [
                Document(content="Oracle vector search", meta={"run_id": run_id, "tags": ["oracle", "vector"]}),
                Document(content="Haystack pipelines", meta={"run_id": run_id, "tags": ["haystack", "pipeline"]}),
            ],
            policy=DuplicatePolicy.NONE,
        )

        contains_results = store.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.run_id", "operator": "==", "value": run_id},
                    {"field": "meta.tags", "operator": "contains", "value": "oracle"},
                ],
            }
        )
        not_contains_results = store.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.run_id", "operator": "==", "value": run_id},
                    {"field": "meta.tags", "operator": "not contains", "value": "oracle"},
                ],
            }
        )

        assert [doc.content for doc in contains_results] == ["Oracle vector search"]
        assert [doc.content for doc in not_contains_results] == ["Haystack pipelines"]


def test_hnsw_and_ivf_vector_index_creation_live(connection_config) -> None:
    with _temporary_store(connection_config, prefix="HS_HNSW") as hnsw_store:
        hnsw_index_name = f"{hnsw_store.table_name}_HNSW"
        hnsw_store.write_documents(
            [Document(content="hnsw", embedding=[1.0, 0.0, 0.0, 0.0])],
            policy=DuplicatePolicy.NONE,
        )
        try:
            hnsw_store.create_vector_index(
                index_type="HNSW",
                params={
                    "idx_name": hnsw_index_name,
                    "neighbors": 2,
                    "efConstruction": 16,
                    "accuracy": 80,
                    "parallel": 1,
                },
            )
        finally:
            _drop_sql_index_if_exists(hnsw_store, hnsw_index_name)

    with _temporary_store(connection_config, prefix="HS_IVF") as ivf_store:
        ivf_index_name = f"{ivf_store.table_name}_IVF"
        ivf_store.write_documents(
            [Document(content="ivf", embedding=[1.0, 0.0, 0.0, 0.0])],
            policy=DuplicatePolicy.NONE,
        )
        try:
            ivf_store.create_vector_index(
                index_type="IVF",
                params={
                    "idx_name": ivf_index_name,
                    "neighbor_partitions": 1,
                    "accuracy": 90,
                    "parallel": 1,
                },
            )
        finally:
            _drop_sql_index_if_exists(ivf_store, ivf_index_name)


@pytest.mark.asyncio
async def test_async_ivf_vector_index_creation_live(connection_config) -> None:
    with _temporary_store(connection_config, prefix="HS_AIVF") as store:
        index_name = f"{store.table_name}_IVF"
        store.write_documents(
            [Document(content="async ivf", embedding=[1.0, 0.0, 0.0, 0.0])],
            policy=DuplicatePolicy.NONE,
        )
        try:
            await store.create_vector_index_async(
                index_type="IVF",
                params={
                    "idx_name": index_name,
                    "neighbor_partitions": 1,
                    "accuracy": 90,
                    "parallel": 1,
                },
            )
        finally:
            _drop_sql_index_if_exists(store, index_name)


def test_oracle_embedders_pipeline_retrieval_live(connection_config) -> None:
    text_embedder = _text_embedder(connection_config)
    query_embedding = text_embedder.run("Oracle Database vector search")["embedding"]
    run_id = uuid.uuid4().hex

    with _temporary_store(connection_config, embedding_dim=len(query_embedding), prefix="HS_EMB") as store:
        docs = [
            Document(content="Oracle Database supports AI Vector Search.", meta={"run_id": run_id, "title": "Oracle"}),
            Document(content="Haystack pipelines connect components.", meta={"run_id": run_id, "title": "Haystack"}),
        ]
        embedded_docs = _document_embedder(connection_config).run(docs)["documents"]
        store.write_documents(embedded_docs, policy=DuplicatePolicy.NONE)

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component(
            "retriever",
            OracleEmbeddingRetriever(
                document_store=store,
                filters={"field": "meta.run_id", "operator": "==", "value": run_id},
                top_k=2,
            ),
        )
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

        result = pipeline.run({"text_embedder": {"text": "Oracle vector search"}})

        retrieved = result["retriever"]["documents"]
        assert retrieved
        assert any("Oracle Database" in doc.content for doc in retrieved)


@pytest.mark.asyncio
async def test_oracle_text_embedder_async_live(connection_config) -> None:
    if not hasattr(oracledb, "connect_async"):
        pytest.skip("python-oracledb does not provide connect_async")

    result = await _text_embedder(connection_config).run_async("Oracle Database vector search")

    assert result["embedding"]
    assert all(isinstance(value, float) for value in result["embedding"])


def test_vectorizer_preference_create_drop_live(connection_config) -> None:
    preference: OracleVectorizerPreference | None = None
    with _temporary_store(connection_config, prefix="HS_PREF") as store:
        try:
            preference = OracleVectorizerPreference.create(
                store,
                _text_embedder(connection_config),
                preference_name=f"{store.table_name}_PREF",
            )
            assert preference.preference_name == f"{store.table_name}_PREF"
        finally:
            if preference is not None:
                preference.drop()


@pytest.mark.asyncio
async def test_async_hybrid_vector_index_creation_live(connection_config) -> None:
    text_embedder = _text_embedder(connection_config)
    query_embedding = text_embedder.run("Oracle hybrid vector search")["embedding"]
    index_name = ""
    store = OracleDocumentStore(
        connection_config=connection_config(),
        table_name=_table_name("HS_AHYB"),
        embedding_dim=len(query_embedding),
        distance_metric="COSINE",
        create_table_if_not_exists=True,
    )
    preference: OracleVectorizerPreference | None = None
    try:
        store.write_documents(
            [Document(content="Oracle hybrid vector search", embedding=query_embedding)],
            policy=DuplicatePolicy.NONE,
        )
        index_name = f"{store.table_name}_HIDX"
        preference = await store.create_hybrid_vector_index_async(
            index_name,
            text_embedder=text_embedder,
            params={"parallel": 1},
        )
        assert isinstance(preference, OracleVectorizerPreference)
    finally:
        try:
            try:
                if index_name:
                    _drop_sql_index_if_exists(store, index_name)
                _drop_table(store)
            finally:
                if preference is not None:
                    preference.drop()
        finally:
            store.close()


def test_hybrid_retriever_live(connection_config) -> None:
    text_embedder = _text_embedder(connection_config)
    document_embedder = _document_embedder(connection_config)
    query_embedding = text_embedder.run("Oracle hybrid search")["embedding"]
    index_name = ""
    store = OracleDocumentStore(
        connection_config=connection_config(),
        table_name=_table_name("HS_HYB"),
        embedding_dim=len(query_embedding),
        distance_metric="COSINE",
        create_table_if_not_exists=True,
    )
    preference: OracleVectorizerPreference | None = None
    try:
        docs = document_embedder.run(
            [
                Document(content="Oracle Database hybrid vector search.", meta={"title": "Oracle", "lang": "en"}),
                Document(content="Haystack supports retrieval pipelines.", meta={"title": "Haystack", "lang": "de"}),
            ]
        )["documents"]
        store.write_documents(docs, policy=DuplicatePolicy.NONE)
        index_name = f"{store.table_name}_HIDX"
        preference = store.create_hybrid_vector_index(index_name, text_embedder=text_embedder, params={"parallel": 1})

        result = OracleHybridRetriever(
            document_store=store,
            index_name=index_name,
            search_mode="hybrid",
            top_k=2,
            return_scores=True,
        ).run("Oracle hybrid vector search")

        assert result["documents"]
        assert any("Oracle Database" in doc.content for doc in result["documents"])
        assert all(doc.score is not None for doc in result["documents"])

        filtered_result = OracleHybridRetriever(
            document_store=store,
            index_name=index_name,
            search_mode="hybrid",
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
            top_k=2,
        ).run("Oracle hybrid vector search")

        assert filtered_result["documents"]
        assert all(doc.meta["lang"] == "en" for doc in filtered_result["documents"])
    finally:
        try:
            try:
                if index_name:
                    _drop_sql_index_if_exists(store, index_name)
                _drop_table(store)
            finally:
                if preference is not None:
                    preference.drop()
        finally:
            store.close()
