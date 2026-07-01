# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from haystack_integrations.document_stores.oracle import OracleVectorizerPreference


def test_create_vector_index_ivf_sql(patched_store, mock_pool):
    _, _, cursor = mock_pool
    patched_store.create_vector_index(
        index_type="IVF",
        params={"idx_name": "TEST_DOCS_IVF", "neighbor_partitions": 64, "samples_per_partition": 8},
    )

    sql = cursor.execute.call_args[0][0]

    assert "CREATE VECTOR INDEX IF NOT EXISTS TEST_DOCS_IVF" in sql
    assert "ORGANIZATION NEIGHBOR PARTITIONS" in sql
    assert "neighbor partitions 64" in sql
    assert "samples_per_partition 8" in sql


def test_create_hybrid_vector_index_uses_text_column(patched_store, mock_pool):
    _, _, cursor = mock_pool
    preference = OracleVectorizerPreference(patched_store, "PREF_TEST_DOCS")

    returned = patched_store.create_hybrid_vector_index(
        "TEST_DOCS_HYBRID",
        vectorizer_preference=preference,
        params={"parameters": {"language": "american"}, "parallel": 2},
    )
    sql = cursor.execute.call_args[0][0]

    assert returned is preference
    assert "CREATE HYBRID VECTOR INDEX TEST_DOCS_HYBRID ON test_docs(text)" in sql
    assert "vectorizer PREF_TEST_DOCS" in sql
    assert "language american" in sql
    assert "PARALLEL 2" in sql


def test_create_hybrid_vector_index_drops_auto_preference_on_failure(patched_store, mock_pool, monkeypatch):
    _, _, cursor = mock_pool
    cursor.execute.side_effect = RuntimeError("DDL failed")
    preference = OracleVectorizerPreference(patched_store, "PREF_TEST_DOCS")
    drop_calls = []

    def create_preference(_cls, _document_store, _text_embedder):
        return preference

    def drop_preference():
        drop_calls.append(preference.preference_name)

    monkeypatch.setattr(OracleVectorizerPreference, "create", classmethod(create_preference))
    monkeypatch.setattr(preference, "drop", drop_preference)

    with pytest.raises(RuntimeError, match="DDL failed"):
        patched_store.create_hybrid_vector_index("TEST_DOCS_HYBRID", text_embedder=object())

    assert drop_calls == ["PREF_TEST_DOCS"]


def test_create_hybrid_vector_index_keeps_caller_preference_on_failure(patched_store, mock_pool, monkeypatch):
    _, _, cursor = mock_pool
    cursor.execute.side_effect = RuntimeError("DDL failed")
    preference = OracleVectorizerPreference(patched_store, "PREF_TEST_DOCS")
    drop_calls = []

    def drop_preference():
        drop_calls.append(preference.preference_name)

    monkeypatch.setattr(preference, "drop", drop_preference)

    with pytest.raises(RuntimeError, match="DDL failed"):
        patched_store.create_hybrid_vector_index("TEST_DOCS_HYBRID", vectorizer_preference=preference)

    assert drop_calls == []


@pytest.mark.asyncio
async def test_create_hybrid_vector_index_async_drops_auto_preference_on_fallback_failure(patched_store, monkeypatch):
    preference = OracleVectorizerPreference(patched_store, "PREF_TEST_DOCS")
    drop_calls = []

    async def create_preference(_cls, _document_store, _text_embedder):
        return preference

    async def has_async_pool():
        return False

    async def drop_preference():
        drop_calls.append(preference.preference_name)

    def create_index(_idx_name, **_kwargs):
        msg = "DDL failed"
        raise RuntimeError(msg)

    monkeypatch.setattr(OracleVectorizerPreference, "create_async", classmethod(create_preference))
    monkeypatch.setattr(patched_store, "_has_async_pool", has_async_pool)
    monkeypatch.setattr(patched_store, "create_hybrid_vector_index", create_index)
    monkeypatch.setattr(preference, "drop_async", drop_preference)

    with pytest.raises(RuntimeError, match="DDL failed"):
        await patched_store.create_hybrid_vector_index_async("TEST_DOCS_HYBRID", text_embedder=object())

    assert drop_calls == ["PREF_TEST_DOCS"]


def test_hybrid_retrieval_keeps_scores_aligned_when_rowid_disappears(patched_store, mock_pool):
    _, _, cursor = mock_pool
    matching_id = "B" * 32
    search_rows = [
        {"rowid": "deleted_rowid", "score": 0.9, "text_score": 0.8, "vector_score": 0.7},
        {"rowid": "matching_rowid", "score": 0.3, "text_score": 0.2, "vector_score": 0.1},
    ]
    cursor.fetchone.side_effect = [
        (json.dumps(search_rows),),
        None,
        (matching_id, "matched document", '{"lang":"en"}'),
    ]

    documents = patched_store._hybrid_retrieval(
        "query",
        index_name="TEST_DOCS_HYBRID",
        top_k=2,
        return_scores=True,
    )

    assert len(documents) == 1
    assert documents[0].id == matching_id
    assert documents[0].score == 0.3
    assert documents[0].meta == {"lang": "en", "score": 0.3, "text_score": 0.2, "vector_score": 0.1}


def test_default_to_dict_omits_new_vector_index_fields(patched_store):
    data = patched_store.to_dict()

    assert "vector_index_type" not in data["init_parameters"]
    assert "vector_index_params" not in data["init_parameters"]


def test_custom_vector_index_config_roundtrips(patched_store):
    patched_store.vector_index_type = "IVF"
    patched_store.vector_index_params = {"idx_name": "TEST_DOCS_IVF", "neighbor_partitions": 64}

    restored = patched_store.from_dict(patched_store.to_dict())

    assert restored.vector_index_type == "IVF"
    assert restored.vector_index_params == {"idx_name": "TEST_DOCS_IVF", "neighbor_partitions": 64}
