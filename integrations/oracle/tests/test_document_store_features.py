# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

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
