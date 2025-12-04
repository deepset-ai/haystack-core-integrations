# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack_integrations.document_stores.valkey import ValkeyDocumentStore


def test_to_dict():
    document_store = ValkeyDocumentStore(
        nodes_list=[{"host": "localhost", "port": 6379}],
        cluster_mode=False,
        request_timeout=30,
        index_name="test_index",
        distance_metric="cosine",
        embedding_dim=512,
    )

    expected = {
        "type": "haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore",
        "init_parameters": {
            "nodes_list": [{"host": "localhost", "port": 6379}],
            "cluster_mode": False,
            "use_tls": False,
            "username": None,
            "password": None,
            "request_timeout": 30,
            "retry_attempts": 3,
            "retry_base_delay_ms": 1000,
            "retry_exponent_base": 2,
            "batch_size": 100,
            "index_name": "test_index",
            "distance_metric": "cosine",
            "embedding_dim": 512,
        },
    }

    assert document_store.to_dict() == expected


def test_from_dict():
    data = {
        "type": "haystack_integrations.document_stores.valkey.document_store.ValkeyDocumentStore",
        "init_parameters": {
            "nodes_list": [{"host": "localhost", "port": 6379}],
            "cluster_mode": True,
            "request_timeout": 60,
            "index_name": "custom_index",
            "distance_metric": "l2",
            "embedding_dim": 768,
        },
    }

    document_store = ValkeyDocumentStore.from_dict(data)

    assert document_store._nodes_list == [{"host": "localhost", "port": 6379}]
    assert document_store._cluster_mode is True
    assert document_store._request_timeout == 60
    assert document_store._index_name == "custom_index"
    assert document_store._distance_metric.name.lower() == "l2"
    assert document_store._embedding_dim == 768
