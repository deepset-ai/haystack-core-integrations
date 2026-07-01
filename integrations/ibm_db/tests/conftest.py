# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from haystack_integrations.document_stores.ibm_db import Db2ConnectionConfig, Db2DocumentStore

# DB2 connection configuration for docker-compose DB2 instance
DB2_CONFIG = Db2ConnectionConfig(
    database="testdb",
    hostname="localhost",
    port=50000,
    username="db2inst1",
    password="Passw0rd123!",
    protocol="TCPIP",
)


@pytest.fixture
def connection_config():
    """
    Provide DB2 connection configuration for tests.

    This fixture allows tests to access the connection config without
    duplicating the configuration details.
    """
    return DB2_CONFIG


@pytest.fixture
def document_store(request):
    """
    Create a fresh document store for each test with unique table name.

    This fixture is required by Haystack's mixin tests.
    """
    # Use test name to create unique table name per test
    # Include Python version to avoid conflicts when multiple versions run concurrently
    table_name = f"haystack_{request.node.name}_{sys.version_info.major}_{sys.version_info.minor}"

    # Use standard embedding dimension (768) for compatibility with mixin tests
    store = Db2DocumentStore(
        connection_config=DB2_CONFIG,
        table_name=table_name,
        embedding_dim=768,
        distance_metric="COSINE",
        recreate_table=True,
    )

    yield store

    # Cleanup after test
    try:
        conn = store._get_connection()
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE {store.table_name}")
            conn.commit()
    except Exception:
        # Ignore cleanup errors
        pass


# Made with Bob
