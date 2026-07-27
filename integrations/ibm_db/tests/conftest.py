# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
from haystack.utils import Secret

from haystack_integrations.document_stores.ibm_db import IBMDb2DocumentStore

# DB2 connection parameters for the docker-compose DB2 instance.
# Credentials are Secrets built from plain-token values for the local docker instance.
_USERNAME = "db2inst1"
_PASSWORD = "Passw0rd123!"

DB2_CONNECTION = {
    "database": "testdb",
    "hostname": "localhost",
    "port": 50000,
    "protocol": "TCPIP",
    "username": Secret.from_token(_USERNAME),
    "password": Secret.from_token(_PASSWORD),
}


@pytest.fixture
def connection_config():
    """
    Provide DB2 connection parameters for tests.

    This fixture allows tests to access the connection parameters without
    duplicating the configuration details.
    """
    return DB2_CONNECTION


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
    store = IBMDb2DocumentStore(
        **DB2_CONNECTION,
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
