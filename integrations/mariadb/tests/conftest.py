# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore

MARIADB_HOST = os.environ.get("MARIADB_HOST", "localhost")
MARIADB_PORT = int(os.environ.get("MARIADB_PORT", "3306"))
MARIADB_DB = os.environ.get("MARIADB_DATABASE", "haystack")
MARIADB_USER = os.environ.get("MARIADB_USER", "root")
MARIADB_PASSWORD = os.environ.get("MARIADB_PASSWORD", "password")


@pytest.fixture
def document_store(request):
    table_name = f"haystack_{request.node.name}"
    store = MariaDBDocumentStore(
        host=MARIADB_HOST,
        port=MARIADB_PORT,
        database=MARIADB_DB,
        user=MARIADB_USER,
        password=MARIADB_PASSWORD,
        table_name=table_name,
        embedding_dimension=768,
        recreate_table=True,
    )
    yield store
    store.delete_table()
    store.close()
