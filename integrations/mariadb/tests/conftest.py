# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.utils import Secret

from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore


@pytest.fixture
def document_store(request):
    store = MariaDBDocumentStore(
        host="127.0.0.1",
        user=Secret.from_token("root"),
        password=Secret.from_token("password"),
        table_name=f"haystack_{request.node.name}",
        recreate_table=True,
    )
    yield store
    store.delete_table()
    store.close()
