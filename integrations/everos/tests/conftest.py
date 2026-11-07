# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.utils import Secret

from haystack_integrations.memory_stores.everos import EverOSMemoryStore


@pytest.fixture
def configured_store():
    return EverOSMemoryStore(
        base_url="https://memory.example/api/v2",
        timeout=12.5,
        max_retries=4,
        api_key=Secret.from_env_var("MY_EVEROS_KEY"),
    )
