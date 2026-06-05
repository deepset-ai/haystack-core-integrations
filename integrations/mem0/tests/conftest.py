# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def mock_mem0_client():
    """Patch MemoryClient so tests run without a real Mem0 API key."""
    with patch("haystack_integrations.memory_stores.mem0.memory_store.MemoryClient") as mock_cls:
        mock_instance = Mock()
        mock_cls.return_value = mock_instance
        yield mock_instance
