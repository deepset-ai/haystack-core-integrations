# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture
def mock_auth(monkeypatch):
    monkeypatch.setenv("ASTRA_DB_API_ENDPOINT", "http://example.com")
    monkeypatch.setenv("ASTRA_DB_APPLICATION_TOKEN", "test_token")
