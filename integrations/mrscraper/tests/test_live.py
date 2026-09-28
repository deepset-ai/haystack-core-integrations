# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from haystack_integrations.components.mrscraper import MrScraperGetAccountInfo

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("MRSCRAPER_LIVE_TESTS") != "1" or not os.environ.get("MRSCRAPER_API_TOKEN"),
        reason="Set MRSCRAPER_LIVE_TESTS=1 and MRSCRAPER_API_TOKEN to run live tests.",
    ),
]


def test_live_get_account_info():
    result = MrScraperGetAccountInfo().run()
    assert "result" in result
