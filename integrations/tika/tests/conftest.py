# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import urllib.error
import urllib.request

import pytest


@pytest.fixture(scope="session")
def _tika_server():
    try:
        urllib.request.urlopen("http://localhost:9998/tika", timeout=5)
    except (urllib.error.URLError, OSError):
        pytest.skip("Tika server is not running at http://localhost:9998/tika")
