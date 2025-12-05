# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest


@pytest.fixture()
def test_files_path():
    return Path(__file__).parent / "test_files"
