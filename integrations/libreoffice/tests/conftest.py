# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from haystack_integrations.components.converters.libreoffice import LibreOfficeFileConverter


@pytest.fixture
def converter() -> LibreOfficeFileConverter:
    return LibreOfficeFileConverter()


@pytest.fixture
def mock_converter() -> Generator[LibreOfficeFileConverter, None]:
    """A converter that believes soffice is installed, so no LibreOffice binary is needed."""
    with patch("shutil.which", return_value="/usr/bin/soffice"):
        yield LibreOfficeFileConverter()


@pytest.fixture
def test_files_path() -> Path:
    return Path(__file__).parent / "test_files"
