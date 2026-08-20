# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from haystack_integrations.components.converters.libreoffice import LibreOfficeFileConverter

CONVERTER_MODULE = "haystack_integrations.components.converters.libreoffice.converter"


def _write_converted_file(args: list[str]) -> None:
    """
    Stand in for soffice: write the file the real binary would have produced.

    The output path is derived from the same argv the converter built, so the tests
    break if the argv contract changes.
    """
    outdir = Path(args[args.index("--outdir") + 1])
    output_file_type = args[args.index("--convert-to") + 1]
    source = Path(args[-1])
    (outdir / source.name).with_suffix(f".{output_file_type}").write_bytes(b"converted bytes")


@pytest.fixture
def real_converter() -> LibreOfficeFileConverter:
    """A converter backed by an actual LibreOffice installation, for the integration tests."""
    return LibreOfficeFileConverter()


@pytest.fixture
def mock_converter() -> Generator[LibreOfficeFileConverter, None]:
    """A converter that believes soffice is installed, so no LibreOffice binary is needed."""
    with patch("shutil.which", return_value="/usr/bin/soffice"):
        yield LibreOfficeFileConverter()


@pytest.fixture
def fake_soffice() -> Generator[MagicMock, None]:
    """Patch the synchronous soffice call so `run` needs no LibreOffice installation."""
    with patch(f"{CONVERTER_MODULE}.subprocess.run") as mock_run:
        mock_run.side_effect = lambda args, **_: _write_converted_file(args)
        yield mock_run


@pytest.fixture
def fake_soffice_async() -> Generator[MagicMock, None]:
    """
    Patch the asynchronous soffice call so `run_async` needs no LibreOffice installation.

    Set `returncode` on the yielded mock to make the stand-in report a failed conversion.
    """
    with patch(f"{CONVERTER_MODULE}.create_subprocess_exec") as mock_exec:
        mock_exec.returncode = 0

        async def _exec(*args):
            if mock_exec.returncode == 0:
                _write_converted_file(list(args))
            process = AsyncMock()
            process.wait = AsyncMock(return_value=mock_exec.returncode)
            return process

        mock_exec.side_effect = _exec
        yield mock_exec


@pytest.fixture
def test_files_path() -> Path:
    """Directory holding the sample office documents, resolved relative to this file."""
    return Path(__file__).parent / "test_files"
