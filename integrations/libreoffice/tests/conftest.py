# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from haystack_integrations.components.converters.libreoffice import LibreOfficeFileConverter

CONVERTER_MODULE = "haystack_integrations.components.converters.libreoffice.converter"


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
def fake_soffice() -> Generator[SimpleNamespace, None]:
    """
    Stand in for soffice on both the sync and the async call, so no LibreOffice is needed.

    Writes the file the real binary would have produced, deriving the output path from the
    same argv the converter built, so the tests break if that contract changes. `calls`
    records every argv; set `returncode` non-zero to report a failed conversion.
    """
    fake = SimpleNamespace(calls=[], returncode=0)

    def _convert(args: list[str]) -> None:
        fake.calls.append(args)
        if fake.returncode:
            return
        outdir = Path(args[args.index("--outdir") + 1])
        suffix = args[args.index("--convert-to") + 1]
        (outdir / Path(args[-1]).name).with_suffix(f".{suffix}").write_bytes(b"converted bytes")

    def _run(args: list[str], **_: object) -> None:
        _convert(args)
        if fake.returncode:
            raise subprocess.CalledProcessError(fake.returncode, args)

    async def _exec(*args: str) -> AsyncMock:
        _convert(list(args))
        process = AsyncMock()
        process.wait = AsyncMock(return_value=fake.returncode)
        return process

    with (
        patch(f"{CONVERTER_MODULE}.subprocess.run", side_effect=_run),
        patch(f"{CONVERTER_MODULE}.create_subprocess_exec", side_effect=_exec),
    ):
        yield fake


@pytest.fixture
def test_files_path() -> Path:
    """Directory holding the sample office documents, resolved relative to this file."""
    return Path(__file__).parent / "test_files"
