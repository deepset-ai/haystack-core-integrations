# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.libreoffice import LibreOfficeFileConverter

CONVERTER_MODULE = "haystack_integrations.components.converters.libreoffice.converter"


def _write_converted_file(args: list[str]) -> None:
    """
    Stand in for soffice: write the file the real binary would have produced.

    The output path is derived from the same argv the converter built, so these
    tests break if the argv contract changes.
    """
    outdir = Path(args[args.index("--outdir") + 1])
    output_file_type = args[args.index("--convert-to") + 1]
    source = Path(args[-1])
    (outdir / source.name).with_suffix(f".{output_file_type}").write_bytes(b"converted bytes")


@pytest.fixture
def fake_soffice():
    """Patch the synchronous soffice call so `run` needs no LibreOffice installation."""
    with patch(f"{CONVERTER_MODULE}.subprocess.run") as mock_run:
        mock_run.side_effect = lambda args, **_: _write_converted_file(args)
        yield mock_run


@pytest.fixture
def fake_soffice_async():
    """Patch the asynchronous soffice call so `run_async` needs no LibreOffice installation."""
    with patch(f"{CONVERTER_MODULE}.create_subprocess_exec") as mock_exec:

        async def _exec(*args):
            _write_converted_file(list(args))
            process = AsyncMock()
            process.wait = AsyncMock(return_value=0)
            return process

        mock_exec.side_effect = _exec
        yield mock_exec


class TestGetConversionArgs:
    def test_builds_the_soffice_command_line(self, mock_converter, tmp_path):
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")
        outdir = tmp_path / "out"
        outdir.mkdir()

        output_path, args = mock_converter._get_conversion_args(source, outdir, "pdf")

        assert args == [
            "/usr/bin/soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(outdir),
            str(source),
        ]
        assert output_path == outdir / "sample.pdf"

    def test_rejects_an_output_directory_that_does_not_exist(self, mock_converter, tmp_path):
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        with pytest.raises(OSError, match="must exist and be writable"):
            mock_converter._get_conversion_args(source, tmp_path / "missing", "pdf")


class TestRun:
    def test_converts_each_source_in_order(self, mock_converter, fake_soffice, tmp_path):
        first = tmp_path / "first.doc"
        first.write_bytes(b"doc")
        second = tmp_path / "second.ppt"
        second.write_bytes(b"ppt")

        output = mock_converter.run([first, second], output_file_type="pdf")["output"]

        assert len(output) == 2
        assert all(stream.data == b"converted bytes" for stream in output)
        assert all(stream.mime_type == "application/pdf" for stream in output)
        assert fake_soffice.call_count == 2

    def test_writes_a_bytestream_to_a_temporary_file_before_converting(self, mock_converter, fake_soffice):
        output = mock_converter.run([ByteStream(data=b"raw doc bytes")], output_file_type="pdf")["output"]

        assert len(output) == 1
        assert output[0].data == b"converted bytes"
        # The source handed to soffice is the temporary file, not the original path.
        assert Path(fake_soffice.call_args.args[0][-1]).name == "input"


class TestRunAsync:
    @pytest.mark.asyncio
    async def test_converts_each_source_in_order(self, mock_converter, fake_soffice_async, tmp_path):
        first = tmp_path / "first.doc"
        first.write_bytes(b"doc")
        second = tmp_path / "second.ppt"
        second.write_bytes(b"ppt")

        result = await mock_converter.run_async([first, second], output_file_type="pdf")

        output = result["output"]
        assert len(output) == 2
        assert all(stream.data == b"converted bytes" for stream in output)
        assert all(stream.mime_type == "application/pdf" for stream in output)
        assert fake_soffice_async.call_count == 2

    @pytest.mark.asyncio
    async def test_writes_a_bytestream_to_a_temporary_file_before_converting(self, mock_converter, fake_soffice_async):
        result = await mock_converter.run_async([ByteStream(data=b"raw doc bytes")], output_file_type="pdf")

        output = result["output"]
        assert len(output) == 1
        assert output[0].data == b"converted bytes"
        assert Path(fake_soffice_async.call_args.args[-1]).name == "input"


class TestSerialization:
    def test_to_dict_and_from_dict_round_trip_the_output_file_type(self):
        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeFileConverter(output_file_type="pdf")

            data = converter.to_dict()
            assert data["init_parameters"] == {"output_file_type": "pdf"}

            deserialized = LibreOfficeFileConverter.from_dict(data)

        assert deserialized.output_file_type == "pdf"
