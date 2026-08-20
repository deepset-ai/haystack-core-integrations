# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.libreoffice import LibreOfficeFileConverter

TYPE = "haystack_integrations.components.converters.libreoffice.converter.LibreOfficeFileConverter"


def _converted_to(args: list[str]) -> str:
    """The output file type the converter asked soffice for."""
    return args[args.index("--convert-to") + 1]


async def _convert(converter: LibreOfficeFileConverter, is_async: bool, *args: Any, **kwargs: Any) -> Any:
    """Call `run_async` or `run`, so one test body can cover both."""
    return await converter.run_async(*args, **kwargs) if is_async else converter.run(*args, **kwargs)


def _configured_converter(output_file_type: str) -> LibreOfficeFileConverter:
    """A converter with an `output_file_type` set at init time, needing no LibreOffice installation."""
    with patch("shutil.which", return_value="/usr/bin/soffice"):
        return LibreOfficeFileConverter(output_file_type=output_file_type)


class TestInit:
    def test_init(self, mock_converter: LibreOfficeFileConverter) -> None:
        assert isinstance(mock_converter.soffice_path, str)
        assert mock_converter.output_file_type is None
        assert _configured_converter("pdf").output_file_type == "pdf"

    def test_raises_when_soffice_is_not_installed(self) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="LibreOffice"):
                LibreOfficeFileConverter()


class TestSerde:
    def test_to_dict(self, mock_converter: LibreOfficeFileConverter) -> None:
        assert mock_converter.to_dict() == {"type": TYPE, "init_parameters": {"output_file_type": None}}
        assert _configured_converter("pdf").to_dict() == {"type": TYPE, "init_parameters": {"output_file_type": "pdf"}}

    def test_from_dict(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeFileConverter.from_dict({"type": TYPE, "init_parameters": {}})
        assert isinstance(converter.soffice_path, str)
        # Dictionaries serialized before `output_file_type` was included still deserialize.
        assert converter.output_file_type is None

    def test_round_trip_keeps_converting_to_the_configured_type(
        self, fake_soffice: SimpleNamespace, tmp_path: Path
    ) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            deserialized = LibreOfficeFileConverter.from_dict(_configured_converter("pdf").to_dict())

        # The converter that came back out of the round trip still needs no `output_file_type` argument.
        assert deserialized.run([source])["output"][0].mime_type == "application/pdf"
        assert _converted_to(fake_soffice.calls[-1]) == "pdf"


class TestGetConversionArgs:
    def test_builds_the_soffice_command_line(self, mock_converter: LibreOfficeFileConverter, tmp_path: Path) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        output_path, args = mock_converter._get_conversion_args(source, tmp_path, "pdf")

        assert args == ["/usr/bin/soffice", "--headless", "--convert-to", "pdf", "--outdir", str(tmp_path), str(source)]
        assert output_path == tmp_path / "sample.pdf"

    def test_rejects_an_output_directory_that_does_not_exist(
        self, mock_converter: LibreOfficeFileConverter, tmp_path: Path
    ) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        with pytest.raises(OSError, match="must exist and be writable"):
            mock_converter._get_conversion_args(source, tmp_path / "missing", "pdf")


class TestValidateArgs:
    @pytest.mark.parametrize(
        ("output_file_type", "error"),
        [
            # .pdf is not a supported input type in SUPPORTED_TYPES
            ("docx", "input_file_type"),
            # .doc -> .png is not a valid conversion
            ("png", "is not supported for"),
            # .xyz is not one of the OUTPUT_FILE_TYPE values at all
            ("xyz", "is not supported and must be one of type"),
        ],
    )
    def test_rejects_unsupported_conversions(
        self, mock_converter: LibreOfficeFileConverter, output_file_type: str, error: str
    ) -> None:
        source = "test_file.pdf" if output_file_type == "docx" else "test_file.doc"
        with pytest.raises(ValueError, match=error):
            mock_converter.run([source], output_file_type=output_file_type)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["run", "run_async"])
class TestConvert:
    """`run` and `run_async` share their logic, so every case here covers both."""

    async def test_converts_each_source_in_order(
        self, mock_converter: LibreOfficeFileConverter, fake_soffice: SimpleNamespace, tmp_path: Path, is_async: bool
    ) -> None:
        first, second = tmp_path / "first.doc", tmp_path / "second.ppt"
        first.write_bytes(b"doc")
        second.write_bytes(b"ppt")

        result = await _convert(mock_converter, is_async, [first, second], output_file_type="pdf")

        assert [stream.data for stream in result["output"]] == [b"converted bytes"] * 2
        assert [stream.mime_type for stream in result["output"]] == ["application/pdf"] * 2
        assert len(fake_soffice.calls) == 2

    async def test_writes_a_bytestream_to_a_temporary_file_before_converting(
        self, mock_converter: LibreOfficeFileConverter, fake_soffice: SimpleNamespace, is_async: bool
    ) -> None:
        result = await _convert(mock_converter, is_async, [ByteStream(data=b"raw doc bytes")], output_file_type="pdf")

        assert [stream.data for stream in result["output"]] == [b"converted bytes"]
        # The source handed to soffice is the temporary file, not the original path.
        assert Path(fake_soffice.calls[-1][-1]).name == "input"

    async def test_rejects_a_source_that_does_not_exist(
        self, mock_converter: LibreOfficeFileConverter, is_async: bool
    ) -> None:
        with pytest.raises(FileNotFoundError):
            await _convert(mock_converter, is_async, ["nonexistent_file.doc"], output_file_type="docx")

    async def test_uses_the_output_file_type_from_init(
        self, fake_soffice: SimpleNamespace, tmp_path: Path, is_async: bool
    ) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        result = await _convert(_configured_converter("pdf"), is_async, [source])

        assert result["output"][0].mime_type == "application/pdf"
        assert _converted_to(fake_soffice.calls[-1]) == "pdf"

    async def test_argument_overrides_the_output_file_type_from_init(
        self, fake_soffice: SimpleNamespace, tmp_path: Path, is_async: bool
    ) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        await _convert(_configured_converter("docx"), is_async, [source], output_file_type="pdf")

        assert _converted_to(fake_soffice.calls[-1]) == "pdf"

    async def test_requires_an_output_file_type_somewhere(
        self, mock_converter: LibreOfficeFileConverter, is_async: bool
    ) -> None:
        with pytest.raises(ValueError, match="output_file_type must be provided"):
            await _convert(mock_converter, is_async, ["sample.doc"])

    @pytest.mark.parametrize("as_bytestream", [False, True])
    async def test_raises_when_soffice_exits_non_zero(
        self,
        mock_converter: LibreOfficeFileConverter,
        fake_soffice: SimpleNamespace,
        tmp_path: Path,
        is_async: bool,
        as_bytestream: bool,
    ) -> None:
        path = tmp_path / "sample.doc"
        path.write_bytes(b"doc")
        fake_soffice.returncode = 1
        source = ByteStream(data=b"raw doc bytes") if as_bytestream else path

        with pytest.raises(subprocess.CalledProcessError):
            await _convert(mock_converter, is_async, [source], output_file_type="pdf")


class TestResolveMimeType:
    @pytest.mark.parametrize(
        ("guessed", "output_file_type", "expected"),
        [
            ("application/pdf", "pdf", "application/pdf"),
            (None, "docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            (None, "unknown", None),
        ],
    )
    def test_falls_back_to_the_mapping_when_the_guess_fails(
        self,
        mock_converter: LibreOfficeFileConverter,
        guessed: str | None,
        output_file_type: str,
        expected: str | None,
    ) -> None:
        with patch("mimetypes.guess_type", return_value=(guessed, None)):
            assert mock_converter._resolve_mime_type(Path(f"test.{output_file_type}"), output_file_type) == expected


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["run", "run_async"])
class TestRunIntegration:
    """End-to-end against a real LibreOffice installation, for both `run` and `run_async`."""

    @pytest.mark.parametrize("as_bytestream", [False, True])
    async def test_converts_office_files_to_pdf(
        self,
        real_converter: LibreOfficeFileConverter,
        test_files_path: Path,
        is_async: bool,
        as_bytestream: bool,
    ) -> None:
        paths = [
            test_files_path / "doc" / "sample_doc.doc",
            test_files_path / "ppt" / "sample_ppt.ppt",
            test_files_path / "xls" / "basic_tables_two_sheets.xls",
        ]
        sources = [ByteStream(data=paths[0].read_bytes())] if as_bytestream else paths

        result = await _convert(real_converter, is_async, sources, output_file_type="pdf")

        output = result["output"]
        assert len(output) == len(sources)
        for stream in output:
            assert isinstance(stream, ByteStream)
            assert len(stream.data) > 0
            assert stream.mime_type == "application/pdf"
