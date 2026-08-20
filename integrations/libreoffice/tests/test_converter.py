# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.libreoffice import LibreOfficeFileConverter

TYPE = "haystack_integrations.components.converters.libreoffice.converter.LibreOfficeFileConverter"


def _converted_to(args: list[str]) -> str:
    """The output file type the converter asked soffice for."""
    return args[args.index("--convert-to") + 1]


def _configured_converter(output_file_type: str) -> LibreOfficeFileConverter:
    """A converter with an `output_file_type` set at init time, needing no LibreOffice installation."""
    with patch("shutil.which", return_value="/usr/bin/soffice"):
        return LibreOfficeFileConverter(output_file_type=output_file_type)


class TestInit:
    def test_init(self, mock_converter: LibreOfficeFileConverter) -> None:
        assert isinstance(mock_converter, LibreOfficeFileConverter)
        assert isinstance(mock_converter.soffice_path, str)
        assert mock_converter.output_file_type is None

    def test_init_with_output_file_type(self) -> None:
        assert _configured_converter("pdf").output_file_type == "pdf"

    def test_raises_when_soffice_is_not_installed(self) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="LibreOffice"):
                LibreOfficeFileConverter()


class TestSerde:
    def test_to_dict(self, mock_converter: LibreOfficeFileConverter) -> None:
        assert mock_converter.to_dict() == {"type": TYPE, "init_parameters": {"output_file_type": None}}

    def test_to_dict_with_output_file_type(self) -> None:
        assert _configured_converter("pdf").to_dict() == {"type": TYPE, "init_parameters": {"output_file_type": "pdf"}}

    def test_from_dict(self) -> None:
        data = {"type": TYPE, "init_parameters": {"output_file_type": "pdf"}}
        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeFileConverter.from_dict(data)
        assert isinstance(converter.soffice_path, str)
        assert converter.output_file_type == "pdf"

    def test_from_dict_without_init_parameters(self) -> None:
        # Dictionaries serialized before `output_file_type` was included still deserialize.
        data: dict = {"type": TYPE, "init_parameters": {}}
        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeFileConverter.from_dict(data)
        assert converter.output_file_type is None

    def test_round_trip_keeps_converting_to_the_configured_type(self, fake_soffice: MagicMock, tmp_path: Path) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        with patch("shutil.which", return_value="/usr/bin/soffice"):
            deserialized = LibreOfficeFileConverter.from_dict(_configured_converter("pdf").to_dict())

        # The converter that came back out of the round trip still needs no `output_file_type` argument.
        output = deserialized.run([source])["output"]

        assert output[0].mime_type == "application/pdf"
        assert _converted_to(fake_soffice.call_args.args[0]) == "pdf"


class TestGetConversionArgs:
    def test_builds_the_soffice_command_line(self, mock_converter: LibreOfficeFileConverter, tmp_path: Path) -> None:
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

    def test_rejects_an_output_directory_that_does_not_exist(
        self, mock_converter: LibreOfficeFileConverter, tmp_path: Path
    ) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        with pytest.raises(OSError, match="must exist and be writable"):
            mock_converter._get_conversion_args(source, tmp_path / "missing", "pdf")


class TestValidateArgs:
    def test_rejects_an_unsupported_input_type(self, mock_converter: LibreOfficeFileConverter) -> None:
        # .pdf is not a supported input type in SUPPORTED_TYPES
        with pytest.raises(ValueError, match="input_file_type"):
            mock_converter.run(["test_file.pdf"], output_file_type="docx")

    def test_rejects_an_output_type_the_input_type_cannot_reach(self, mock_converter: LibreOfficeFileConverter) -> None:
        # .doc -> .png is not a valid conversion
        with pytest.raises(ValueError, match="is not supported for"):
            mock_converter.run(["test_file.doc"], output_file_type="png")

    def test_rejects_an_output_type_outside_the_supported_literal(
        self, mock_converter: LibreOfficeFileConverter
    ) -> None:
        # .xyz is not one of the OUTPUT_FILE_TYPE values at all
        with pytest.raises(ValueError, match="is not supported and must be one of type"):
            mock_converter.run(["test_file.doc"], output_file_type="xyz")


class TestRun:
    def test_converts_each_source_in_order(
        self, mock_converter: LibreOfficeFileConverter, fake_soffice: MagicMock, tmp_path: Path
    ) -> None:
        first = tmp_path / "first.doc"
        first.write_bytes(b"doc")
        second = tmp_path / "second.ppt"
        second.write_bytes(b"ppt")

        output = mock_converter.run([first, second], output_file_type="pdf")["output"]

        assert len(output) == 2
        assert all(stream.data == b"converted bytes" for stream in output)
        assert all(stream.mime_type == "application/pdf" for stream in output)
        assert fake_soffice.call_count == 2

    def test_writes_a_bytestream_to_a_temporary_file_before_converting(
        self, mock_converter: LibreOfficeFileConverter, fake_soffice: MagicMock
    ) -> None:
        output = mock_converter.run([ByteStream(data=b"raw doc bytes")], output_file_type="pdf")["output"]

        assert len(output) == 1
        assert output[0].data == b"converted bytes"
        # The source handed to soffice is the temporary file, not the original path.
        assert Path(fake_soffice.call_args.args[0][-1]).name == "input"

    def test_rejects_a_source_that_does_not_exist(self, mock_converter: LibreOfficeFileConverter) -> None:
        with pytest.raises(FileNotFoundError):
            mock_converter.run(["nonexistent_file.doc"], output_file_type="docx")

    def test_uses_the_output_file_type_from_init(self, fake_soffice: MagicMock, tmp_path: Path) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        output = _configured_converter("pdf").run([source])["output"]

        assert output[0].mime_type == "application/pdf"
        assert _converted_to(fake_soffice.call_args.args[0]) == "pdf"

    def test_argument_overrides_the_output_file_type_from_init(self, fake_soffice: MagicMock, tmp_path: Path) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        output = _configured_converter("docx").run([source], output_file_type="pdf")["output"]

        assert output[0].mime_type == "application/pdf"
        assert _converted_to(fake_soffice.call_args.args[0]) == "pdf"

    def test_requires_an_output_file_type_somewhere(self, mock_converter: LibreOfficeFileConverter) -> None:
        with pytest.raises(ValueError, match="output_file_type must be provided"):
            mock_converter.run(["sample.doc"])


@pytest.mark.asyncio
class TestRunAsync:
    async def test_converts_each_source_in_order(
        self, mock_converter: LibreOfficeFileConverter, fake_soffice_async: MagicMock, tmp_path: Path
    ) -> None:
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

    async def test_writes_a_bytestream_to_a_temporary_file_before_converting(
        self, mock_converter: LibreOfficeFileConverter, fake_soffice_async: MagicMock
    ) -> None:
        result = await mock_converter.run_async([ByteStream(data=b"raw doc bytes")], output_file_type="pdf")

        output = result["output"]
        assert len(output) == 1
        assert output[0].data == b"converted bytes"
        assert Path(fake_soffice_async.call_args.args[-1]).name == "input"

    async def test_uses_the_output_file_type_from_init(self, fake_soffice_async: MagicMock, tmp_path: Path) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        result = await _configured_converter("pdf").run_async([source])

        assert result["output"][0].mime_type == "application/pdf"
        assert _converted_to(list(fake_soffice_async.call_args.args)) == "pdf"

    async def test_argument_overrides_the_output_file_type_from_init(
        self, fake_soffice_async: MagicMock, tmp_path: Path
    ) -> None:
        source = tmp_path / "sample.doc"
        source.write_bytes(b"doc")

        result = await _configured_converter("docx").run_async([source], output_file_type="pdf")

        assert result["output"][0].mime_type == "application/pdf"
        assert _converted_to(list(fake_soffice_async.call_args.args)) == "pdf"

    @pytest.mark.parametrize("as_bytestream", [False, True])
    async def test_raises_when_soffice_exits_non_zero(
        self,
        mock_converter: LibreOfficeFileConverter,
        fake_soffice_async: MagicMock,
        tmp_path: Path,
        as_bytestream: bool,
    ) -> None:
        path = tmp_path / "sample.doc"
        path.write_bytes(b"doc")
        source = ByteStream(data=b"raw doc bytes") if as_bytestream else path
        fake_soffice_async.returncode = 1

        with pytest.raises(subprocess.CalledProcessError):
            await mock_converter.run_async([source], output_file_type="pdf")

    async def test_requires_an_output_file_type_somewhere(self, mock_converter: LibreOfficeFileConverter) -> None:
        with pytest.raises(ValueError, match="output_file_type must be provided"):
            await mock_converter.run_async(["sample.doc"])


class TestResolveMimeType:
    def test_prefers_the_type_guessed_from_the_output_path(self, mock_converter: LibreOfficeFileConverter) -> None:
        with patch("mimetypes.guess_type", return_value=("application/pdf", None)):
            assert mock_converter._resolve_mime_type(Path("test.pdf"), "pdf") == "application/pdf"

    def test_falls_back_to_the_mapping_when_the_guess_fails(self, mock_converter: LibreOfficeFileConverter) -> None:
        with patch("mimetypes.guess_type", return_value=(None, None)):
            assert (
                mock_converter._resolve_mime_type(Path("test.docx"), "docx")
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

    def test_returns_none_for_a_type_that_is_in_neither(self, mock_converter: LibreOfficeFileConverter) -> None:
        with patch("mimetypes.guess_type", return_value=(None, None)):
            assert mock_converter._resolve_mime_type(Path("test.unknown"), "unknown") is None


@pytest.mark.integration
class TestRunIntegration:
    def test_run(self, real_converter: LibreOfficeFileConverter, test_files_path: Path) -> None:
        paths = [
            test_files_path / "doc" / "sample_doc.doc",
            test_files_path / "ppt" / "sample_ppt.ppt",
            test_files_path / "xls" / "basic_tables_two_sheets.xls",
        ]

        results = real_converter.run(paths, output_file_type="pdf")

        output = results["output"]
        assert len(output) == 3
        for stream in output:
            assert isinstance(stream, ByteStream)
            assert len(stream.data) > 0
            assert stream.mime_type == "application/pdf"

    def test_run_bytestream_source(self, real_converter: LibreOfficeFileConverter, test_files_path: Path) -> None:
        source_path = test_files_path / "doc" / "sample_doc.doc"
        bytestream = ByteStream(data=source_path.read_bytes())

        results = real_converter.run([bytestream], output_file_type="pdf")

        output = results["output"]
        assert len(output) == 1
        assert isinstance(output[0], ByteStream)
        assert len(output[0].data) > 0
        assert output[0].mime_type == "application/pdf"

    @pytest.mark.asyncio
    async def test_run_async(self, real_converter: LibreOfficeFileConverter, test_files_path: Path) -> None:
        paths = [
            test_files_path / "doc" / "sample_doc.doc",
            test_files_path / "ppt" / "sample_ppt.ppt",
            test_files_path / "xls" / "basic_tables_two_sheets.xls",
        ]

        results = await real_converter.run_async(paths, output_file_type="pdf")

        output = results["output"]
        assert len(output) == 3
        for stream in output:
            assert isinstance(stream, ByteStream)
            assert len(stream.data) > 0
            assert stream.mime_type == "application/pdf"

    @pytest.mark.asyncio
    async def test_run_async_bytestream_source(
        self, real_converter: LibreOfficeFileConverter, test_files_path: Path
    ) -> None:
        source_path = test_files_path / "doc" / "sample_doc.doc"
        bytestream = ByteStream(data=source_path.read_bytes())

        results = await real_converter.run_async([bytestream], output_file_type="pdf")

        output = results["output"]
        assert len(output) == 1
        assert isinstance(output[0], ByteStream)
        assert len(output[0].data) > 0
        assert output[0].mime_type == "application/pdf"
