# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.libreoffice import LibreOfficeFileConverter


@pytest.fixture
def converter() -> LibreOfficeFileConverter:
    return LibreOfficeFileConverter()


@pytest.fixture
def mock_converter() -> Generator[LibreOfficeFileConverter, None]:
    with patch("shutil.which", return_value="/usr/bin/soffice"):
        yield LibreOfficeFileConverter()


@pytest.fixture
def test_files_path() -> Path:
    return Path("tests") / "test_files"


class TestLibreOfficeFileConverter:
    def test_init(self, mock_converter: LibreOfficeFileConverter) -> None:
        assert isinstance(mock_converter, LibreOfficeFileConverter)
        assert isinstance(mock_converter.soffice_path, str)

    def test_init_raises_when_soffice_not_found(self) -> None:
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="LibreOffice"):
                LibreOfficeFileConverter()

    def test_to_dict(self, mock_converter: LibreOfficeFileConverter) -> None:
        data = mock_converter.to_dict()
        assert data == {
            "type": "haystack_integrations.components.converters.libreoffice.converter.LibreOfficeFileConverter",
            "init_parameters": {},
        }

    def test_from_dict(self) -> None:
        data = {
            "type": "haystack_integrations.components.converters.libreoffice.converter.LibreOfficeFileConverter",
            "init_parameters": {},
        }
        with patch("shutil.which", return_value="/usr/bin/soffice"):
            converter = LibreOfficeFileConverter.from_dict(data)
        assert isinstance(converter.soffice_path, str)

    def test_run_unsupported_input_type(self, mock_converter: LibreOfficeFileConverter) -> None:
        # .pdf is not a supported input type in SUPPORTED_TYPES
        with pytest.raises(ValueError):
            mock_converter.run(["test_file.pdf"], output_file_type="docx")

    def test_run_unsupported_output_type(self, mock_converter: LibreOfficeFileConverter) -> None:
        # .doc -> .png is not a valid conversion
        with pytest.raises(ValueError):
            mock_converter.run(["test_file.doc"], output_file_type="png")

    def test_run_no_file(self, mock_converter: LibreOfficeFileConverter) -> None:
        with pytest.raises(FileNotFoundError):
            mock_converter.run(["nonexistent_file.doc"], output_file_type="docx")

    @pytest.mark.integration
    def test_run(self, converter: LibreOfficeFileConverter, test_files_path: Path) -> None:
        paths = [
            test_files_path / "doc" / "sample_doc.doc",
            test_files_path / "ppt" / "sample_ppt.ppt",
            test_files_path / "xls" / "basic_tables_two_sheets.xls",
        ]

        results = converter.run(paths, output_file_type="pdf")

        output = results["output"]
        assert len(output) == 3
        for stream in output:
            assert isinstance(stream, ByteStream)
            assert len(stream.data) > 0
            assert stream.mime_type == "application/pdf"

    @pytest.mark.integration
    def test_run_bytestream_source(self, converter: LibreOfficeFileConverter, test_files_path: Path) -> None:
        source_path = test_files_path / "doc" / "sample_doc.doc"
        bytestream = ByteStream(data=source_path.read_bytes())

        results = converter.run([bytestream], output_file_type="pdf")

        output = results["output"]
        assert len(output) == 1
        assert isinstance(output[0], ByteStream)
        assert len(output[0].data) > 0
        assert output[0].mime_type == "application/pdf"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_async(self, converter: LibreOfficeFileConverter, test_files_path: Path) -> None:
        paths = [
            test_files_path / "doc" / "sample_doc.doc",
            test_files_path / "ppt" / "sample_ppt.ppt",
            test_files_path / "xls" / "basic_tables_two_sheets.xls",
        ]

        results = await converter.run_async(paths, output_file_type="pdf")

        output = results["output"]
        assert len(output) == 3
        for stream in output:
            assert isinstance(stream, ByteStream)
            assert len(stream.data) > 0
            assert stream.mime_type == "application/pdf"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_async_bytestream_source(
        self, converter: LibreOfficeFileConverter, test_files_path: Path
    ) -> None:
        source_path = test_files_path / "doc" / "sample_doc.doc"
        bytestream = ByteStream(data=source_path.read_bytes())

        results = await converter.run_async([bytestream], output_file_type="pdf")

        output = results["output"]
        assert len(output) == 1
        assert isinstance(output[0], ByteStream)
        assert len(output[0].data) > 0
        assert output[0].mime_type == "application/pdf"

    def test_resolve_mime_type(self, mock_converter: LibreOfficeFileConverter) -> None:
        with patch("mimetypes.guess_type", return_value=("application/pdf", None)):
            mime = mock_converter._resolve_mime_type(Path("test.pdf"), "pdf")
            assert mime == "application/pdf"

        with patch("mimetypes.guess_type", return_value=(None, None)):
            mime = mock_converter._resolve_mime_type(Path("test.docx"), "docx")
            assert mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        with patch("mimetypes.guess_type", return_value=(None, None)):
            mime = mock_converter._resolve_mime_type(Path("test.unknown"), "unknown")
            assert mime is None
