# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.tika import TikaDocumentConverter

TIKA_PARSER_PATH = "haystack_integrations.components.converters.tika.converter.tika_parser"


@pytest.fixture
def test_files_path():
    return Path(__file__).parent / "test_files"


class TestTikaDocumentConverter:
    @patch(TIKA_PARSER_PATH + ".from_buffer")
    def test_run(self, mock_tika_parser):
        mock_tika_parser.return_value = {"content": "<div><p>Content of mock source</p></div>"}

        component = TikaDocumentConverter()
        source = ByteStream(data=b"placeholder data")
        documents = component.run(sources=[source])["documents"]

        assert len(documents) == 1
        assert documents[0].content == "Content of mock source"

    @patch(TIKA_PARSER_PATH + ".from_buffer")
    def test_run_with_meta(self, mock_tika_parser):
        mock_tika_parser.return_value = {"content": "<div><p>text</p></div>"}
        bytestream = ByteStream(data=b"test", meta={"author": "test_author", "language": "en"})

        converter = TikaDocumentConverter()
        with patch(TIKA_PARSER_PATH + ".from_buffer"):
            output = converter.run(sources=[bytestream], meta={"language": "it"})

        assert output["documents"][0].meta["author"] == "test_author"
        assert output["documents"][0].meta["language"] == "it"

    @patch(TIKA_PARSER_PATH + ".from_buffer")
    def test_run_with_store_full_path_false(self, mock_tika_parser):
        mock_tika_parser.return_value = {"content": "<div><p>text</p></div>"}
        bytestream = ByteStream(data=b"test")
        bytestream.meta["file_path"] = "/some/path/to/doc_3.txt"

        converter = TikaDocumentConverter(store_full_path=False)
        with patch(TIKA_PARSER_PATH + ".from_buffer"):
            output = converter.run(sources=[bytestream])

        assert output["documents"][0].meta["file_path"] == "doc_3.txt"

    @patch(TIKA_PARSER_PATH + ".from_buffer")
    def test_run_with_store_full_path_true(self, mock_tika_parser):
        mock_tika_parser.return_value = {"content": "<div><p>text</p></div>"}
        bytestream = ByteStream(data=b"test")
        bytestream.meta["file_path"] = "/some/path/to/doc_3.txt"

        converter = TikaDocumentConverter(store_full_path=True)
        with patch(TIKA_PARSER_PATH + ".from_buffer"):
            output = converter.run(sources=[bytestream])

        assert output["documents"][0].meta["file_path"] == "/some/path/to/doc_3.txt"

    def test_run_nonexistent_file(self, caplog):
        component = TikaDocumentConverter()
        with caplog.at_level("WARNING"):
            component.run(sources=["nonexistent.pdf"])
            assert "Could not read nonexistent.pdf. Skipping it." in caplog.text

    def test_to_dict(self):
        converter = TikaDocumentConverter()
        data = converter.to_dict()
        assert data == {
            "type": "haystack_integrations.components.converters.tika.converter.TikaDocumentConverter",
            "init_parameters": {"tika_url": "http://localhost:9998/tika", "store_full_path": False},
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.converters.tika.converter.TikaDocumentConverter",
            "init_parameters": {"tika_url": "http://localhost:9998/tika", "store_full_path": True},
        }
        converter = TikaDocumentConverter.from_dict(data)
        assert converter.tika_url == "http://localhost:9998/tika"
        assert converter.store_full_path is True

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skipif(sys.platform != "linux", reason="Tika Docker container can only run on Ubuntu GitHub runners")
    def test_run_with_txt_files(self, test_files_path):
        component = TikaDocumentConverter()
        output = component.run(sources=[test_files_path / "txt" / "doc_1.txt", test_files_path / "txt" / "doc_2.txt"])
        documents = output["documents"]
        assert len(documents) == 2
        assert "Some text for testing.\nTwo lines in here." in documents[0].content
        assert "This is a test line.\n123 456 789\n987 654 321" in documents[1].content

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skipif(sys.platform != "linux", reason="Tika Docker container can only run on Ubuntu GitHub runners")
    def test_run_with_pdf_file(self, test_files_path):
        component = TikaDocumentConverter()
        output = component.run(
            sources=[test_files_path / "pdf" / "sample_pdf_1.pdf", test_files_path / "pdf" / "sample_pdf_2.pdf"]
        )
        documents = output["documents"]
        assert len(documents) == 2
        assert "A sample PDF file" in documents[0].content
        assert "Page 2 of Sample PDF" in documents[0].content
        assert "Page 4 of Sample PDF" in documents[0].content
        assert documents[0].content.count("\f") == 3  # 4 pages

        assert "First Page" in documents[1].content
        assert (
            "Wiki engines usually allow content to be written using a simplified markup language"
            in documents[1].content
        )
        assert documents[1].content.count("\f") == 3  # 4 pages

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skipif(sys.platform != "linux", reason="Tika Docker container can only run on Ubuntu GitHub runners")
    def test_run_with_docx_file(self, test_files_path):
        component = TikaDocumentConverter()
        output = component.run(sources=[test_files_path / "docx" / "sample_docx.docx"])
        documents = output["documents"]
        assert len(documents) == 1
        assert "Sample Docx File" in documents[0].content
        assert "Now we are in Page 2" in documents[0].content
        assert "Page 3 was empty this is page 4" in documents[0].content
