# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

import pytest
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.tika import TikaDocumentConverter

TIKA_PARSER_PATH = "haystack_integrations.components.converters.tika.converter.tika_parser"


def tika_response(xhtml, status=200):
    """Build what tika_parser.from_buffer() returns for an xmlContent call."""
    return {"status": status, "content": xhtml, "metadata": {"Content-Type": "text/plain"}}


@pytest.fixture
def test_files_path():
    return Path(__file__).parent / "test_files"


class TestTikaDocumentConverter:
    @patch(TIKA_PARSER_PATH + ".from_buffer")
    def test_run(self, mock_tika_parser):
        mock_tika_parser.return_value = tika_response("<div><p>Content of mock source</p></div>")

        component = TikaDocumentConverter()
        source = ByteStream(data=b"placeholder data")
        documents = component.run(sources=[source])["documents"]

        assert len(documents) == 1
        assert documents[0].content == "Content of mock source"

    @patch(TIKA_PARSER_PATH + ".from_buffer")
    def test_run_with_non_200_status(self, mock_tika_parser, caplog):
        mock_tika_parser.return_value = tika_response("<div><p>text</p></div>", status=500)

        with caplog.at_level("WARNING"):
            output = TikaDocumentConverter().run(sources=[ByteStream(data=b"test")])

        assert output["documents"] == []
        assert "returned status 500" in caplog.text

    @patch(TIKA_PARSER_PATH + ".from_buffer")
    def test_run_with_no_content(self, mock_tika_parser, caplog):
        """A 4.x server reports content under a key the pinned 3.x client does not read."""
        mock_tika_parser.return_value = {"status": 200, "content": None}

        with caplog.at_level("WARNING"):
            output = TikaDocumentConverter().run(sources=[ByteStream(data=b"test")])

        assert output["documents"] == []
        assert "requires a Tika 3.x server" in caplog.text

    @patch(TIKA_PARSER_PATH + ".from_buffer")
    def test_run_with_meta(self, mock_tika_parser):
        mock_tika_parser.return_value = tika_response("<div><p>text</p></div>")
        bytestream = ByteStream(data=b"test", meta={"author": "test_author", "language": "en"})

        converter = TikaDocumentConverter()
        output = converter.run(sources=[bytestream], meta={"language": "it"})

        assert output["documents"][0].meta["author"] == "test_author"
        assert output["documents"][0].meta["language"] == "it"

    @patch(TIKA_PARSER_PATH + ".from_buffer")
    def test_run_with_store_full_path_false(self, mock_tika_parser):
        mock_tika_parser.return_value = tika_response("<div><p>text</p></div>")
        bytestream = ByteStream(data=b"test")
        bytestream.meta["file_path"] = "/some/path/to/doc_3.txt"

        converter = TikaDocumentConverter(store_full_path=False)
        output = converter.run(sources=[bytestream])

        assert output["documents"][0].meta["file_path"] == "doc_3.txt"

    @patch(TIKA_PARSER_PATH + ".from_buffer")
    def test_run_with_store_full_path_true(self, mock_tika_parser):
        mock_tika_parser.return_value = tika_response("<div><p>text</p></div>")
        bytestream = ByteStream(data=b"test")
        bytestream.meta["file_path"] = "/some/path/to/doc_3.txt"

        converter = TikaDocumentConverter(store_full_path=True)
        output = converter.run(sources=[bytestream])

        assert output["documents"][0].meta["file_path"] == "/some/path/to/doc_3.txt"

    def test_run_nonexistent_file(self, caplog):
        component = TikaDocumentConverter()
        with caplog.at_level("WARNING"):
            component.run(sources=["nonexistent.pdf"])
            assert "Could not read nonexistent.pdf. Skipping it." in caplog.text

    @pytest.mark.integration
    def test_run_with_txt_files(self, _tika_server, test_files_path):
        component = TikaDocumentConverter()
        output = component.run(sources=[test_files_path / "txt" / "doc_1.txt", test_files_path / "txt" / "doc_2.txt"])
        documents = output["documents"]
        assert len(documents) == 2
        content_0 = documents[0].content.replace("\r\n", "\n")
        content_1 = documents[1].content.replace("\r\n", "\n")
        assert "Some text for testing.\nTwo lines in here." in content_0
        assert "This is a test line.\n123 456 789\n987 654 321" in content_1

    @pytest.mark.integration
    def test_run_with_pdf_file(self, _tika_server, test_files_path):
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
    def test_run_with_docx_file(self, _tika_server, test_files_path):
        component = TikaDocumentConverter()
        output = component.run(sources=[test_files_path / "docx" / "sample_docx.docx"])
        documents = output["documents"]
        assert len(documents) == 1
        assert "Sample Docx File" in documents[0].content
        assert "Now we are in Page 2" in documents[0].content
        assert "Page 3 was empty this is page 4" in documents[0].content
