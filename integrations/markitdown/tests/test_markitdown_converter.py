# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.markitdown import MarkItDownConverter


@pytest.fixture
def mock_result():
    result = MagicMock()
    result.text_content = "# Hello\n\nThis is converted markdown."
    return result


class TestMarkItDownConverter:
    def test_init_defaults(self):
        converter = MarkItDownConverter()
        assert converter.store_full_path is False

    def test_init_store_full_path(self):
        converter = MarkItDownConverter(store_full_path=True)
        assert converter.store_full_path is True

    def test_to_dict(self):
        converter = MarkItDownConverter()
        data = component_to_dict(converter, "MarkItDownConverter")
        assert (
            data["type"]
            == "haystack_integrations.components.converters.markitdown.markitdown_converter.MarkItDownConverter"
        )
        assert data["init_parameters"]["store_full_path"] is False

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.converters.markitdown.markitdown_converter.MarkItDownConverter",
            "init_parameters": {"store_full_path": True},
        }
        converter = component_from_dict(MarkItDownConverter, data, "MarkItDownConverter")
        assert converter.store_full_path is True

    def test_run_with_file_path(self, mock_result, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello world")

        converter = MarkItDownConverter()
        converter._converter = MagicMock()
        converter._converter.convert.return_value = mock_result

        result = converter.run(sources=[txt_file])

        assert len(result["documents"]) == 1
        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content == "# Hello\n\nThis is converted markdown."

    def test_run_stores_filename_not_full_path(self, mock_result, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello world")

        converter = MarkItDownConverter(store_full_path=False)
        converter._converter = MagicMock()
        converter._converter.convert.return_value = mock_result

        result = converter.run(sources=[txt_file])

        assert result["documents"][0].meta["file_path"] == "test.txt"

    def test_run_stores_full_path(self, mock_result, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello world")

        converter = MarkItDownConverter(store_full_path=True)
        converter._converter = MagicMock()
        converter._converter.convert.return_value = mock_result

        result = converter.run(sources=[txt_file])

        assert result["documents"][0].meta["file_path"] == str(txt_file)

    def test_run_with_bytestream(self, mock_result):
        bytestream = ByteStream(data=b"some content", meta={"file_path": "doc.txt"})

        converter = MarkItDownConverter()
        converter._converter = MagicMock()
        converter._converter.convert_stream.return_value = mock_result

        result = converter.run(sources=[bytestream])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "# Hello\n\nThis is converted markdown."
        converter._converter.convert_stream.assert_called_once()

    def test_run_with_meta(self, mock_result, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello world")

        converter = MarkItDownConverter()
        converter._converter = MagicMock()
        converter._converter.convert.return_value = mock_result

        result = converter.run(sources=[txt_file], meta={"author": "Alice"})

        assert result["documents"][0].meta["author"] == "Alice"

    def test_run_skips_unreadable_source(self, caplog):
        converter = MarkItDownConverter()
        converter._converter = MagicMock()
        converter._converter.convert.side_effect = FileNotFoundError("No such file or directory")

        with caplog.at_level(logging.WARNING):
            result = converter.run(sources=[Path("/nonexistent/file.pdf")])

        assert result["documents"] == []
        assert "Could not convert" in caplog.text

    def test_run_skips_conversion_error(self, tmp_path, caplog):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello world")

        converter = MarkItDownConverter()
        converter._converter = MagicMock()
        converter._converter.convert.side_effect = Exception("Conversion failed")

        with caplog.at_level(logging.WARNING):
            result = converter.run(sources=[txt_file])

        assert result["documents"] == []
        assert "Could not convert" in caplog.text

    def test_run_multiple_sources(self, mock_result, tmp_path):
        files = []
        for i in range(3):
            f = tmp_path / f"file{i}.txt"
            f.write_text(f"Content {i}")
            files.append(f)

        converter = MarkItDownConverter()
        converter._converter = MagicMock()
        converter._converter.convert.return_value = mock_result

        result = converter.run(sources=files)

        assert len(result["documents"]) == 3

    def test_run_meta_list(self, mock_result, tmp_path):
        files = []
        for i in range(2):
            f = tmp_path / f"file{i}.txt"
            f.write_text(f"Content {i}")
            files.append(f)

        converter = MarkItDownConverter()
        converter._converter = MagicMock()
        converter._converter.convert.return_value = mock_result

        result = converter.run(sources=files, meta=[{"tag": "a"}, {"tag": "b"}])

        assert result["documents"][0].meta["tag"] == "a"
        assert result["documents"][1].meta["tag"] == "b"

    @pytest.mark.integration
    def test_run_integration(self, tmp_path):
        txt_file = tmp_path / "hello.txt"
        txt_file.write_text("Hello, world!")

        converter = MarkItDownConverter()
        result = converter.run(sources=[txt_file])

        assert len(result["documents"]) == 1
        assert "Hello" in result["documents"][0].content
