# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.opendataloader import OpenDataLoaderConverter


class TestOpenDataLoaderConvertor:
    def test_init_default(self):
        convertor = OpenDataLoaderConverter()

        assert convertor.output_format == "text"
        assert convertor.split_pages is False

    def test_init_custom_params(self):
        convertor = OpenDataLoaderConverter(
            output_format="markdown", split_pages=True, convert_kwargs={"quiet": True, "pages": "2,3"}
        )

        assert convertor.output_format == "markdown"
        assert convertor.split_pages is True
        assert convertor.convert_kwargs == {"quiet": True, "pages": "2,3"}

    def test_component_to_dict_defaults(self):
        convertor = OpenDataLoaderConverter()
        assert convertor.to_dict() == {
            "type": "haystack_integrations.components.converters.opendataloader.converter.OpenDataLoaderConverter",
            "init_parameters": {
                "output_format": "text",
                "split_pages": False,
                "convert_kwargs": {},
            },
        }

    def test_component_to_dict_custom_params(self):
        convertor = OpenDataLoaderConverter(convert_kwargs={"quiet": True, "pages": "2,3"})
        assert convertor.to_dict() == {
            "type": "haystack_integrations.components.converters.opendataloader.converter.OpenDataLoaderConverter",
            "init_parameters": {
                "output_format": "text",
                "split_pages": False,
                "convert_kwargs": {
                    "quiet": True,
                    "pages": "2,3",
                },
            },
        }

    def test_component_from_dict_defaults(self):
        data = {
            "type": "haystack_integrations.components.converters.opendataloader.converter.OpenDataLoaderConverter",
            "init_parameters": {
                "output_format": "text",
                "split_pages": False,
                "convert_kwargs": {},
            },
        }
        restored = OpenDataLoaderConverter.from_dict(data)

        assert not restored.split_pages

    def test_component_from_dict_custom_params(self):
        data = {
            "type": "haystack_integrations.components.converters.opendataloader.converter.OpenDataLoaderConverter",
            "init_parameters": {
                "output_format": "markdown",
                "split_pages": True,
                "convert_kwargs": {"quiet": True, "pages": "2,3"},
            },
        }
        restored = OpenDataLoaderConverter.from_dict(data)

        assert restored.output_format == "markdown"
        assert restored.split_pages is True
        assert restored.convert_kwargs == {"quiet": True, "pages": "2,3"}

    def test_is_java_not_available(self, fixtures_dir: Path):
        converter = OpenDataLoaderConverter(split_pages=True)
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="Java must be installed"):
                converter.run(sources=[fixtures_dir / "sample.pdf"])

    def test_empty_sources_list(self):
        converter = OpenDataLoaderConverter(split_pages=True)
        result = converter.run(sources=[])
        assert result["documents"] == []

    def test_run_with_sources_invalid_file(self, fixtures_dir: Path, caplog):
        converter = OpenDataLoaderConverter(split_pages=True)
        invalid_file_path = fixtures_dir / "invalid_file.txt"

        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=[invalid_file_path])

        assert results["documents"] == []
        assert "Could not read from" in caplog.text

    @pytest.mark.integration
    @pytest.mark.skipif(not OpenDataLoaderConverter._is_java_available(), reason="Java is not installed")
    def test_run_with_source_path(self, fixtures_dir: Path):
        converter = OpenDataLoaderConverter(split_pages=True)
        results = converter.run(sources=[fixtures_dir / "sample.pdf"])
        documents = results["documents"]

        assert len(documents) == 3  # sample.pdf has 3 pages
        for doc in documents:
            assert doc.content is not None
            assert doc.meta.get("file_path") == "sample.pdf"
            assert doc.meta.get("source") == "sample.pdf"
            assert doc.meta.get("page_number") in [1, 2, 3]
            assert doc.meta.get("format") == "text"

    @pytest.mark.integration
    @pytest.mark.skipif(not OpenDataLoaderConverter._is_java_available(), reason="Java is not installed")
    def test_run_with_source_bytestream(self, fixtures_dir: Path):
        stream = ByteStream.from_file_path(fixtures_dir / "sample.pdf")
        converter = OpenDataLoaderConverter(split_pages=True)

        results = converter.run(sources=[stream])
        documents = results["documents"]

        assert len(documents) == 3  # sample.pdf has 3 pages
        for doc in documents:
            assert doc.content is not None
            assert doc.meta.get("file_path") == "document"
            assert doc.meta.get("source") == "document"
            assert doc.meta.get("page_number") in [1, 2, 3]
            assert doc.meta.get("format") == "text"

    def test_run_with_source_bytestream_invalid_source(self, caplog):
        converter = OpenDataLoaderConverter()

        with caplog.at_level(logging.WARNING):
            results = converter.run(sources=[b"hello world"])

        assert results["documents"] == []
        assert "Could not read" in caplog.text
        assert "Unsupported source type <class 'bytes'>" in caplog.text

    @pytest.mark.integration
    @pytest.mark.skipif(not OpenDataLoaderConverter._is_java_available(), reason="Java is not installed")
    def test_run_with_sources_markdown_output_format(self, fixtures_dir: Path):
        converter = OpenDataLoaderConverter(output_format="markdown", split_pages=True)
        results = converter.run(sources=[fixtures_dir / "sample.pdf"])
        documents = results["documents"]

        assert len(documents) == 3  # sample.pdf has 3 pages
        for doc in documents:
            assert doc.content is not None
            assert doc.meta.get("file_path") == "sample.pdf"
            assert doc.meta.get("source") == "sample.pdf"
            assert doc.meta.get("page_number") in [1, 2, 3]
            assert doc.meta.get("format") == "markdown"

    @pytest.mark.integration
    @pytest.mark.skipif(not OpenDataLoaderConverter._is_java_available(), reason="Java is not installed")
    def test_run_with_sources_json_output_format(self, fixtures_dir: Path):
        converter = OpenDataLoaderConverter(output_format="json", split_pages=True)
        results = converter.run(sources=[fixtures_dir / "sample.pdf"])
        documents = results["documents"]

        assert len(documents) == 3  # sample.pdf has 3 pages
        for doc in documents:
            assert doc.content is not None
            parsed_json = json.loads(doc.content)
            assert isinstance(parsed_json, dict)
            assert doc.meta.get("file_path") == "sample.pdf"
            assert doc.meta.get("source") == "sample.pdf"
            assert doc.meta.get("page_number") in [1, 2, 3]
            assert doc.meta.get("format") == "json"

    @pytest.mark.integration
    @pytest.mark.skipif(not OpenDataLoaderConverter._is_java_available(), reason="Java is not installed")
    def test_run_with_sources_no_split(self, fixtures_dir: Path):
        converter = OpenDataLoaderConverter()
        results = converter.run(sources=[fixtures_dir / "sample.pdf"])
        documents = results["documents"]

        assert len(documents) == 1  # sample.pdf is not split into pages
        doc = documents[0]
        assert doc.content is not None
        assert doc.meta.get("file_path") == "sample.pdf"
        assert doc.meta.get("source") == "sample.pdf"
        assert doc.meta.get("page_number") is None
        assert doc.meta.get("format") == "text"

    @pytest.mark.integration
    @pytest.mark.skipif(not OpenDataLoaderConverter._is_java_available(), reason="Java is not installed")
    def test_run_convert_kwargs(self, fixtures_dir: Path):
        converter = OpenDataLoaderConverter(split_pages=True, convert_kwargs={"pages": "2"})
        results = converter.run(sources=[fixtures_dir / "sample.pdf"])
        documents = results["documents"]

        assert len(documents) == 1  # Only page 2 should be converted
        doc = documents[0]
        assert doc.content is not None
        assert doc.meta.get("file_path") == "sample.pdf"
        assert doc.meta.get("source") == "sample.pdf"
        assert doc.meta.get("page_number") == 2
        assert doc.meta.get("format") == "text"
