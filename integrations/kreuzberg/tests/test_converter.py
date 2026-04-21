# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import ByteStream
from kreuzberg import (
    ExtractedTable,
    ExtractionConfig,
    ExtractionResult,
    LanguageDetectionConfig,
    OcrConfig,
    config_to_json,
)

from haystack_integrations.components.converters.kreuzberg import KreuzbergConverter
from haystack_integrations.components.converters.kreuzberg.utils import (
    _is_batch_error,
    _serialize_keywords,
    _serialize_warnings,
)

CONVERTER_MODULE = "haystack_integrations.components.converters.kreuzberg.converter"


def test_init_default() -> None:
    converter = KreuzbergConverter()
    assert converter.config is None
    assert converter.config_path is None
    assert converter.store_full_path is False
    assert converter.batch is True
    assert converter.easyocr_kwargs is None


def test_init_with_all_params() -> None:
    config = ExtractionConfig(output_format="markdown")
    converter = KreuzbergConverter(
        config=config,
        store_full_path=True,
        batch=False,
        easyocr_kwargs={"gpu": False},
    )
    assert converter.config is config
    assert converter.config_path is None
    assert converter.store_full_path is True
    assert converter.batch is False
    assert converter.easyocr_kwargs == {"gpu": False}


def test_serialization_roundtrip_defaults() -> None:
    converter = KreuzbergConverter()
    d = converter.to_dict()
    assert d == {
        "type": "haystack_integrations.components.converters.kreuzberg.converter.KreuzbergConverter",
        "init_parameters": {
            "config": None,
            "config_path": None,
            "store_full_path": False,
            "batch": True,
            "easyocr_kwargs": None,
        },
    }
    restored = KreuzbergConverter.from_dict(d)
    assert restored.config is None
    assert restored.config_path is None
    assert restored.store_full_path is False
    assert restored.batch is True
    assert restored.easyocr_kwargs is None


def test_serialization_roundtrip_all_params() -> None:
    converter = KreuzbergConverter(
        config=ExtractionConfig(
            output_format="html",
            ocr=OcrConfig(backend="tesseract", language="deu"),
        ),
        store_full_path=True,
        batch=False,
        easyocr_kwargs={"gpu": True, "beam_width": 3},
    )
    # to_dict serializes config as JSON string
    d = converter.to_dict()
    params = d["init_parameters"]
    assert isinstance(params["config"], str)
    assert json.loads(params["config"])["output_format"] == "html"
    assert params["store_full_path"] is True
    assert params["batch"] is False
    assert params["easyocr_kwargs"] == {"gpu": True, "beam_width": 3}

    # Save config JSON before from_dict mutates d in place
    original_config_json = json.loads(params["config"])

    # from_dict restores all fields
    restored = KreuzbergConverter.from_dict(d)
    assert restored.config.output_format == "html"
    assert restored.config.ocr.backend == "tesseract"
    assert restored.config.ocr.language == "deu"
    assert restored.config_path is None
    assert restored.store_full_path is True
    assert restored.batch is False
    assert restored.easyocr_kwargs == {"gpu": True, "beam_width": 3}

    # Double roundtrip produces identical to_dict output
    d2 = restored.to_dict()
    p1 = {k: v for k, v in d["init_parameters"].items() if k != "config"}
    p2 = {k: v for k, v in d2["init_parameters"].items() if k != "config"}
    assert p1 == p2
    assert original_config_json == json.loads(d2["init_parameters"]["config"])


def test_serialization_roundtrip_config_path() -> None:
    # String config_path
    converter = KreuzbergConverter(config_path="/tmp/kreuzberg.toml")
    d = converter.to_dict()
    assert d["init_parameters"]["config_path"] == "/tmp/kreuzberg.toml"
    restored = KreuzbergConverter.from_dict(d)
    assert restored.config_path == "/tmp/kreuzberg.toml"

    # Path object is normalized to str
    converter2 = KreuzbergConverter(config_path=Path("/some/path/config.json"))
    d2 = converter2.to_dict()
    assert d2["init_parameters"]["config_path"] == "/some/path/config.json"
    assert isinstance(d2["init_parameters"]["config_path"], str)


def test_serialization_from_dict_empty_init_parameters() -> None:
    d = {
        "type": "haystack_integrations.components.converters.kreuzberg.converter.KreuzbergConverter",
        "init_parameters": {},
    }
    converter = KreuzbergConverter.from_dict(d)
    assert converter.config is None
    assert converter.config_path is None
    assert converter.store_full_path is False
    assert converter.batch is True
    assert converter.easyocr_kwargs is None


def test_build_config_default(converter: KreuzbergConverter) -> None:
    config = converter._build_config()
    assert config.output_format == "plain"
    assert config.language_detection is not None
    assert config.language_detection.enabled is True


def test_build_config_does_not_mutate_self_config() -> None:
    base = ExtractionConfig(output_format="html")
    converter = KreuzbergConverter(config=base)
    converter._build_config()
    assert base.output_format == "html"


def test_build_config_from_file() -> None:
    config = ExtractionConfig(output_format="markdown")
    json_str = config_to_json(config)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json_str)
        path = f.name

    try:
        converter = KreuzbergConverter(config_path=path)
        built = converter._build_config()
        assert built.output_format == "markdown"
    finally:
        Path(path).unlink(missing_ok=True)


def test_raises_when_both_config_and_config_path() -> None:
    with pytest.raises(ValueError, match="Cannot specify both"):
        KreuzbergConverter(
            config=ExtractionConfig(output_format="markdown"),
            config_path="/tmp/config.json",
        )


def test_table_assembly_appends_markdown_to_content() -> None:
    table = MagicMock(spec=ExtractedTable)
    table.markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
    content = KreuzbergConverter._assemble_content("Main text", [table], "plain")
    assert content == "Main text\n\n| A | B |\n|---|---|\n| 1 | 2 |"

    # Also works with dict-style tables
    dict_table = {"markdown": "| A |\n|---|\n| 1 |", "cells": [["A"], ["1"]], "page_number": 1}
    content = KreuzbergConverter._assemble_content("Main text", [dict_table], "plain")
    assert content == "Main text\n\n| A |\n|---|\n| 1 |"


def test_table_assembly_appends_multiple_tables() -> None:
    t1 = MagicMock(spec=ExtractedTable)
    t1.markdown = "| A |\n|---|\n| 1 |"
    t2 = MagicMock(spec=ExtractedTable)
    t2.markdown = "| B |\n|---|\n| 2 |"
    content = KreuzbergConverter._assemble_content("Text", [t1, t2], "plain")
    assert content == "Text\n\n| A |\n|---|\n| 1 |\n\n| B |\n|---|\n| 2 |"


def test_table_assembly_skips_tables_with_empty_markdown() -> None:
    table = MagicMock(spec=ExtractedTable)
    table.markdown = ""
    content = KreuzbergConverter._assemble_content("Main text", [table], "plain")
    assert content == "Main text"

    # Also works with dict-style tables
    dict_table: dict[str, Any] = {"markdown": "", "cells": [], "page_number": 1}
    content = KreuzbergConverter._assemble_content("Main text", [dict_table], "plain")
    assert content == "Main text"


def test_table_assembly_no_tables_returns_text_unchanged() -> None:
    assert KreuzbergConverter._assemble_content("text", None, "plain") == "text"
    assert KreuzbergConverter._assemble_content("text", [], "plain") == "text"


def test_table_assembly_skipped_for_markdown_and_html_format() -> None:
    table = MagicMock(spec=ExtractedTable)
    table.markdown = "| A |"
    assert KreuzbergConverter._assemble_content("text", [table], "markdown") == "text"
    assert KreuzbergConverter._assemble_content("text", [table], "html") == "text"


def test_edge_empty_sources_list(sequential_converter: KreuzbergConverter) -> None:
    converter = sequential_converter
    result = converter.run(sources=[])
    assert result["documents"] == []


@patch(CONVERTER_MODULE + ".extract_file_sync")
def test_edge_sequential_extraction_error_skipped(
    mock_extract: MagicMock, sequential_converter: KreuzbergConverter, fixtures_dir: Path
) -> None:
    mock_extract.side_effect = RuntimeError("extraction failed")
    converter = sequential_converter
    result = converter.run(sources=[fixtures_dir / "sample.txt"])
    assert result["documents"] == []


def test_metadata_mock_images_excludes_binary_data(converter: KreuzbergConverter, make_mock_result) -> None:
    result = make_mock_result(
        images=[
            {
                "format": "png",
                "page_number": 1,
                "width": 200,
                "height": 100,
                "description": "chart",
                "image_index": 0,
                "data": b"binary_data_here",
            },
        ]
    )

    meta = converter._build_extraction_metadata(result)
    assert meta["image_count"] == 1
    assert meta["images"][0]["format"] == "png"
    assert meta["images"][0]["width"] == 200
    assert meta["images"][0]["height"] == 100
    assert meta["images"][0]["description"] == "chart"
    assert meta["images"][0]["image_index"] == 0
    assert meta["images"][0]["page_number"] == 1
    assert "data" not in meta["images"][0]


def test_metadata_mock_all_fields_populated(converter: KreuzbergConverter, make_mock_result) -> None:
    warning = MagicMock()
    warning.source = "parser"
    warning.message = "skipped element"
    ann = MagicMock()
    ann.annotation_type = "link"
    ann.content = "https://example.com"
    ann.page_number = 1
    result = make_mock_result(
        processing_warnings=[warning],
        images=[
            {
                "format": "jpeg",
                "page_number": 2,
                "width": 640,
                "height": 480,
                "description": None,
                "image_index": 0,
                "data": b"...",
            }
        ],
        annotations=[ann],
        quality_score=0.95,
        detected_languages=["en"],
        output_format="markdown",
        result_format="unified",
        mime_type="application/pdf",
    )

    meta = converter._build_extraction_metadata(result)
    # quality_score and output_format come from result.metadata (not top-level)
    assert meta["quality_score"] == 0.95
    assert meta["detected_languages"] == ["en"]
    assert meta["output_format"] == "markdown"
    assert meta["result_format"] == "unified"
    assert meta["mime_type"] == "application/pdf"
    assert meta["processing_warnings"] == [{"source": "parser", "message": "skipped element"}]
    assert meta["image_count"] == 1
    assert meta["annotations"] == [{"type": "link", "content": "https://example.com", "page_number": 1}]


def test_metadata_file_extensions_mock(converter: KreuzbergConverter, make_mock_result) -> None:
    result = make_mock_result(mime_type="application/pdf")

    with patch(f"{CONVERTER_MODULE}.get_extensions_for_mime", return_value=["pdf"]):
        meta = converter._build_extraction_metadata(result)

    assert meta["mime_type"] == "application/pdf"
    assert meta["file_extensions"] == ["pdf"]


def test_chunked_creates_one_document_per_chunk(converter: KreuzbergConverter) -> None:
    result = MagicMock(spec=ExtractionResult)
    chunk1 = MagicMock()
    chunk1.content = "chunk one"
    chunk1.embedding = [0.1, 0.2, 0.3]
    chunk2 = MagicMock()
    chunk2.content = "chunk two"
    chunk2.embedding = None
    result.chunks = [chunk1, chunk2]

    docs = converter._create_chunked_documents(
        result,
        base_meta={"output_format": "plain"},
        source_meta={"file_path": "test.pdf"},
        user_meta={"custom": "value"},
    )
    assert len(docs) == 2

    assert docs[0].content == "chunk one"
    assert docs[0].embedding == [0.1, 0.2, 0.3]
    assert docs[0].meta["chunk_index"] == 0
    assert docs[0].meta["total_chunks"] == 2
    assert docs[0].meta["output_format"] == "plain"
    assert docs[0].meta["file_path"] == "test.pdf"
    assert docs[0].meta["custom"] == "value"

    assert docs[1].content == "chunk two"
    assert docs[1].embedding is None
    assert docs[1].meta["chunk_index"] == 1
    assert docs[1].meta["total_chunks"] == 2

    # Single chunk boundary case
    chunk3 = MagicMock()
    chunk3.content = "only chunk"
    chunk3.embedding = None
    result.chunks = [chunk3]
    single_docs = converter._create_chunked_documents(
        result,
        base_meta={},
        source_meta={"file_path": "doc.txt"},
        user_meta={},
    )
    assert len(single_docs) == 1
    assert single_docs[0].meta["chunk_index"] == 0
    assert single_docs[0].meta["total_chunks"] == 1


def test_per_page_mock_with_object_tables(converter: KreuzbergConverter) -> None:
    result = MagicMock(spec=ExtractionResult)
    result.output_format = "plain"
    table = MagicMock(spec=ExtractedTable)
    table.markdown = "| X |\n|---|\n| 1 |"
    table.cells = [["X"], ["1"]]
    table.page_number = 1
    result.pages = [
        {
            "page_number": 1,
            "content": "Page one text",
            "is_blank": False,
            "tables": [table],
            "images": [],
        },
    ]

    docs = converter._create_per_page_documents(
        result,
        base_meta={"mime_type": "application/pdf"},
        source_meta={"file_path": "test.pdf"},
        user_meta={},
    )
    assert len(docs) == 1
    assert "| X |" in docs[0].content
    assert "Page one text" in docs[0].content
    assert docs[0].meta["page_number"] == 1


def test_per_page_mock_with_dict_tables(converter: KreuzbergConverter) -> None:
    result = MagicMock(spec=ExtractionResult)
    result.output_format = "plain"
    result.pages = [
        {
            "page_number": 1,
            "content": "Page text",
            "is_blank": False,
            "tables": [
                {"markdown": "| Y |\n|---|\n| 2 |", "cells": [["Y"], ["2"]], "page_number": 1},
            ],
            "images": [],
        },
    ]

    docs = converter._create_per_page_documents(
        result,
        base_meta={},
        source_meta={"file_path": "test.pdf"},
        user_meta={},
    )
    assert "| Y |" in docs[0].content


def test_per_page_mock_with_images(converter: KreuzbergConverter) -> None:
    result = MagicMock(spec=ExtractionResult)
    result.output_format = "plain"
    result.pages = [
        {
            "page_number": 1,
            "content": "Page with image",
            "is_blank": False,
            "tables": [],
            "images": [
                {"format": "jpeg", "page_number": 1, "width": 640, "height": 480, "data": b"img_bytes"},
            ],
        },
    ]

    docs = converter._create_per_page_documents(
        result,
        base_meta={},
        source_meta={"file_path": "test.pdf"},
        user_meta={},
    )
    assert docs[0].meta["image_count"] == 1
    images = docs[0].meta["images"]
    assert images[0]["format"] == "jpeg"
    assert images[0]["width"] == 640
    assert "data" not in images[0]


def test_deepcopy_per_page_nested_meta_not_shared(converter: KreuzbergConverter) -> None:
    result = MagicMock(spec=ExtractionResult)
    result.output_format = "plain"
    result.pages = [
        {"page_number": 1, "content": "Page 1", "is_blank": False, "tables": [], "images": []},
        {"page_number": 2, "content": "Page 2", "is_blank": False, "tables": [], "images": []},
    ]

    user_meta = {"tags": ["original"]}
    docs = converter._create_per_page_documents(
        result,
        base_meta={},
        source_meta={"file_path": "test.pdf"},
        user_meta=user_meta,
    )
    assert len(docs) == 2

    # Mutate one document's nested meta
    docs[0].meta["tags"].append("mutated")

    # Other document and original must be unaffected
    assert docs[1].meta["tags"] == ["original"]
    assert user_meta["tags"] == ["original"]


def test_deepcopy_chunked_nested_meta_not_shared(converter: KreuzbergConverter) -> None:
    result = MagicMock(spec=ExtractionResult)
    chunk1 = MagicMock()
    chunk1.content = "chunk one"
    chunk1.embedding = None
    chunk2 = MagicMock()
    chunk2.content = "chunk two"
    chunk2.embedding = None
    result.chunks = [chunk1, chunk2]

    user_meta = {"tags": ["original"]}
    docs = converter._create_chunked_documents(
        result,
        base_meta={},
        source_meta={"file_path": "test.pdf"},
        user_meta=user_meta,
    )
    assert len(docs) == 2

    docs[0].meta["tags"].append("mutated")

    assert docs[1].meta["tags"] == ["original"]
    assert user_meta["tags"] == ["original"]


def test_deepcopy_unified_nested_meta_not_shared(converter: KreuzbergConverter, make_mock_result) -> None:
    result = make_mock_result(content="hello")

    bytestream = MagicMock()
    bytestream.meta = {"file_path": "test.txt"}

    user_meta = {"tags": ["original"]}
    docs = converter._create_documents(result, bytestream, user_meta)
    assert len(docs) == 1

    docs[0].meta["tags"].append("mutated")

    assert user_meta["tags"] == ["original"]


def test_helper_serialize_warnings_with_dicts() -> None:
    warnings = [{"source": "ocr", "message": "low confidence"}]
    result = _serialize_warnings(warnings)
    assert result == [{"source": "ocr", "message": "low confidence"}]


def test_helper_serialize_warnings_with_objects() -> None:
    w = MagicMock()
    w.source = "parser"
    w.message = "skipped element"
    result = _serialize_warnings([w])
    assert result == [{"source": "parser", "message": "skipped element"}]


def test_helper_serialize_keywords() -> None:
    k = MagicMock(spec=["text", "score", "algorithm", "positions"])
    k.text = "haystack"
    k.score = 0.87
    k.algorithm = "yake"
    k.positions = [(0, 8), (42, 50)]
    assert _serialize_keywords([k]) == [
        {"text": "haystack", "score": 0.87, "algorithm": "yake", "positions": [(0, 8), (42, 50)]}
    ]


def test_helper_serialize_keywords_with_none_positions() -> None:
    k = MagicMock(spec=["text", "score", "algorithm", "positions"])
    k.text = "kreuzberg"
    k.score = 0.5
    k.algorithm = "yake"
    k.positions = None
    assert _serialize_keywords([k]) == [{"text": "kreuzberg", "score": 0.5, "algorithm": "yake", "positions": None}]


def test_build_config_skips_auto_language_detection_when_already_set() -> None:
    config = ExtractionConfig(language_detection=LanguageDetectionConfig(enabled=False))
    converter = KreuzbergConverter(config=config)
    built = converter._build_config()
    assert built.language_detection.enabled is False


@patch(CONVERTER_MODULE + ".extract_file_sync")
def test_run_expands_directory_sources(mock_extract: MagicMock, tmp_path: Path, make_mock_result) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    mock_extract.return_value = make_mock_result(content="text")

    converter = KreuzbergConverter(batch=False)
    result = converter.run(sources=[tmp_path])

    assert len(result["documents"]) == 2
    assert mock_extract.call_count == 2


@patch(CONVERTER_MODULE + ".extract_bytes_sync")
@patch(CONVERTER_MODULE + ".detect_mime_type")
def test_extract_sequential_with_bytestream(
    mock_detect: MagicMock, mock_extract: MagicMock, converter: KreuzbergConverter, make_mock_result
) -> None:
    mock_detect.return_value = "application/octet-stream"
    mock_result = make_mock_result()
    mock_extract.return_value = mock_result

    bs = ByteStream(data=b"hello world")
    mock_bytestream = MagicMock()
    mock_bytestream.meta = {}

    results = converter._extract_sequential([bs], [mock_bytestream], ExtractionConfig())

    assert len(results) == 1
    assert results[0] is mock_result
    mock_extract.assert_called_once()
    assert mock_extract.call_args[0][0] == b"hello world"


@patch(CONVERTER_MODULE + ".batch_extract_bytes_sync")
@patch(CONVERTER_MODULE + ".batch_extract_files_sync")
def test_extract_batch_dispatches_by_source_type(
    mock_batch_files: MagicMock, mock_batch_bytes: MagicMock, converter: KreuzbergConverter, make_mock_result
) -> None:
    file_result = make_mock_result(content="from file")
    bytes_result = make_mock_result(content="from bytes")
    mock_batch_files.return_value = [file_result]
    mock_batch_bytes.return_value = [bytes_result]

    bs = ByteStream(data=b"hello", mime_type="text/plain")

    results = converter._extract_batch([Path("a.pdf"), bs], ExtractionConfig())

    assert results[0] is file_result
    assert results[1] is bytes_result
    mock_batch_files.assert_called_once()
    mock_batch_bytes.assert_called_once()


def test_metadata_file_extensions_edge_cases(converter: KreuzbergConverter, make_mock_result) -> None:
    # RuntimeError from get_extensions_for_mime is handled gracefully
    result = make_mock_result(mime_type="application/pdf")
    with patch(f"{CONVERTER_MODULE}.get_extensions_for_mime", side_effect=RuntimeError("unknown mime")):
        meta = converter._build_extraction_metadata(result)
    assert "file_extensions" not in meta
    assert meta["mime_type"] == "application/pdf"

    # Unknown MIME type yields no file_extensions key
    result2 = make_mock_result(mime_type="application/x-unknown-format")
    meta2 = converter._build_extraction_metadata(result2)
    assert meta2["mime_type"] == "application/x-unknown-format"
    assert "file_extensions" not in meta2


def test_create_documents_dispatches_to_chunked(converter: KreuzbergConverter, make_mock_result) -> None:
    chunk = MagicMock()
    chunk.content = "chunk text"
    chunk.embedding = None
    result = make_mock_result(chunks=[chunk], pages=None)

    bytestream = MagicMock()
    bytestream.meta = {"file_path": "test.txt"}

    with patch.object(converter, "_create_chunked_documents", wraps=converter._create_chunked_documents) as spy:
        converter._create_documents(result, bytestream, {})
        spy.assert_called_once()


def test_create_documents_dispatches_to_per_page(converter: KreuzbergConverter, make_mock_result) -> None:
    page = {"page_number": 1, "content": "page text", "is_blank": False, "tables": [], "images": []}
    result = make_mock_result(pages=[page], chunks=None)

    bytestream = MagicMock()
    bytestream.meta = {"file_path": "test.txt"}

    with patch.object(converter, "_create_per_page_documents", wraps=converter._create_per_page_documents) as spy:
        converter._create_documents(result, bytestream, {})
        spy.assert_called_once()


def test_run_raises_for_directory_with_list_meta(converter: KreuzbergConverter, tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        converter.run(sources=[str(tmp_path)], meta=[{}, {}])


@patch(CONVERTER_MODULE + ".get_bytestream_from_source")
def test_run_uses_batch_path_for_multiple_sources(mock_get_bs: MagicMock, converter: KreuzbergConverter) -> None:
    mock_get_bs.return_value = MagicMock(meta={})
    with patch.object(converter, "_extract_batch", return_value=[None, None]) as mock_batch:
        result = converter.run(sources=["a.pdf", "b.pdf"])
    mock_batch.assert_called_once()
    assert result == {"documents": []}


@patch(CONVERTER_MODULE + ".get_bytestream_from_source")
def test_run_sequential_skips_unreadable_sources(mock_get_bs: MagicMock) -> None:
    mock_get_bs.side_effect = Exception("read error")
    converter = KreuzbergConverter(batch=False)
    result = converter.run(sources=[Path("nonexistent.txt")])
    assert result == {"documents": []}


def test_extract_sequential_skips_none_bytestreams() -> None:
    converter = KreuzbergConverter(batch=False)
    results = converter._extract_sequential([Path("test.txt")], [None], ExtractionConfig())

    assert results == [None]


@patch(CONVERTER_MODULE + ".extract_file_sync")
def test_extract_sequential_success_path(mock_extract: MagicMock, make_mock_result) -> None:
    mock_extract.return_value = make_mock_result()
    mock_bytestream = MagicMock()
    mock_bytestream.meta = {"file_path": "test.txt"}

    converter = KreuzbergConverter(batch=False)
    results = converter._extract_sequential([Path("test.txt")], [mock_bytestream], ExtractionConfig())

    assert len(results) == 1
    assert results[0] is not None


@patch(CONVERTER_MODULE + ".get_bytestream_from_source")
def test_run_batch_skips_failed_sources(
    mock_get_bs: MagicMock, converter: KreuzbergConverter, make_mock_result
) -> None:
    mock_bytestream = MagicMock()
    mock_bytestream.meta = {"file_path": "b.pdf"}
    mock_get_bs.side_effect = [Exception("read error"), mock_bytestream]

    mock_result = make_mock_result()

    with patch.object(converter, "_extract_batch", return_value=[None, mock_result]):
        result = converter.run(sources=[Path("a.pdf"), Path("b.pdf")])

    assert len(result["documents"]) == 1


@patch(CONVERTER_MODULE + ".get_bytestream_from_source")
def test_run_batch_success(mock_get_bs: MagicMock, converter: KreuzbergConverter, make_mock_result) -> None:
    mock_bytestream = MagicMock()
    mock_bytestream.meta = {"file_path": "test.pdf"}
    mock_get_bs.return_value = mock_bytestream

    mock_result = make_mock_result()

    with patch.object(converter, "_extract_batch", return_value=[mock_result, mock_result]):
        result = converter.run(sources=[Path("a.pdf"), Path("b.pdf")])

    assert len(result["documents"]) == 2


def test_is_batch_error_detects_error_result() -> None:
    """Batch error results have empty metadata and None quality_score."""
    result = MagicMock(spec=ExtractionResult)
    result.metadata = {}
    result.quality_score = None
    result.content = "Error: could not parse file"

    assert _is_batch_error(result) is True


def test_is_batch_error_passes_valid_result() -> None:
    """Valid results have populated metadata and are not flagged."""
    result = MagicMock(spec=ExtractionResult)
    result.metadata = {"output_format": "plain", "quality_score": 0.85}
    result.quality_score = 0.85
    result.content = "Hello world"

    assert _is_batch_error(result) is False


def test_is_batch_error_no_false_positive_on_error_content() -> None:
    """Content starting with 'Error:' but valid metadata is NOT a batch error."""
    result = MagicMock(spec=ExtractionResult)
    result.metadata = {"output_format": "plain"}
    result.quality_score = 0.5
    result.content = "Error: this is actual document content that starts with Error:"

    assert _is_batch_error(result) is False
