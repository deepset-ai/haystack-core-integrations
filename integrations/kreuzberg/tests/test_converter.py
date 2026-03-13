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
from haystack_integrations.components.converters.kreuzberg.converter import (
    _serialize_annotations,
    _serialize_keywords,
    _serialize_page_tables,
    _serialize_tables,
    _serialize_warnings,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CONVERTER_MODULE = "haystack_integrations.components.converters.kreuzberg.converter"


@pytest.mark.unit
def test_init_default() -> None:
    converter = KreuzbergConverter()
    assert converter.config is None
    assert converter.config_path is None
    assert converter.store_full_path is False
    assert converter.batch is True
    assert converter.easyocr_kwargs is None


@pytest.mark.unit
def test_init_with_all_params() -> None:
    config = ExtractionConfig(output_format="markdown")
    converter = KreuzbergConverter(
        config=config,
        config_path="/tmp/config.json",
        store_full_path=True,
        batch=False,
        easyocr_kwargs={"gpu": False},
    )
    assert converter.config is config
    assert converter.config_path == "/tmp/config.json"
    assert converter.store_full_path is True
    assert converter.batch is False
    assert converter.easyocr_kwargs == {"gpu": False}


@pytest.mark.unit
def test_serialization_to_dict_default() -> None:
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


@pytest.mark.unit
def test_serialization_to_dict_with_config() -> None:
    config = ExtractionConfig(output_format="markdown")
    converter = KreuzbergConverter(config=config)
    d = converter.to_dict()
    # config should be serialized as JSON string
    config_value = d["init_parameters"]["config"]
    assert isinstance(config_value, str)
    parsed = json.loads(config_value)
    assert parsed["output_format"] == "markdown"


@pytest.mark.unit
def test_serialization_from_dict_default() -> None:
    d = {
        "type": "haystack_integrations.components.converters.kreuzberg.converter.KreuzbergConverter",
        "init_parameters": {
            "config": None,
            "store_full_path": True,
        },
    }
    converter = KreuzbergConverter.from_dict(d)
    assert converter.config is None
    assert converter.store_full_path is True


@pytest.mark.unit
def test_serialization_roundtrip_with_config() -> None:
    config = ExtractionConfig(
        output_format="markdown",
        ocr=OcrConfig(backend="tesseract", language="eng"),
    )
    converter = KreuzbergConverter(config=config)
    d = converter.to_dict()
    restored = KreuzbergConverter.from_dict(d)
    assert restored.config is not None
    assert restored.config.output_format == "markdown"
    assert restored.config.ocr is not None
    assert restored.config.ocr.backend == "tesseract"
    assert restored.config.ocr.language == "eng"


@pytest.mark.unit
def test_serialization_to_dict_all_non_default_params() -> None:
    config = ExtractionConfig(output_format="html")
    converter = KreuzbergConverter(
        config=config,
        config_path="/tmp/config.toml",
        store_full_path=True,
        batch=False,
        easyocr_kwargs={"gpu": True, "beam_width": 10},
    )
    d = converter.to_dict()
    params = d["init_parameters"]
    assert isinstance(params["config"], str)
    assert json.loads(params["config"])["output_format"] == "html"
    assert params["config_path"] == "/tmp/config.toml"
    assert params["store_full_path"] is True
    assert params["batch"] is False
    assert params["easyocr_kwargs"] == {"gpu": True, "beam_width": 10}


@pytest.mark.unit
def test_serialization_to_dict_with_config_path() -> None:
    converter = KreuzbergConverter(config_path="/some/path/config.yaml")
    d = converter.to_dict()
    assert d["init_parameters"]["config_path"] == "/some/path/config.yaml"


@pytest.mark.unit
def test_serialization_to_dict_config_path_from_path_object() -> None:
    converter = KreuzbergConverter(config_path=Path("/some/path/config.json"))
    d = converter.to_dict()
    # Path should be stored as str
    assert d["init_parameters"]["config_path"] == "/some/path/config.json"
    assert isinstance(d["init_parameters"]["config_path"], str)


@pytest.mark.unit
def test_serialization_from_dict_with_config_json_string() -> None:
    config_json = config_to_json(ExtractionConfig(output_format="markdown"))
    d = {
        "type": "haystack_integrations.components.converters.kreuzberg.converter.KreuzbergConverter",
        "init_parameters": {
            "config": config_json,
        },
    }
    converter = KreuzbergConverter.from_dict(d)
    assert converter.config is not None
    assert converter.config.output_format == "markdown"


@pytest.mark.unit
def test_serialization_from_dict_all_params() -> None:
    config_json = config_to_json(ExtractionConfig(output_format="html", ocr=OcrConfig(backend="tesseract")))
    d = {
        "type": "haystack_integrations.components.converters.kreuzberg.converter.KreuzbergConverter",
        "init_parameters": {
            "config": config_json,
            "config_path": "/tmp/config.toml",
            "store_full_path": True,
            "batch": False,
            "easyocr_kwargs": {"gpu": False},
        },
    }
    converter = KreuzbergConverter.from_dict(d)
    assert converter.config is not None
    assert converter.config.output_format == "html"
    assert converter.config.ocr.backend == "tesseract"
    assert converter.config_path == "/tmp/config.toml"
    assert converter.store_full_path is True
    assert converter.batch is False
    assert converter.easyocr_kwargs == {"gpu": False}


@pytest.mark.unit
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


@pytest.mark.unit
def test_serialization_roundtrip_easyocr_kwargs() -> None:
    converter = KreuzbergConverter(easyocr_kwargs={"gpu": False, "beam_width": 5})
    d = converter.to_dict()
    restored = KreuzbergConverter.from_dict(d)
    assert restored.easyocr_kwargs == {"gpu": False, "beam_width": 5}


@pytest.mark.unit
def test_serialization_roundtrip_config_path() -> None:
    converter = KreuzbergConverter(config_path="/tmp/kreuzberg.toml")
    d = converter.to_dict()
    restored = KreuzbergConverter.from_dict(d)
    assert restored.config_path == "/tmp/kreuzberg.toml"


@pytest.mark.unit
def test_serialization_roundtrip_all_non_default_params() -> None:
    converter = KreuzbergConverter(
        config=ExtractionConfig(
            output_format="html",
            ocr=OcrConfig(backend="tesseract", language="deu"),
        ),
        config_path="/tmp/fallback.yaml",
        store_full_path=True,
        batch=False,
        easyocr_kwargs={"gpu": True, "beam_width": 3},
    )
    d = converter.to_dict()
    restored = KreuzbergConverter.from_dict(d)
    assert restored.config.output_format == "html"
    assert restored.config.ocr.backend == "tesseract"
    assert restored.config.ocr.language == "deu"
    assert restored.config_path == "/tmp/fallback.yaml"
    assert restored.store_full_path is True
    assert restored.batch is False
    assert restored.easyocr_kwargs == {"gpu": True, "beam_width": 3}


@pytest.mark.unit
def test_serialization_roundtrip_preserves_to_dict_equality() -> None:
    converter = KreuzbergConverter(
        config=ExtractionConfig(output_format="markdown"),
        store_full_path=True,
        batch=False,
    )
    d1 = converter.to_dict()
    restored = KreuzbergConverter.from_dict(d1)
    d2 = restored.to_dict()
    # Compare non-config params directly
    p1 = {k: v for k, v in d1["init_parameters"].items() if k != "config"}
    p2 = {k: v for k, v in d2["init_parameters"].items() if k != "config"}
    assert p1 == p2
    # Compare config semantically (JSON round-trip may differ in repr)
    c1 = d1["init_parameters"]["config"]
    c2 = d2["init_parameters"]["config"]
    assert json.loads(c1 if isinstance(c1, str) else config_to_json(c1)) == json.loads(
        c2 if isinstance(c2, str) else config_to_json(c2)
    )


@pytest.mark.unit
def test_build_config_default() -> None:
    converter = KreuzbergConverter()
    config = converter._build_config()
    assert config.output_format == "plain"
    assert config.language_detection is not None
    assert config.language_detection.enabled is True


@pytest.mark.unit
def test_build_config_does_not_mutate_self_config() -> None:
    base = ExtractionConfig(output_format="html")
    converter = KreuzbergConverter(config=base)
    converter._build_config()
    assert base.output_format == "html"


@pytest.mark.unit
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


@pytest.mark.unit
def test_build_config_merges_config_and_config_path() -> None:
    file_config = ExtractionConfig(output_format="html")
    json_str = config_to_json(file_config)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json_str)
        path = f.name

    try:
        converter = KreuzbergConverter(
            config=ExtractionConfig(output_format="markdown"),
            config_path=path,
        )
        built = converter._build_config()
        # Explicit config takes priority over file config
        assert built.output_format == "markdown"
        # Language detection auto-enabled
        assert built.language_detection is not None
        assert built.language_detection.enabled is True
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.unit
def test_table_assembly_appends_markdown_to_content() -> None:
    table = MagicMock(spec=ExtractedTable)
    table.markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
    content = KreuzbergConverter._assemble_content("Main text", [table], "plain")
    assert content == "Main text\n\n| A | B |\n|---|---|\n| 1 | 2 |"


@pytest.mark.unit
def test_table_assembly_appends_multiple_tables() -> None:
    t1 = MagicMock(spec=ExtractedTable)
    t1.markdown = "| A |\n|---|\n| 1 |"
    t2 = MagicMock(spec=ExtractedTable)
    t2.markdown = "| B |\n|---|\n| 2 |"
    content = KreuzbergConverter._assemble_content("Text", [t1, t2], "plain")
    assert content == "Text\n\n| A |\n|---|\n| 1 |\n\n| B |\n|---|\n| 2 |"


@pytest.mark.unit
def test_table_assembly_skips_tables_with_empty_markdown() -> None:
    table = MagicMock(spec=ExtractedTable)
    table.markdown = ""
    content = KreuzbergConverter._assemble_content("Main text", [table], "plain")
    assert content == "Main text"


@pytest.mark.unit
def test_table_assembly_no_tables_returns_text_unchanged() -> None:
    assert KreuzbergConverter._assemble_content("text", None, "plain") == "text"
    assert KreuzbergConverter._assemble_content("text", [], "plain") == "text"


@pytest.mark.unit
def test_table_assembly_skipped_for_markdown_format() -> None:
    table = MagicMock(spec=ExtractedTable)
    table.markdown = "| A |"
    assert KreuzbergConverter._assemble_content("text", [table], "markdown") == "text"


@pytest.mark.unit
def test_table_assembly_skipped_for_html_format() -> None:
    table = MagicMock(spec=ExtractedTable)
    table.markdown = "| A |"
    assert KreuzbergConverter._assemble_content("text", [table], "html") == "text"


@pytest.mark.unit
def test_introspection_supported_extractors() -> None:
    extractors = KreuzbergConverter.supported_extractors()
    assert isinstance(extractors, list)


@pytest.mark.unit
def test_introspection_supported_ocr_backends() -> None:
    backends = KreuzbergConverter.supported_ocr_backends()
    assert isinstance(backends, list)
    assert len(backends) > 0


@pytest.mark.unit
def test_edge_empty_sources_list(sequential_converter: KreuzbergConverter) -> None:
    converter = sequential_converter
    result = converter.run(sources=[])
    assert result["documents"] == []
    assert result["raw_extraction"] == []


@pytest.mark.unit
@patch(CONVERTER_MODULE + ".extract_file_sync")
def test_edge_sequential_extraction_error_skipped(
    mock_extract: MagicMock, sequential_converter: KreuzbergConverter
) -> None:
    mock_extract.side_effect = RuntimeError("extraction failed")
    converter = sequential_converter
    result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
    assert result["documents"] == []
    assert result["raw_extraction"] == []


def _make_mock_result(**overrides: Any) -> MagicMock:
    """Create a mock ExtractionResult with realistic defaults.

    Fields that are never ``None`` at runtime (``metadata``, ``tables``,
    ``processing_warnings``, ``output_format``, ``result_format``,
    ``mime_type``) use their actual default values.  Nullable fields
    default to ``None``.
    """
    result = MagicMock(spec=ExtractionResult)
    defaults: dict[str, Any] = {
        "content": "",
        "metadata": {},
        "quality_score": None,
        "processing_warnings": [],
        "detected_languages": None,
        "extracted_keywords": None,
        "output_format": "plain",
        "result_format": "unified",
        "mime_type": "text/plain",
        "tables": [],
        "images": None,
        "annotations": None,
        "pages": None,
        "chunks": None,
    }
    for key, default in defaults.items():
        setattr(result, key, overrides.get(key, default))
    return result


@pytest.mark.unit
def test_metadata_mock_processing_warnings() -> None:
    warning = MagicMock()
    warning.source = "ocr"
    warning.message = "low confidence"
    result = _make_mock_result(processing_warnings=[warning])

    converter = KreuzbergConverter()
    meta = converter._build_extraction_metadata(result)
    assert meta["processing_warnings"] == [{"source": "ocr", "message": "low confidence"}]


@pytest.mark.unit
def test_metadata_mock_images_excludes_binary_data() -> None:
    result = _make_mock_result(
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

    converter = KreuzbergConverter()
    meta = converter._build_extraction_metadata(result)
    assert meta["image_count"] == 1
    assert meta["images"][0]["format"] == "png"
    assert meta["images"][0]["width"] == 200
    assert meta["images"][0]["height"] == 100
    assert meta["images"][0]["description"] == "chart"
    assert meta["images"][0]["image_index"] == 0
    assert meta["images"][0]["page_number"] == 1
    assert "data" not in meta["images"][0]


@pytest.mark.unit
def test_metadata_mock_annotations() -> None:
    ann = MagicMock()
    ann.annotation_type = "highlight"
    ann.content = "important text"
    ann.page_number = 3
    result = _make_mock_result(annotations=[ann])

    converter = KreuzbergConverter()
    meta = converter._build_extraction_metadata(result)
    assert meta["annotations"] == [
        {"type": "highlight", "content": "important text", "page_number": 3},
    ]


@pytest.mark.unit
def test_metadata_mock_tables() -> None:
    table = MagicMock(spec=ExtractedTable)
    table.cells = [["A", "B"], ["1", "2"]]
    table.markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
    table.page_number = 1
    result = _make_mock_result(tables=[table])

    converter = KreuzbergConverter()
    meta = converter._build_extraction_metadata(result)
    assert meta["table_count"] == 1
    assert meta["tables"][0]["cells"] == [["A", "B"], ["1", "2"]]
    assert meta["tables"][0]["markdown"] == "| A | B |\n|---|---|\n| 1 | 2 |"
    assert meta["tables"][0]["page_number"] == 1


@pytest.mark.unit
def test_metadata_mock_all_fields_populated() -> None:
    warning = MagicMock()
    warning.source = "parser"
    warning.message = "skipped element"
    ann = MagicMock()
    ann.annotation_type = "link"
    ann.content = "https://example.com"
    ann.page_number = 1
    result = _make_mock_result(
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

    converter = KreuzbergConverter()
    meta = converter._build_extraction_metadata(result)
    assert meta["quality_score"] == 0.95
    assert meta["detected_languages"] == ["en"]
    assert meta["output_format"] == "markdown"
    assert meta["result_format"] == "unified"
    assert meta["mime_type"] == "application/pdf"
    assert meta["processing_warnings"][0]["source"] == "parser"
    assert meta["image_count"] == 1
    assert meta["annotations"][0]["type"] == "link"


@pytest.mark.unit
def test_metadata_file_extensions_mock() -> None:
    result = _make_mock_result(mime_type="application/pdf")
    converter = KreuzbergConverter()

    with patch(f"{CONVERTER_MODULE}.get_extensions_for_mime", return_value=["pdf"]):
        meta = converter._build_extraction_metadata(result)

    assert meta["mime_type"] == "application/pdf"
    assert meta["file_extensions"] == ["pdf"]


@pytest.mark.unit
def test_metadata_no_file_extensions_for_unknown_mime() -> None:
    result = _make_mock_result(mime_type="application/x-unknown-format")
    converter = KreuzbergConverter()
    meta = converter._build_extraction_metadata(result)

    assert meta["mime_type"] == "application/x-unknown-format"
    assert "file_extensions" not in meta


@pytest.mark.unit
def test_chunked_creates_one_document_per_chunk() -> None:
    converter = KreuzbergConverter()
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


@pytest.mark.unit
def test_chunked_single_chunk() -> None:
    converter = KreuzbergConverter()
    result = MagicMock(spec=ExtractionResult)
    chunk = MagicMock()
    chunk.content = "only chunk"
    chunk.embedding = None
    result.chunks = [chunk]

    docs = converter._create_chunked_documents(
        result,
        base_meta={},
        source_meta={"file_path": "doc.txt"},
        user_meta={},
    )
    assert len(docs) == 1
    assert docs[0].meta["chunk_index"] == 0
    assert docs[0].meta["total_chunks"] == 1


@pytest.mark.unit
def test_per_page_mock_with_object_tables() -> None:
    converter = KreuzbergConverter()
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
    assert docs[0].meta["table_count"] == 1
    assert docs[0].meta["page_number"] == 1


@pytest.mark.unit
def test_per_page_mock_with_dict_tables() -> None:
    converter = KreuzbergConverter()
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
    assert docs[0].meta["table_count"] == 1
    assert docs[0].meta["tables"][0]["cells"] == [["Y"], ["2"]]


@pytest.mark.unit
def test_per_page_mock_with_images() -> None:
    converter = KreuzbergConverter()
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


@pytest.mark.unit
def test_per_page_mock_without_tables_removes_document_level_table_meta() -> None:
    converter = KreuzbergConverter()
    result = MagicMock(spec=ExtractionResult)
    result.output_format = "plain"
    result.pages = [
        {
            "page_number": 1,
            "content": "No tables here",
            "is_blank": False,
            "tables": [],
            "images": [],
        },
    ]

    docs = converter._create_per_page_documents(
        result,
        base_meta={"table_count": 5, "tables": [{"markdown": "..."}]},
        source_meta={"file_path": "test.pdf"},
        user_meta={},
    )
    # Document-level table info should be removed for pages without tables
    assert "table_count" not in docs[0].meta
    assert "tables" not in docs[0].meta


@pytest.mark.unit
def test_deepcopy_per_page_nested_meta_not_shared() -> None:
    converter = KreuzbergConverter()
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


@pytest.mark.unit
def test_deepcopy_chunked_nested_meta_not_shared() -> None:
    converter = KreuzbergConverter()
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


@pytest.mark.unit
def test_deepcopy_unified_nested_meta_not_shared() -> None:
    converter = KreuzbergConverter()
    result = _make_mock_result(content="hello")

    bytestream = MagicMock()
    bytestream.meta = {"file_path": "test.txt"}

    user_meta = {"tags": ["original"]}
    docs = converter._create_documents(result, bytestream, user_meta)
    assert len(docs) == 1

    docs[0].meta["tags"].append("mutated")

    assert user_meta["tags"] == ["original"]


@pytest.mark.unit
def test_helper_serialize_page_tables_with_dicts() -> None:
    tables = [{"cells": [["A"], ["1"]], "markdown": "| A |", "page_number": 1}]
    result = _serialize_page_tables(tables)
    assert result == [{"cells": [["A"], ["1"]], "markdown": "| A |", "page_number": 1}]


@pytest.mark.unit
def test_helper_serialize_page_tables_with_objects() -> None:
    table = MagicMock(spec=ExtractedTable)
    table.cells = [["B"], ["2"]]
    table.markdown = "| B |"
    table.page_number = 2
    result = _serialize_page_tables([table])
    assert result == [{"cells": [["B"], ["2"]], "markdown": "| B |", "page_number": 2}]


@pytest.mark.unit
def test_helper_serialize_warnings_with_dicts() -> None:
    warnings = [{"source": "ocr", "message": "low confidence"}]
    result = _serialize_warnings(warnings)
    assert result == [{"source": "ocr", "message": "low confidence"}]


@pytest.mark.unit
def test_helper_serialize_warnings_with_objects() -> None:
    w = MagicMock()
    w.source = "parser"
    w.message = "skipped element"
    result = _serialize_warnings([w])
    assert result == [{"source": "parser", "message": "skipped element"}]


@pytest.mark.unit
def test_table_assembly_with_dict_tables() -> None:
    tables = [{"markdown": "| A |\n|---|\n| 1 |", "cells": [["A"], ["1"]], "page_number": 1}]
    content = KreuzbergConverter._assemble_content("Main text", tables, "plain")
    assert content == "Main text\n\n| A |\n|---|\n| 1 |"


@pytest.mark.unit
def test_table_assembly_with_dict_tables_empty_markdown() -> None:
    tables: list[dict[str, Any]] = [{"markdown": "", "cells": [], "page_number": 1}]
    content = KreuzbergConverter._assemble_content("Main text", tables, "plain")
    assert content == "Main text"


@pytest.mark.unit
def test_helper_serialize_tables_with_objects() -> None:
    table = MagicMock(spec=ExtractedTable)
    table.cells = [["A", "B"], ["1", "2"]]
    table.markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
    table.page_number = 1
    result = _serialize_tables([table])
    assert result == [
        {
            "cells": [["A", "B"], ["1", "2"]],
            "markdown": "| A | B |\n|---|---|\n| 1 | 2 |",
            "page_number": 1,
        }
    ]


@pytest.mark.unit
def test_serialize_keywords() -> None:
    kw1 = MagicMock()
    kw1.text = "machine learning"
    kw1.score = 0.95
    kw1.algorithm = "tfidf"
    kw2 = MagicMock()
    kw2.text = "neural network"
    kw2.score = 0.87
    kw2.algorithm = "yake"

    result = _serialize_keywords([kw1, kw2])

    assert result == [
        {"text": "machine learning", "score": 0.95, "algorithm": "tfidf"},
        {"text": "neural network", "score": 0.87, "algorithm": "yake"},
    ]


@pytest.mark.unit
def test_serialize_annotations_with_objects() -> None:
    ann = MagicMock()
    ann.annotation_type = "highlight"
    ann.content = "important text"
    ann.page_number = 3

    result = _serialize_annotations([ann])

    assert result == [{"type": "highlight", "content": "important text", "page_number": 3}]


@pytest.mark.unit
def test_build_config_skips_auto_language_detection_when_already_set() -> None:
    config = ExtractionConfig(language_detection=LanguageDetectionConfig(enabled=False))
    converter = KreuzbergConverter(config=config)
    built = converter._build_config()
    assert built.language_detection.enabled is False


@pytest.mark.unit
def test_expand_sources_passthrough_bytestream() -> None:
    bs = ByteStream(data=b"hello")
    result = KreuzbergConverter._expand_sources([bs])
    assert result == [bs]


@pytest.mark.unit
def test_expand_sources_expands_directory(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")

    result = KreuzbergConverter._expand_sources([tmp_path])

    assert len(result) == 2
    assert tmp_path / "a.txt" in result
    assert tmp_path / "b.txt" in result


@pytest.mark.unit
@patch(CONVERTER_MODULE + ".extract_bytes_sync")
@patch(CONVERTER_MODULE + ".detect_mime_type")
def test_extract_single_with_bytestream(mock_detect: MagicMock, mock_extract: MagicMock) -> None:
    mock_detect.return_value = "application/octet-stream"
    mock_result = _make_mock_result()
    mock_extract.return_value = mock_result

    converter = KreuzbergConverter()
    config = ExtractionConfig()
    bs = ByteStream(data=b"hello world")  # no mime_type → detect_mime_type called

    result = converter._extract_single(bs, config)

    assert result is mock_result
    assert mock_extract.call_args[0][0] == b"hello world"


@pytest.mark.unit
@patch(CONVERTER_MODULE + ".batch_extract_files_sync")
def test_extract_batch_with_files(mock_batch_files: MagicMock) -> None:
    mock_result = _make_mock_result()
    mock_batch_files.return_value = [mock_result]

    converter = KreuzbergConverter()
    results = converter._extract_batch([Path("a.pdf")], ExtractionConfig())

    assert results[0] is mock_result
    mock_batch_files.assert_called_once()


@pytest.mark.unit
@patch(CONVERTER_MODULE + ".batch_extract_bytes_sync")
def test_extract_batch_with_bytestreams(mock_batch_bytes: MagicMock) -> None:
    mock_result = _make_mock_result()
    mock_batch_bytes.return_value = [mock_result]

    converter = KreuzbergConverter()
    bs = ByteStream(data=b"hello", mime_type="text/plain")

    results = converter._extract_batch([bs], ExtractionConfig())

    assert results[0] is mock_result
    mock_batch_bytes.assert_called_once()


@pytest.mark.unit
@patch(CONVERTER_MODULE + ".batch_extract_bytes_sync")
@patch(CONVERTER_MODULE + ".batch_extract_files_sync")
def test_extract_batch_with_mixed_sources(mock_batch_files: MagicMock, mock_batch_bytes: MagicMock) -> None:
    file_result = _make_mock_result(content="from file")
    bytes_result = _make_mock_result(content="from bytes")
    mock_batch_files.return_value = [file_result]
    mock_batch_bytes.return_value = [bytes_result]

    converter = KreuzbergConverter()
    bs = ByteStream(data=b"hello", mime_type="text/plain")

    results = converter._extract_batch([Path("a.pdf"), bs], ExtractionConfig())

    assert results[0] is file_result
    assert results[1] is bytes_result


@pytest.mark.unit
def test_metadata_detected_languages() -> None:
    result = _make_mock_result(detected_languages=["en", "de"])
    converter = KreuzbergConverter()
    meta = converter._build_extraction_metadata(result)
    assert meta["detected_languages"] == ["en", "de"]


@pytest.mark.unit
def test_metadata_extensions_runtime_error() -> None:
    result = _make_mock_result(mime_type="application/pdf")
    converter = KreuzbergConverter()

    with patch(f"{CONVERTER_MODULE}.get_extensions_for_mime", side_effect=RuntimeError("unknown mime")):
        meta = converter._build_extraction_metadata(result)

    assert "file_extensions" not in meta
    assert meta["mime_type"] == "application/pdf"


@pytest.mark.unit
def test_create_documents_dispatches_to_chunked() -> None:
    chunk = MagicMock()
    chunk.content = "chunk text"
    chunk.embedding = None
    result = _make_mock_result(chunks=[chunk], pages=None)

    bytestream = MagicMock()
    bytestream.meta = {"file_path": "test.txt"}

    converter = KreuzbergConverter()
    with patch.object(converter, "_create_chunked_documents", wraps=converter._create_chunked_documents) as spy:
        converter._create_documents(result, bytestream, {})
        spy.assert_called_once()


@pytest.mark.unit
def test_create_documents_dispatches_to_per_page() -> None:
    page = {"page_number": 1, "content": "page text", "is_blank": False, "tables": [], "images": []}
    result = _make_mock_result(pages=[page], chunks=None)

    bytestream = MagicMock()
    bytestream.meta = {"file_path": "test.txt"}

    converter = KreuzbergConverter()
    with patch.object(converter, "_create_per_page_documents", wraps=converter._create_per_page_documents) as spy:
        converter._create_documents(result, bytestream, {})
        spy.assert_called_once()


@pytest.mark.unit
def test_serialize_result_minimal() -> None:
    result = _make_mock_result()  # tables=[], quality_score=None, etc.

    raw = KreuzbergConverter._serialize_result(result)

    assert "content" in raw
    assert "mime_type" in raw
    assert "output_format" in raw
    assert "result_format" in raw
    assert "metadata" in raw
    # Optional keys absent when empty/None
    assert "tables" not in raw
    assert "quality_score" not in raw
    assert "detected_languages" not in raw
    assert "processing_warnings" not in raw
    assert "extracted_keywords" not in raw
    assert "annotations" not in raw
    assert "pages" not in raw
    assert "chunks" not in raw
    assert "images" not in raw


@pytest.mark.unit
def test_serialize_result_full() -> None:
    table = MagicMock(spec=ExtractedTable)
    table.cells = [["A"]]
    table.markdown = "| A |"
    table.page_number = 1

    warning = MagicMock()
    warning.source = "ocr"
    warning.message = "low confidence"

    kw = MagicMock()
    kw.text = "python"
    kw.score = 0.9
    kw.algorithm = "tfidf"

    ann = MagicMock()
    ann.annotation_type = "highlight"
    ann.content = "important"
    ann.page_number = 1

    chunk = MagicMock()
    chunk.content = "chunk text"
    chunk.metadata = {}

    result = _make_mock_result(
        tables=[table],
        quality_score=0.9,
        detected_languages=["en"],
        processing_warnings=[warning],
        extracted_keywords=[kw],
        annotations=[ann],
        pages=[{"page_number": 1, "content": "text", "is_blank": False, "tables": []}],
        chunks=[chunk],
        images=[{"format": "png", "width": 100, "height": 100}],
    )

    raw = KreuzbergConverter._serialize_result(result)

    assert "tables" in raw
    assert raw["quality_score"] == 0.9
    assert raw["detected_languages"] == ["en"]
    assert "processing_warnings" in raw
    assert raw["extracted_keywords"] == [{"text": "python", "score": 0.9, "algorithm": "tfidf"}]
    assert "annotations" in raw
    assert "pages" in raw
    assert raw["chunks"] == [{"content": "chunk text", "metadata": {}}]
    assert "images" in raw


@pytest.mark.unit
def test_run_raises_for_directory_with_list_meta(tmp_path: Path) -> None:
    converter = KreuzbergConverter()
    with pytest.raises(ValueError):
        converter.run(sources=[str(tmp_path)], meta=[{}, {}])


@pytest.mark.unit
def test_run_uses_batch_path_for_multiple_sources() -> None:
    converter = KreuzbergConverter(batch=True)
    with patch.object(converter, "_run_batch", return_value=([], [])) as mock_batch:
        result = converter.run(sources=["a.pdf", "b.pdf"])
    mock_batch.assert_called_once()
    assert result == {"documents": [], "raw_extraction": []}


@pytest.mark.unit
@patch(CONVERTER_MODULE + ".get_bytestream_from_source")
def test_run_sequential_skips_on_source_read_failure(mock_get_bs: MagicMock) -> None:
    mock_get_bs.side_effect = Exception("read error")

    converter = KreuzbergConverter(batch=False)
    docs, raw = converter._run_sequential([Path("nonexistent.txt")], [{}], ExtractionConfig())

    assert docs == []
    assert raw == []


@pytest.mark.unit
@patch(CONVERTER_MODULE + ".extract_file_sync")
@patch(CONVERTER_MODULE + ".get_bytestream_from_source")
def test_run_sequential_success_path(mock_get_bs: MagicMock, mock_extract: MagicMock) -> None:
    mock_bytestream = MagicMock()
    mock_bytestream.meta = {"file_path": "test.txt"}
    mock_get_bs.return_value = mock_bytestream
    mock_extract.return_value = _make_mock_result()

    converter = KreuzbergConverter(batch=False)
    docs, raw = converter._run_sequential([Path("test.txt")], [{}], ExtractionConfig())

    assert len(docs) == 1
    assert len(raw) == 1


@pytest.mark.unit
@patch(CONVERTER_MODULE + ".get_bytestream_from_source")
def test_run_batch_skips_failed_sources(mock_get_bs: MagicMock) -> None:
    mock_bytestream = MagicMock()
    mock_bytestream.meta = {"file_path": "b.pdf"}
    mock_get_bs.side_effect = [Exception("read error"), mock_bytestream]

    mock_result = _make_mock_result()
    converter = KreuzbergConverter()

    with patch.object(converter, "_extract_batch", return_value=[None, mock_result]):
        docs, raw = converter._run_batch([Path("a.pdf"), Path("b.pdf")], [{}, {}], ExtractionConfig())

    assert len(docs) == 1
    assert len(raw) == 1


@pytest.mark.unit
@patch(CONVERTER_MODULE + ".get_bytestream_from_source")
def test_run_batch_success(mock_get_bs: MagicMock) -> None:
    mock_bytestream = MagicMock()
    mock_bytestream.meta = {"file_path": "test.pdf"}
    mock_get_bs.return_value = mock_bytestream

    mock_result = _make_mock_result()
    converter = KreuzbergConverter()

    with patch.object(converter, "_extract_batch", return_value=[mock_result, mock_result]):
        docs, raw = converter._run_batch([Path("a.pdf"), Path("b.pdf")], [{}, {}], ExtractionConfig())

    assert len(docs) == 2
    assert len(raw) == 2
