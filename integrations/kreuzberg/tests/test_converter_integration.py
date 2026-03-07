# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for KreuzbergConverter.

These tests call kreuzberg's real extraction APIs against fixture files —
no mocking. Since kreuzberg is local-only (no external API), these do
not require any environment variables or skip conditions.
"""

from pathlib import Path

import pytest
from haystack.dataclasses import ByteStream, Document
from kreuzberg import ChunkingConfig, ExtractionConfig, KeywordConfig, PageConfig

from haystack_integrations.components.converters.kreuzberg import KreuzbergConverter

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _docs(result: dict) -> list[Document]:
    """Type-narrow converter output's documents to list[Document]."""
    docs = result["documents"]
    assert all(isinstance(d, Document) for d in docs)
    return docs


@pytest.mark.integration
def test_pdf_extraction() -> None:
    """PDF extraction returns a document with real text content and rich metadata."""
    converter = KreuzbergConverter()
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
    docs = _docs(result)

    assert len(docs) == 1
    doc = docs[0]
    assert "Sample PDF" in doc.content
    assert "Lorem ipsum" in doc.content
    assert doc.meta["file_path"] == "sample.pdf"
    assert doc.meta["mime_type"] == "application/pdf"
    assert doc.meta["title"] == "Sample PDF"
    assert doc.meta["authors"] == ["Philip Hutchison"]
    assert doc.meta["page_count"] == 3
    assert doc.meta["format_type"] == "pdf"
    assert doc.meta["is_encrypted"] is False
    assert doc.meta["pdf_version"] == "1.3"


@pytest.mark.integration
def test_txt_extraction() -> None:
    """TXT extraction returns content matching the source file."""
    converter = KreuzbergConverter()
    result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
    docs = _docs(result)

    assert len(docs) == 1
    assert len(result["raw_extraction"]) == 1
    assert "sample text document for testing the Kreuzberg converter" in docs[0].content
    assert "multiple paragraphs" in docs[0].content
    assert docs[0].meta["mime_type"] == "text/plain"
    assert docs[0].meta["format_type"] == "text"


@pytest.mark.integration
def test_docx_extraction() -> None:
    """DOCX extraction returns real text content."""
    converter = KreuzbergConverter()
    result = converter.run(sources=[FIXTURES_DIR / "sample.docx"])
    docs = _docs(result)

    assert len(docs) == 1
    assert "Demonstration of DOCX support" in docs[0].content
    assert docs[0].meta["mime_type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.mark.integration
def test_html_extraction() -> None:
    """HTML extraction returns text stripped of markup."""
    converter = KreuzbergConverter()
    result = converter.run(sources=[FIXTURES_DIR / "sample.html"])
    docs = _docs(result)

    assert len(docs) == 1
    assert "Sample Document" in docs[0].content
    assert "sample HTML document for testing the Kreuzberg converter" in docs[0].content
    # Should not contain raw HTML tags
    assert "<h1>" not in docs[0].content


@pytest.mark.integration
def test_multiple_mixed_sources() -> None:
    """All four fixture file types processed, each producing a document."""
    sources = [
        FIXTURES_DIR / "sample.pdf",
        FIXTURES_DIR / "sample.txt",
        FIXTURES_DIR / "sample.docx",
        FIXTURES_DIR / "sample.html",
    ]
    converter = KreuzbergConverter()
    result = converter.run(sources=sources)
    docs = _docs(result)

    assert len(docs) == 4
    assert all(doc.content for doc in docs)


@pytest.mark.integration
def test_extraction_multiple_sources(sequential_converter: KreuzbergConverter) -> None:
    """Two sources produce two documents and two raw extractions."""
    converter = sequential_converter
    result = converter.run(sources=[FIXTURES_DIR / "sample.txt", FIXTURES_DIR / "sample.pdf"])
    assert len(result["documents"]) == 2
    assert len(result["raw_extraction"]) == 2


@pytest.mark.integration
def test_extraction_with_string_path(sequential_converter: KreuzbergConverter) -> None:
    """String path (not Path object) is accepted and converted."""
    converter = sequential_converter
    result = converter.run(sources=[str(FIXTURES_DIR / "sample.txt")])

    assert len(result["documents"]) == 1
    assert _docs(result)[0].meta["file_path"] == "sample.txt"


@pytest.mark.integration
def test_bytestream_source() -> None:
    """ByteStream input with PDF bytes produces equivalent output to file path input."""
    pdf_path = FIXTURES_DIR / "sample.pdf"
    bytestream = ByteStream(data=pdf_path.read_bytes(), mime_type="application/pdf")

    converter = KreuzbergConverter()
    result = converter.run(sources=[bytestream])
    docs = _docs(result)

    assert len(docs) == 1
    assert "Sample PDF" in docs[0].content


@pytest.mark.integration
def test_bytestream_plain_text(sequential_converter: KreuzbergConverter) -> None:
    """ByteStream with plain text content is extracted correctly."""
    bs = ByteStream(data=b"Hello from ByteStream!", mime_type="text/plain")
    converter = sequential_converter
    result = converter.run(sources=[bs])

    doc = _docs(result)[0]
    assert doc.content == "Hello from ByteStream!"


@pytest.mark.integration
def test_extraction_with_bytestream_auto_detect_mime(sequential_converter: KreuzbergConverter) -> None:
    """ByteStream without explicit MIME type is auto-detected."""
    bs = ByteStream(data=b"Hello auto-detect", mime_type=None)
    converter = sequential_converter
    result = converter.run(sources=[bs])

    assert len(result["documents"]) == 1
    doc = _docs(result)[0]
    assert doc.content == "Hello auto-detect"
    assert "mime_type" in doc.meta
    assert doc.meta["mime_type"] is not None


@pytest.mark.integration
def test_metadata_populated_on_real_docs() -> None:
    """Real extraction populates mime_type, output_format, and file_path."""
    converter = KreuzbergConverter()
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
    meta = _docs(result)[0].meta

    assert "mime_type" in meta
    assert "output_format" in meta
    assert meta["file_path"] == "sample.pdf"


@pytest.mark.integration
def test_metadata_quality_score(sequential_converter: KreuzbergConverter) -> None:
    """Extraction populates a float quality_score."""
    converter = sequential_converter
    result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
    doc = _docs(result)[0]
    assert "quality_score" in doc.meta
    assert isinstance(doc.meta["quality_score"], float)


@pytest.mark.integration
def test_metadata_detected_languages(sequential_converter: KreuzbergConverter) -> None:
    """Language detection populates detected_languages list."""
    converter = sequential_converter
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
    doc = _docs(result)[0]
    assert "detected_languages" in doc.meta
    assert isinstance(doc.meta["detected_languages"], list)


@pytest.mark.integration
def test_metadata_output_format_tracking(sequential_converter: KreuzbergConverter) -> None:
    """Default extraction tracks output_format as 'plain' and result_format as 'unified'."""
    converter = sequential_converter
    result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
    doc = _docs(result)[0]
    assert doc.meta["output_format"] == "plain"
    assert doc.meta["result_format"] == "unified"


@pytest.mark.integration
def test_metadata_keyword_extraction() -> None:
    """KeywordConfig produces extracted_keywords with text, score, algorithm."""
    converter = KreuzbergConverter(
        config=ExtractionConfig(keywords=KeywordConfig(max_keywords=3)),
        batch=False,
    )
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
    doc = _docs(result)[0]
    assert "extracted_keywords" in doc.meta
    keywords = doc.meta["extracted_keywords"]
    assert len(keywords) == 3
    assert "text" in keywords[0]
    assert "score" in keywords[0]
    assert "algorithm" in keywords[0]


@pytest.mark.integration
def test_metadata_store_full_path_false() -> None:
    """store_full_path=False stores just the filename."""
    converter = KreuzbergConverter(store_full_path=False, batch=False)
    result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
    doc = _docs(result)[0]
    assert doc.meta["file_path"] == "sample.txt"


@pytest.mark.integration
def test_metadata_store_full_path_true() -> None:
    """store_full_path=True stores the absolute path."""
    converter = KreuzbergConverter(store_full_path=True, batch=False)
    result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
    doc = _docs(result)[0]
    # Full path should contain directory separators
    assert "/" in str(doc.meta["file_path"]) or "\\" in str(doc.meta["file_path"])


@pytest.mark.integration
def test_metadata_user_overrides_extraction(sequential_converter: KreuzbergConverter) -> None:
    """User-supplied metadata overrides extraction-generated metadata."""
    converter = sequential_converter
    result = converter.run(
        sources=[FIXTURES_DIR / "sample.pdf"],
        meta={"title": "User Override Title"},
    )
    doc = _docs(result)[0]
    assert doc.meta["title"] == "User Override Title"


@pytest.mark.integration
def test_metadata_file_extensions_for_pdf(sequential_converter: KreuzbergConverter) -> None:
    """PDF extraction should include file_extensions in metadata."""
    converter = sequential_converter
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])

    doc = _docs(result)[0]
    assert "file_extensions" in doc.meta
    assert "pdf" in doc.meta["file_extensions"]


@pytest.mark.integration
def test_metadata_file_extensions_for_text(sequential_converter: KreuzbergConverter) -> None:
    """Text file extraction should include file_extensions in metadata."""
    converter = sequential_converter
    result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])

    doc = _docs(result)[0]
    assert "file_extensions" in doc.meta
    assert "txt" in doc.meta["file_extensions"]


@pytest.mark.integration
def test_tables_appended_to_content_for_plain_text() -> None:
    """With default plain text output, tables are appended to content."""
    converter = KreuzbergConverter()
    result = converter.run(sources=[FIXTURES_DIR / "sample.docx"])
    doc = _docs(result)[0]

    # Table markdown should be in content
    assert "| ITEM | NEEDED |" in doc.content
    # Tables also tracked in metadata
    assert doc.meta["table_count"] == 5
    assert len(doc.meta["tables"]) == 5


@pytest.mark.integration
def test_tables_auto_inlined_for_markdown_output() -> None:
    """With markdown/html output, kreuzberg auto-inlines tables into content."""
    for fmt in ("markdown", "html"):
        converter = KreuzbergConverter(config=ExtractionConfig(output_format=fmt))
        result = converter.run(sources=[FIXTURES_DIR / "sample.docx"])
        doc = _docs(result)[0]

        # kreuzberg inlines table data into content for markdown/html
        assert "ITEM" in doc.content, f"Table content missing for {fmt}"
        # Tables also tracked in metadata
        assert doc.meta["table_count"] == 5, f"table_count wrong for {fmt}"


@pytest.mark.integration
def test_tables_in_metadata_regardless_of_format() -> None:
    """Tables are available in metadata for all output formats."""
    for fmt in ("plain", "markdown", "html"):
        converter = KreuzbergConverter(config=ExtractionConfig(output_format=fmt))
        result = converter.run(sources=[FIXTURES_DIR / "sample.docx"])
        doc = _docs(result)[0]

        assert doc.meta["table_count"] == 5, f"table_count missing for {fmt}"
        assert len(doc.meta["tables"]) == 5, f"tables list wrong for {fmt}"
        assert doc.meta["tables"][0]["markdown"], f"table markdown empty for {fmt}"


@pytest.mark.integration
def test_custom_metadata_single_dict() -> None:
    """A single metadata dict is applied to all output documents."""
    sources = [FIXTURES_DIR / "sample.pdf", FIXTURES_DIR / "sample.txt"]
    converter = KreuzbergConverter()
    result = converter.run(sources=sources, meta={"project": "haystack"})
    docs = _docs(result)

    assert len(docs) == 2
    assert all(doc.meta["project"] == "haystack" for doc in docs)


@pytest.mark.integration
def test_custom_metadata_per_source() -> None:
    """Per-source metadata list applies correct dict to each document."""
    sources = [FIXTURES_DIR / "sample.pdf", FIXTURES_DIR / "sample.txt"]
    meta = [{"idx": 0}, {"idx": 1}]
    converter = KreuzbergConverter()
    result = converter.run(sources=sources, meta=meta)
    docs = _docs(result)

    assert docs[0].meta["idx"] == 0
    assert docs[1].meta["idx"] == 1


@pytest.mark.integration
def test_per_page_extraction() -> None:
    """Per-page config on a 3-page PDF produces 3 documents."""
    config = ExtractionConfig(pages=PageConfig(extract_pages=True))
    converter = KreuzbergConverter(config=config)
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
    docs = _docs(result)

    assert len(docs) == 3
    assert all(doc.content for doc in docs)
    page_numbers = [doc.meta.get("page_number") for doc in docs]
    assert page_numbers == [1, 2, 3]


@pytest.mark.integration
def test_per_page_raw_extraction_is_one_per_source() -> None:
    """Even with per-page docs, raw_extraction has one entry per source."""
    config = ExtractionConfig(pages=PageConfig(extract_pages=True))
    converter = KreuzbergConverter(config=config)
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])

    assert len(result["documents"]) == 3
    assert len(result["raw_extraction"]) == 1


@pytest.mark.integration
def test_per_page_document_metadata(per_page_converter: KreuzbergConverter) -> None:
    """Each page document has page_number, is_blank, and file_path metadata."""
    converter = per_page_converter
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
    docs = _docs(result)

    for i, doc in enumerate(docs, start=1):
        assert doc.meta["page_number"] == i
        assert "is_blank" in doc.meta
        assert doc.meta["file_path"] == "sample.pdf"


@pytest.mark.integration
def test_per_page_with_user_metadata(per_page_converter: KreuzbergConverter) -> None:
    """User metadata is applied to all page documents."""
    converter = per_page_converter
    result = converter.run(
        sources=[FIXTURES_DIR / "sample.pdf"],
        meta={"source": "test"},
    )
    for doc in _docs(result):
        assert doc.meta["source"] == "test"


@pytest.mark.integration
def test_chunking_produces_multiple_documents() -> None:
    """Chunking config splits PDF content into multiple documents."""
    config = ExtractionConfig(chunking=ChunkingConfig(preset="sentence"))
    converter = KreuzbergConverter(config=config)
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])
    docs = _docs(result)

    assert len(docs) > 1
    assert all(doc.content for doc in docs)
    # Each chunk should have chunk_index metadata
    assert all("chunk_index" in doc.meta for doc in docs)
    assert all("total_chunks" in doc.meta for doc in docs)
    assert docs[0].meta["chunk_index"] == 0


@pytest.mark.integration
def test_batch_vs_sequential_parity() -> None:
    """Batch and sequential extraction produce equivalent content."""
    sources = [
        FIXTURES_DIR / "sample.pdf",
        FIXTURES_DIR / "sample.txt",
        FIXTURES_DIR / "sample.docx",
        FIXTURES_DIR / "sample.html",
    ]
    batch_result = KreuzbergConverter(batch=True).run(sources=sources)
    sequential_result = KreuzbergConverter(batch=False).run(sources=sources)

    batch_docs = _docs(batch_result)
    seq_docs = _docs(sequential_result)

    assert len(batch_docs) == len(seq_docs)
    for b, s in zip(batch_docs, seq_docs, strict=True):
        assert b.content == s.content


@pytest.mark.integration
def test_batch_extraction() -> None:
    """Batch mode extracts multiple sources correctly."""
    converter = KreuzbergConverter(batch=True)
    result = converter.run(sources=[FIXTURES_DIR / "sample.txt", FIXTURES_DIR / "sample.pdf"])
    assert len(result["documents"]) == 2
    assert len(result["raw_extraction"]) == 2


@pytest.mark.integration
def test_batch_single_source_uses_sequential() -> None:
    """When only one source, batch mode should use sequential extraction."""
    converter = KreuzbergConverter(batch=True)
    result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
    assert len(result["documents"]) == 1


@pytest.mark.integration
def test_batch_with_bytestream() -> None:
    """Batch mode handles mixed file and ByteStream sources."""
    bs = ByteStream(data=b"Batch bytestream", mime_type="text/plain")
    converter = KreuzbergConverter(batch=True)
    result = converter.run(sources=[FIXTURES_DIR / "sample.txt", bs])
    assert len(result["documents"]) == 2


@pytest.mark.integration
def test_batch_skips_failed_sources() -> None:
    """Batch mode skips sources that fail extraction."""
    converter = KreuzbergConverter(batch=True)
    result = converter.run(sources=["nonexistent.pdf", FIXTURES_DIR / "sample.txt"])
    # nonexistent should be skipped, sample.txt should succeed
    assert len(result["documents"]) == 1


@pytest.mark.integration
def test_directory_expansion() -> None:
    """Passing a directory processes all files inside it."""
    converter = KreuzbergConverter()
    result = converter.run(sources=[FIXTURES_DIR])
    docs = _docs(result)

    # fixtures/ contains 4 files
    assert len(docs) == 4
    assert all(doc.content for doc in docs)
    filenames = sorted(d.meta["file_path"] for d in docs)
    assert filenames == ["sample.docx", "sample.html", "sample.pdf", "sample.txt"]


@pytest.mark.integration
def test_directory_with_single_dict_meta(sequential_converter: KreuzbergConverter) -> None:
    """Single metadata dict is applied to all documents from directory expansion."""
    converter = sequential_converter
    result = converter.run(
        sources=[FIXTURES_DIR],
        meta={"source": "fixtures"},
    )
    for doc in _docs(result):
        assert doc.meta["source"] == "fixtures"


@pytest.mark.integration
def test_directory_with_list_meta_raises(sequential_converter: KreuzbergConverter) -> None:
    """Per-source metadata list raises ValueError when directories are present."""
    converter = sequential_converter
    with pytest.raises(ValueError, match="directories are present"):
        converter.run(sources=[FIXTURES_DIR], meta=[{"a": 1}])


@pytest.mark.integration
def test_directory_mixed_with_file(sequential_converter: KreuzbergConverter) -> None:
    """Directory and ByteStream sources can be mixed."""
    converter = sequential_converter
    bs = ByteStream(data=b"Extra source", mime_type="text/plain")
    result = converter.run(sources=[FIXTURES_DIR, bs])
    # 4 from fixtures dir + 1 bytestream = 5
    assert len(result["documents"]) == 5


@pytest.mark.integration
def test_raw_extraction_output() -> None:
    """raw_extraction list has one entry per source with content, mime_type, output_format, and metadata."""
    sources = [FIXTURES_DIR / "sample.pdf", FIXTURES_DIR / "sample.txt"]
    converter = KreuzbergConverter()
    result = converter.run(sources=sources)
    raw = result["raw_extraction"]

    assert len(raw) == 2
    assert all("content" in r for r in raw)
    assert all("mime_type" in r for r in raw)
    assert raw[0]["mime_type"] == "application/pdf"
    assert "output_format" in raw[0]
    assert "metadata" in raw[0]


@pytest.mark.integration
def test_raw_extraction_metadata_structure(sequential_converter: KreuzbergConverter) -> None:
    """Raw extraction metadata dict includes title and format_type."""
    converter = sequential_converter
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])

    raw_meta = result["raw_extraction"][0]["metadata"]
    assert isinstance(raw_meta, dict)
    assert "title" in raw_meta
    assert "format_type" in raw_meta


@pytest.mark.integration
def test_raw_extraction_with_keywords() -> None:
    """Raw extraction includes extracted_keywords when KeywordConfig is used."""
    converter = KreuzbergConverter(
        config=ExtractionConfig(keywords=KeywordConfig(max_keywords=3)),
        batch=False,
    )
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])

    raw = result["raw_extraction"][0]
    assert "extracted_keywords" in raw
    assert len(raw["extracted_keywords"]) == 3


@pytest.mark.integration
def test_raw_extraction_pages_when_per_page(per_page_converter: KreuzbergConverter) -> None:
    """Raw extraction includes pages list when per-page extraction is enabled."""
    converter = per_page_converter
    result = converter.run(sources=[FIXTURES_DIR / "sample.pdf"])

    raw = result["raw_extraction"][0]
    assert "pages" in raw
    assert len(raw["pages"]) == 3


@pytest.mark.integration
def test_output_format_markdown() -> None:
    """Markdown output format is tracked in document metadata."""
    converter = KreuzbergConverter(config=ExtractionConfig(output_format="markdown"), batch=False)
    result = converter.run(sources=[FIXTURES_DIR / "sample.html"])
    doc = _docs(result)[0]
    assert doc.meta["output_format"] == "markdown"


@pytest.mark.integration
def test_output_format_html() -> None:
    """HTML output format is tracked in document metadata."""
    converter = KreuzbergConverter(config=ExtractionConfig(output_format="html"), batch=False)
    result = converter.run(sources=[FIXTURES_DIR / "sample.txt"])
    doc = _docs(result)[0]
    assert doc.meta["output_format"] == "html"


@pytest.mark.integration
def test_extraction_nonexistent_file_skipped(sequential_converter: KreuzbergConverter) -> None:
    """Nonexistent file is skipped; valid source still produces output."""
    converter = sequential_converter
    result = converter.run(sources=["nonexistent.pdf", FIXTURES_DIR / "sample.txt"])
    assert len(result["documents"]) == 1
    assert _docs(result)[0].meta["file_path"] == "sample.txt"


@pytest.mark.integration
def test_edge_all_sources_fail(sequential_converter: KreuzbergConverter) -> None:
    """When all sources fail, result has empty documents and raw_extraction."""
    converter = sequential_converter
    result = converter.run(sources=["nonexistent1.pdf", "nonexistent2.pdf"])
    assert result["documents"] == []
    assert result["raw_extraction"] == []


@pytest.mark.integration
def test_edge_config_not_mutated_across_runs() -> None:
    """Original config is not mutated by multiple converter runs."""
    config = ExtractionConfig(output_format="html")
    converter = KreuzbergConverter(config=config, batch=False)
    converter.run(sources=[FIXTURES_DIR / "sample.txt"])
    converter.run(sources=[FIXTURES_DIR / "sample.txt"])
    # Original config should not be mutated
    assert config.output_format == "html"
