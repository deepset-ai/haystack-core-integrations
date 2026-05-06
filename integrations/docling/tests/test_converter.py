import json
import mimetypes
import warnings
from io import BytesIO
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream
from haystack.dataclasses import ByteStream

from haystack_integrations.components.converters.docling import (
    DoclingConverter,
    ExportType,
    MetaExtractor,
)
from haystack_integrations.components.converters.docling.converter import _bytestream_to_document_stream


def test_run_doc_chunks_minimal() -> None:
    paths = ["file-a.pdf", "file-b.pdf"]
    converter_mock = MagicMock()
    chunker_mock = MagicMock()
    meta_extractor_mock = MagicMock()

    converter_mock.convert.side_effect = [
        SimpleNamespace(document="dl-doc-for-file-a.pdf"),
        SimpleNamespace(document="dl-doc-for-file-b.pdf"),
    ]

    def chunk_side_effect(dl_doc: Any) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(text=f"chunk-1-of-{dl_doc}"),
            SimpleNamespace(text=f"chunk-2-of-{dl_doc}"),
        ]

    chunker_mock.chunk.side_effect = chunk_side_effect
    chunker_mock.contextualize.side_effect = lambda chunk: f"contextualized-{chunk.text}"

    meta_extractor_mock.extract_chunk_meta.side_effect = lambda chunk: {"chunk_id": chunk.text}

    converter = DoclingConverter(
        converter=converter_mock,
        export_type=ExportType.DOC_CHUNKS,
        chunker=chunker_mock,
        meta_extractor=meta_extractor_mock,
    )

    result = converter.run(paths=paths)
    documents = result["documents"]

    # Two chunks per input path from our mocked implementation.
    assert len(documents) == 4
    contents = [doc.content for doc in documents]
    metas = [doc.meta for doc in documents]

    assert "contextualized-chunk-1-of-dl-doc-for-file-a.pdf" in contents
    assert "contextualized-chunk-2-of-dl-doc-for-file-a.pdf" in contents
    assert {"chunk_id": "chunk-1-of-dl-doc-for-file-a.pdf"} in metas

    # Ensure our collaborators were actually exercised.
    assert converter_mock.convert.call_count == len(paths)
    assert chunker_mock.chunk.call_count == len(paths)
    assert meta_extractor_mock.extract_chunk_meta.call_count == len(documents)


def test_run_markdown_minimal() -> None:
    paths = ["doc-1.json"]
    converter_mock = MagicMock()
    meta_extractor_mock = MagicMock()

    dl_doc = MagicMock()
    dl_doc.export_to_markdown.return_value = "markdown-for-doc-1.json-image_placeholder=[img]"
    converter_mock.convert.return_value = SimpleNamespace(document=dl_doc)
    meta_extractor_mock.extract_dl_doc_meta.return_value = {
        "doc_id": "DummyMarkdownDoc(name='doc-1.json')",
    }

    converter = DoclingConverter(
        converter=converter_mock,
        export_type=ExportType.MARKDOWN,
        meta_extractor=meta_extractor_mock,
        md_export_kwargs={"image_placeholder": "[img]"},
    )

    result = converter.run(paths=paths)
    documents = result["documents"]

    assert len(documents) == 1
    doc = documents[0]

    # Content and meta are derived entirely from our mocked implementations.
    assert doc.content == "markdown-for-doc-1.json-image_placeholder=[img]"
    assert doc.meta == {"doc_id": "DummyMarkdownDoc(name='doc-1.json')"}

    converter_mock.convert.assert_called_once()
    dl_doc.export_to_markdown.assert_called_once_with(image_placeholder="[img]")
    meta_extractor_mock.extract_dl_doc_meta.assert_called_once_with(dl_doc=dl_doc)


def test_run_json_minimal() -> None:
    paths = ["doc-1.json"]
    converter_mock = MagicMock()
    meta_extractor_mock = MagicMock()

    dl_doc = MagicMock()
    dl_doc.export_to_dict.return_value = {"name": "doc-1.json", "kind": "dummy-json"}
    converter_mock.convert.return_value = SimpleNamespace(document=dl_doc)
    meta_extractor_mock.extract_dl_doc_meta.return_value = {
        "doc_id": "DummyJsonDoc(name='doc-1.json')",
    }

    converter = DoclingConverter(
        converter=converter_mock,
        export_type=ExportType.JSON,
        meta_extractor=meta_extractor_mock,
    )

    result = converter.run(paths=paths)
    documents = result["documents"]

    assert len(documents) == 1
    doc = documents[0]

    # Content is JSON-encoded export dict from our mocked implementation.
    assert json.loads(doc.content) == {"name": "doc-1.json", "kind": "dummy-json"}
    assert doc.meta == {"doc_id": "DummyJsonDoc(name='doc-1.json')"}

    converter_mock.convert.assert_called_once()
    dl_doc.export_to_dict.assert_called_once_with()
    meta_extractor_mock.extract_dl_doc_meta.assert_called_once_with(dl_doc=dl_doc)


def test_legacy_import_path() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from docling_haystack.converter import DoclingConverter as LegacyDoclingConverter

    assert LegacyDoclingConverter is DoclingConverter
    assert any(
        issubclass(w.category, DeprecationWarning) and "docling_haystack.converter" in str(w.message) for w in caught
    )


def test_component_to_dict_defaults() -> None:
    converter = DoclingConverter()
    assert converter.to_dict() == {
        "type": "haystack_integrations.components.converters.docling.converter.DoclingConverter",
        "init_parameters": {
            "converter": None,
            "convert_kwargs": {},
            "export_type": "doc_chunks",
            "md_export_kwargs": {"image_placeholder": ""},
            "chunker": None,
            "meta_extractor": None,
        },
    }


def test_component_to_dict_custom_params() -> None:
    converter = DoclingConverter(
        converter=DocumentConverter(),
        convert_kwargs={"raises_on_error": False},
        export_type=ExportType.MARKDOWN,
        md_export_kwargs={"image_placeholder": "[img]"},
        meta_extractor=MetaExtractor(),
    )
    assert converter.to_dict() == {
        "type": "haystack_integrations.components.converters.docling.converter.DoclingConverter",
        "init_parameters": {
            "converter": None,
            "convert_kwargs": {"raises_on_error": False},
            "export_type": "markdown",
            "md_export_kwargs": {"image_placeholder": "[img]"},
            "chunker": None,
            "meta_extractor": {
                "type": "haystack_integrations.components.converters.docling.converter.MetaExtractor",
                "data": {},
            },
        },
    }


def test_component_from_dict_defaults() -> None:
    # null kwargs mirror the pre-refactor serialization format and must still deserialize correctly
    data = {
        "type": "haystack_integrations.components.converters.docling.converter.DoclingConverter",
        "init_parameters": {
            "converter": None,
            "convert_kwargs": None,
            "export_type": "doc_chunks",
            "md_export_kwargs": None,
            "chunker": None,
            "meta_extractor": None,
        },
    }
    restored = DoclingConverter.from_dict(data)

    assert restored.converter is None
    assert restored.convert_kwargs == {}
    assert restored.export_type == ExportType.DOC_CHUNKS
    assert restored.md_export_kwargs == {"image_placeholder": ""}
    assert restored.chunker is None
    assert restored.meta_extractor is None


def test_component_from_dict_custom_params() -> None:
    data = {
        "type": "haystack_integrations.components.converters.docling.converter.DoclingConverter",
        "init_parameters": {
            "converter": None,
            "convert_kwargs": {"raises_on_error": False},
            "export_type": "json",
            "md_export_kwargs": {"image_placeholder": "[img]"},
            "chunker": None,
            "meta_extractor": {
                "type": "haystack_integrations.components.converters.docling.converter.MetaExtractor",
                "data": {},
            },
        },
    }
    restored = DoclingConverter.from_dict(data)

    assert restored.converter is None
    assert restored.convert_kwargs == {"raises_on_error": False}
    assert restored.export_type == ExportType.JSON
    assert restored.md_export_kwargs == {"image_placeholder": "[img]"}
    assert restored.chunker is None
    assert isinstance(restored.meta_extractor, MetaExtractor)


def test_component_to_dict_chunker_warns_and_is_dropped() -> None:
    converter = DoclingConverter(chunker=HybridChunker(merge_peers=False))

    assert converter.to_dict() == {
        "type": "haystack_integrations.components.converters.docling.converter.DoclingConverter",
        "init_parameters": {
            "converter": None,
            "convert_kwargs": {},
            "export_type": "doc_chunks",
            "md_export_kwargs": {"image_placeholder": ""},
            "chunker": None,
            "meta_extractor": None,
        },
    }


def test_run_with_sources_parameter() -> None:
    converter_mock = MagicMock()
    chunker_mock = MagicMock()
    meta_extractor_mock = MagicMock()

    converter_mock.convert.return_value = SimpleNamespace(document="dl-doc")
    chunker_mock.chunk.return_value = [SimpleNamespace(text="chunk-1")]
    chunker_mock.contextualize.return_value = "contextualized-chunk-1"
    meta_extractor_mock.extract_chunk_meta.return_value = {}

    converter = DoclingConverter(
        converter=converter_mock,
        export_type=ExportType.DOC_CHUNKS,
        chunker=chunker_mock,
        meta_extractor=meta_extractor_mock,
    )

    result = converter.run(sources=["file.pdf"])
    assert len(result["documents"]) == 1


def test_run_paths_deprecated() -> None:
    converter_mock = MagicMock()
    chunker_mock = MagicMock()
    meta_extractor_mock = MagicMock()

    converter_mock.convert.return_value = SimpleNamespace(document="dl-doc")
    chunker_mock.chunk.return_value = [SimpleNamespace(text="chunk-1")]
    chunker_mock.contextualize.return_value = "contextualized-chunk-1"
    meta_extractor_mock.extract_chunk_meta.return_value = {}

    converter = DoclingConverter(
        converter=converter_mock,
        export_type=ExportType.DOC_CHUNKS,
        chunker=chunker_mock,
        meta_extractor=meta_extractor_mock,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = converter.run(paths=["file.pdf"])

    assert len(result["documents"]) == 1
    assert any(issubclass(w.category, DeprecationWarning) and "paths" in str(w.message) for w in caught)


def test_run_meta_single_dict_doc_chunks() -> None:
    converter_mock = MagicMock()
    chunker_mock = MagicMock()
    meta_extractor_mock = MagicMock()

    converter_mock.convert.side_effect = [
        SimpleNamespace(document="dl-doc-a"),
        SimpleNamespace(document="dl-doc-b"),
    ]
    chunker_mock.chunk.side_effect = lambda dl_doc: [SimpleNamespace(text=f"chunk-of-{dl_doc}")]
    chunker_mock.contextualize.side_effect = lambda chunk: chunk.text
    meta_extractor_mock.extract_chunk_meta.return_value = {"extractor_key": "extractor_val"}

    converter = DoclingConverter(
        converter=converter_mock,
        export_type=ExportType.DOC_CHUNKS,
        chunker=chunker_mock,
        meta_extractor=meta_extractor_mock,
    )

    result = converter.run(sources=["a.pdf", "b.pdf"], meta={"custom": "value"})
    documents = result["documents"]

    assert len(documents) == 2
    for doc in documents:
        assert doc.meta["custom"] == "value"
        assert doc.meta["extractor_key"] == "extractor_val"


def test_run_meta_list_of_dicts_markdown() -> None:
    converter_mock = MagicMock()
    meta_extractor_mock = MagicMock()

    dl_doc_a = MagicMock()
    dl_doc_a.export_to_markdown.return_value = "markdown-a"
    dl_doc_b = MagicMock()
    dl_doc_b.export_to_markdown.return_value = "markdown-b"

    converter_mock.convert.side_effect = [
        SimpleNamespace(document=dl_doc_a),
        SimpleNamespace(document=dl_doc_b),
    ]
    meta_extractor_mock.extract_dl_doc_meta.return_value = {}

    converter = DoclingConverter(
        converter=converter_mock,
        export_type=ExportType.MARKDOWN,
        meta_extractor=meta_extractor_mock,
    )

    result = converter.run(
        sources=["a.pdf", "b.pdf"],
        meta=[{"source_id": "doc-a"}, {"source_id": "doc-b"}],
    )
    documents = result["documents"]

    assert len(documents) == 2
    assert documents[0].meta["source_id"] == "doc-a"
    assert documents[1].meta["source_id"] == "doc-b"


def test_run_meta_list_length_mismatch_raises() -> None:
    converter_mock = MagicMock()
    meta_extractor_mock = MagicMock()

    converter = DoclingConverter(
        converter=converter_mock,
        export_type=ExportType.MARKDOWN,
        meta_extractor=meta_extractor_mock,
    )

    with pytest.raises(ValueError):
        converter.run(sources=["a.pdf", "b.pdf"], meta=[{"x": 1}])


def test_run_with_bytestream_source() -> None:
    converter_mock = MagicMock()
    meta_extractor_mock = MagicMock()

    dl_doc = MagicMock()
    dl_doc.export_to_markdown.return_value = "markdown-content"
    converter_mock.convert.return_value = SimpleNamespace(document=dl_doc)
    meta_extractor_mock.extract_dl_doc_meta.return_value = {}

    converter = DoclingConverter(
        converter=converter_mock,
        export_type=ExportType.MARKDOWN,
        meta_extractor=meta_extractor_mock,
    )

    bytestream = ByteStream(data=b"%PDF-1.4 fake pdf content", meta={"file_path": "uploaded.pdf"})

    result = converter.run(sources=[bytestream])

    documents = result["documents"]
    assert len(documents) == 1
    # ByteStream meta is merged into the output document
    assert documents[0].meta["file_path"] == "uploaded.pdf"
    # docling was called with a DocumentStream, not a temp file path
    call_args = converter_mock.convert.call_args
    passed_source = call_args.kwargs["source"]
    assert isinstance(passed_source, DocumentStream)
    assert passed_source.name == "uploaded.pdf"
    assert isinstance(passed_source.stream, BytesIO)


class TestBytestreamToDocumentStream:
    def test_uses_file_path(self) -> None:
        bs = ByteStream(data=b"data", meta={"file_path": "report.pdf"})
        ds = _bytestream_to_document_stream(bs)
        assert ds.name == "report.pdf"
        assert ds.stream.read() == b"data"

    def test_strips_directory_from_file_path(self) -> None:
        bs = ByteStream(data=b"data", meta={"file_path": "/some/deep/path/report.pdf"})
        ds = _bytestream_to_document_stream(bs)
        assert ds.name == "report.pdf"

    def test_uses_file_name_key(self) -> None:
        bs = ByteStream(data=b"data", meta={"file_name": "slide-deck.pptx"})
        ds = _bytestream_to_document_stream(bs)
        assert ds.name == "slide-deck.pptx"

    def test_uses_name_key(self) -> None:
        bs = ByteStream(data=b"data", meta={"name": "notes.docx"})
        ds = _bytestream_to_document_stream(bs)
        assert ds.name == "notes.docx"

    def test_file_path_takes_priority_over_file_name(self) -> None:
        bs = ByteStream(data=b"data", meta={"file_path": "real.pdf", "file_name": "other.pdf"})
        ds = _bytestream_to_document_stream(bs)
        assert ds.name == "real.pdf"

    def test_file_name_takes_priority_over_name(self) -> None:
        bs = ByteStream(data=b"data", meta={"file_name": "chosen.pdf", "name": "ignored.pdf"})
        ds = _bytestream_to_document_stream(bs)
        assert ds.name == "chosen.pdf"

    def test_guesses_extension_from_mime_type(self) -> None:
        mime = "application/pdf"
        expected_ext = mimetypes.guess_extension(mime)
        bs = ByteStream(data=b"data", meta={"file_path": "report"}, mime_type=mime)
        ds = _bytestream_to_document_stream(bs)
        assert ds.name == f"report{expected_ext}"

    def test_keeps_extension_when_present(self) -> None:
        # mime_type should not override an already-present extension
        bs = ByteStream(data=b"data", meta={"file_path": "report.pdf"}, mime_type="text/plain")
        ds = _bytestream_to_document_stream(bs)
        assert ds.name == "report.pdf"

    def test_no_meta_no_mime_type(self) -> None:
        bs = ByteStream(data=b"data")
        ds = _bytestream_to_document_stream(bs)
        assert ds.name == "document"

    def test_no_meta_with_mime_type(self) -> None:
        mime = "application/pdf"
        expected_ext = mimetypes.guess_extension(mime)
        bs = ByteStream(data=b"data", mime_type=mime)
        ds = _bytestream_to_document_stream(bs)
        assert ds.name == f"document{expected_ext}"

    def test_empty_meta_no_mime_type(self) -> None:
        bs = ByteStream(data=b"data", meta={})
        ds = _bytestream_to_document_stream(bs)
        assert ds.name == "document"

    def test_returns_document_stream_with_bytesio(self) -> None:
        bs = ByteStream(data=b"hello", meta={"file_path": "f.pdf"})
        ds = _bytestream_to_document_stream(bs)
        assert isinstance(ds, DocumentStream)
        assert isinstance(ds.stream, BytesIO)

    def test_unknown_mime_type_keeps_base_name(self) -> None:
        # mimetypes.guess_extension returns None for unknown types, so the name stays as-is.
        assert mimetypes.guess_extension("application/x-totally-made-up-type") is None
        bs = ByteStream(
            data=b"data",
            meta={"file_path": "report"},
            mime_type="application/x-totally-made-up-type",
        )
        ds = _bytestream_to_document_stream(bs)
        assert ds.name == "report"


class TestMetaExtractor:
    def test_extract_chunk_meta_wraps_export_json_dict(self) -> None:
        chunk = MagicMock()
        chunk.export_json_dict.return_value = {"some": "dict"}

        result = MetaExtractor().extract_chunk_meta(chunk=chunk)

        assert result == {"dl_meta": {"some": "dict"}}
        chunk.export_json_dict.assert_called_once_with()

    def test_extract_dl_doc_meta_with_origin(self) -> None:
        dl_doc = MagicMock()
        dl_doc.origin.model_dump.return_value = {"filename": "foo.pdf", "mimetype": "application/pdf"}

        result = MetaExtractor().extract_dl_doc_meta(dl_doc=dl_doc)

        assert result == {"dl_meta": {"origin": {"filename": "foo.pdf", "mimetype": "application/pdf"}}}
        dl_doc.origin.model_dump.assert_called_once_with(exclude_none=True)

    def test_extract_dl_doc_meta_without_origin(self) -> None:
        dl_doc = MagicMock()
        dl_doc.origin = None

        result = MetaExtractor().extract_dl_doc_meta(dl_doc=dl_doc)

        assert result == {}


def test_run_without_sources_or_paths_raises_value_error() -> None:
    converter = DoclingConverter(converter=MagicMock(), meta_extractor=MagicMock())
    with pytest.raises(ValueError, match=r"Either 'sources' or the deprecated 'paths' parameter must be provided."):
        converter.run()
