import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from haystack.core.serialization import component_from_dict, component_to_dict

from haystack_integrations.components.converters.docling import DoclingConverter, ExportType


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
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from docling_haystack.converter import DoclingConverter as LegacyDoclingConverter

    assert LegacyDoclingConverter is DoclingConverter
    assert any(issubclass(w.category, DeprecationWarning) and "docling_haystack.converter" in str(w.message) for w in caught)


def test_component_from_dict_legacy_nulls() -> None:
    # Before the public-attribute refactor, default serialization couldn't find
    # the _-prefixed attributes and fell back to the init defaults, so
    # convert_kwargs and md_export_kwargs were always serialized as null.
    # Verify that such a serialized dict still deserializes correctly.
    legacy_data = {
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
    restored = component_from_dict(DoclingConverter, legacy_data, "docling_converter")

    assert restored.convert_kwargs == {}
    assert restored.md_export_kwargs == {"image_placeholder": ""}
    assert restored.export_type == ExportType.DOC_CHUNKS
    assert restored.converter is None
    assert restored.chunker is None
    assert restored.meta_extractor is None


def test_component_to_dict_defaults() -> None:
    converter = DoclingConverter()
    data = component_to_dict(converter, "docling_converter")

    init_params = data["init_parameters"]
    assert init_params["converter"] is None
    assert init_params["convert_kwargs"] == {}
    assert init_params["export_type"] == ExportType.DOC_CHUNKS
    assert init_params["md_export_kwargs"] == {"image_placeholder": ""}
    assert init_params["chunker"] is None
    assert init_params["meta_extractor"] is None


def test_component_to_dict_custom_params() -> None:
    converter = DoclingConverter(
        convert_kwargs={"raises_on_error": False},
        export_type=ExportType.MARKDOWN,
        md_export_kwargs={"image_placeholder": "[img]"},
    )
    data = component_to_dict(converter, "docling_converter")

    init_params = data["init_parameters"]
    assert init_params["convert_kwargs"] == {"raises_on_error": False}
    assert init_params["export_type"] == ExportType.MARKDOWN
    assert init_params["md_export_kwargs"] == {"image_placeholder": "[img]"}


def test_component_from_dict_defaults() -> None:
    converter = DoclingConverter()
    data = component_to_dict(converter, "docling_converter")
    restored = component_from_dict(DoclingConverter, data, "docling_converter")

    assert restored.converter is None
    assert restored.convert_kwargs == {}
    assert restored.export_type == ExportType.DOC_CHUNKS
    assert restored.md_export_kwargs == {"image_placeholder": ""}
    assert restored.chunker is None
    assert restored.meta_extractor is None


def test_component_from_dict_custom_params() -> None:
    converter = DoclingConverter(
        convert_kwargs={"raises_on_error": False},
        export_type=ExportType.JSON,
        md_export_kwargs={"image_placeholder": "[img]"},
    )
    data = component_to_dict(converter, "docling_converter")
    restored = component_from_dict(DoclingConverter, data, "docling_converter")

    assert restored.convert_kwargs == {"raises_on_error": False}
    assert restored.export_type == ExportType.JSON
    assert restored.md_export_kwargs == {"image_placeholder": "[img]"}
