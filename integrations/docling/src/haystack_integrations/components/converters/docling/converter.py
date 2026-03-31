"""Docling Haystack converter module."""

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any

from haystack import Document, component

from docling.chunking import BaseChunk, BaseChunker, HybridChunker
from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter


class ExportType(str, Enum):
    """Enumeration of available export types."""

    MARKDOWN = "markdown"
    DOC_CHUNKS = "doc_chunks"
    JSON = "json"


class BaseMetaExtractor(ABC):
    """BaseMetaExtractor."""

    @abstractmethod
    def extract_chunk_meta(self, chunk: BaseChunk) -> dict[str, Any]:
        """Extract chunk meta."""
        raise NotImplementedError()

    @abstractmethod
    def extract_dl_doc_meta(self, dl_doc: DoclingDocument) -> dict[str, Any]:
        """Extract Docling document meta."""
        raise NotImplementedError()


class MetaExtractor(BaseMetaExtractor):
    """MetaExtractor."""

    def extract_chunk_meta(self, chunk: BaseChunk) -> dict[str, Any]:
        """Extract chunk meta."""
        return {"dl_meta": chunk.export_json_dict()}

    def extract_dl_doc_meta(self, dl_doc: DoclingDocument) -> dict[str, Any]:
        """Extract Docling document meta."""
        return {"dl_meta": {"origin": dl_doc.origin.model_dump(exclude_none=True)}} if dl_doc.origin else {}


@component
class DoclingConverter:
    """Docling Haystack converter."""

    def __init__(
        self,
        converter: DocumentConverter | None = None,
        convert_kwargs: dict[str, Any] | None = None,
        export_type: ExportType = ExportType.DOC_CHUNKS,
        md_export_kwargs: dict[str, Any] | None = None,
        chunker: BaseChunker | None = None,
        meta_extractor: BaseMetaExtractor | None = None,
    ) -> None:
        """
        Create a Docling Haystack converter.

        Args:
            converter: The Docling `DocumentConverter` to use; if not set, a system
                default is used.
            convert_kwargs: Any parameters to pass to Docling conversion; if not set, a
                system default is used.
            export_type: The export mode to use:
                * `ExportType.MARKDOWN` captures each input document as a single
                  markdown `Document`.
                * `ExportType.DOC_CHUNKS` (default) first chunks each input document
                  and then returns one `Document` per chunk.
                * `ExportType.JSON` serializes the full Docling document to a JSON string.
            md_export_kwargs: Any parameters to pass to Markdown export (applicable in
                case of `ExportType.MARKDOWN`).
            chunker: The Docling chunker instance to use; if not set, a system default
                is used.
            meta_extractor: The extractor instance to use for populating the output
                document metadata; if not set, a system default is used.
        """
        self._converter = converter or DocumentConverter()
        self._convert_kwargs = convert_kwargs if convert_kwargs is not None else {}
        self._export_type = export_type
        self._md_export_kwargs = md_export_kwargs if md_export_kwargs is not None else {"image_placeholder": ""}
        if self._export_type == ExportType.DOC_CHUNKS:
            self._chunker = chunker or HybridChunker()
        self._meta_extractor = meta_extractor or MetaExtractor()

    @component.output_types(documents=list[Document])
    def run(
        self,
        paths: Iterable[Path | str],
    ) -> dict[str, list[Document]]:
        """
        Run the DoclingConverter.

        Args:
            paths: The input document locations, either as local paths or URLs.

        Returns:
            list[Document]: The output Haystack Documents.
        """
        documents: list[Document] = []
        for filepath in paths:
            dl_doc = self._converter.convert(
                source=filepath,
                **self._convert_kwargs,
            ).document

            if self._export_type == ExportType.DOC_CHUNKS:
                chunk_iter = self._chunker.chunk(dl_doc=dl_doc)
                hs_docs = [
                    Document(
                        content=self._chunker.contextualize(chunk=chunk),
                        meta=self._meta_extractor.extract_chunk_meta(chunk=chunk),
                    )
                    for chunk in chunk_iter
                ]
                documents.extend(hs_docs)
            elif self._export_type == ExportType.MARKDOWN:
                hs_doc = Document(
                    content=dl_doc.export_to_markdown(**self._md_export_kwargs),
                    meta=self._meta_extractor.extract_dl_doc_meta(dl_doc=dl_doc),
                )
                documents.append(hs_doc)
            elif self._export_type == ExportType.JSON:
                hs_doc = Document(
                    content=json.dumps(dl_doc.export_to_dict()),
                    meta=self._meta_extractor.extract_dl_doc_meta(dl_doc=dl_doc),
                )
                documents.append(hs_doc)
            else:
                err_msg = f"Unexpected export type: {self._export_type}"
                raise RuntimeError(err_msg)
        return {"documents": documents}
