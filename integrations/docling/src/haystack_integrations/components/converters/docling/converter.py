"""Docling Haystack converter module."""

import json
import mimetypes
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any

from docling_core.types.io import DocumentStream
from haystack import Document, component, logging
from haystack.components.converters.utils import normalize_metadata
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ByteStream
from haystack.utils.base_serialization import deserialize_class_instance, serialize_class_instance

from docling.chunking import BaseChunk, BaseChunker, HybridChunker
from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


def _bytestream_to_document_stream(source: ByteStream) -> DocumentStream:
    """
    Build a `DocumentStream` from a Haystack `ByteStream`.

    Resolves the stream name by checking common metadata keys (`file_path`, `file_name`, `name`) and falling back to
    MIME-type extension guessing so that docling can reliably detect the input format.
    """
    meta = source.meta or {}
    raw_name = meta.get("file_path") or meta.get("file_name") or meta.get("name")

    if raw_name:
        name = Path(raw_name).name
    else:
        name = "document"

    if not Path(name).suffix and source.mime_type:
        ext = mimetypes.guess_extension(source.mime_type)
        if ext:
            name = f"{name}{ext}"

    return DocumentStream(name=name, stream=BytesIO(source.data))


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseMetaExtractor":  # noqa: ARG003
        """Deserialize from a dictionary."""
        return cls()


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

        :param converter: The Docling `DocumentConverter` to use; if not set, a system
            default is used.
        :param convert_kwargs: Any parameters to pass to Docling conversion; if not set, a
            system default is used.
        :param export_type: The export mode to use:
            * `ExportType.MARKDOWN` captures each input document as a single
              markdown `Document`.
            * `ExportType.DOC_CHUNKS` (default) first chunks each input document
              and then returns one `Document` per chunk.
            * `ExportType.JSON` serializes the full Docling document to a JSON string.
        :param md_export_kwargs: Any parameters to pass to Markdown export (applicable in
            case of `ExportType.MARKDOWN`).
        :param chunker: The Docling chunker instance to use; if not set, a system default
            is used.
        :param meta_extractor: The extractor instance to use for populating the output
            document metadata; if not set, a system default is used.
        """
        # Public attributes match init parameter names 1:1 for default serialization.
        self.converter = converter
        self.convert_kwargs = convert_kwargs if convert_kwargs is not None else {}
        self.export_type = ExportType(export_type)
        self.md_export_kwargs = md_export_kwargs if md_export_kwargs is not None else {"image_placeholder": ""}
        self.chunker = chunker
        self.meta_extractor = meta_extractor

        # Resolved instances used internally at runtime.
        self._converter_instance = converter or DocumentConverter()
        if self.export_type == ExportType.DOC_CHUNKS:
            self._chunker_instance = chunker or HybridChunker()
        self._meta_extractor_instance = meta_extractor or MetaExtractor()

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        if self.converter is not None:
            logger.warning(
                "DoclingConverter.to_dict: the 'converter' parameter cannot be serialized and will be dropped. "
                "The component will use the default DocumentConverter when restored from the serialized form."
            )
        if self.chunker is not None:
            logger.warning(
                "DoclingConverter.to_dict: the 'chunker' parameter cannot be serialized and will be dropped. "
                "The component will use the default chunker when restored from the serialized form."
            )

        meta_extractor_data = None
        if self.meta_extractor is not None:
            meta_extractor_data = serialize_class_instance(self.meta_extractor)

        return default_to_dict(
            self,
            converter=None,
            convert_kwargs=self.convert_kwargs,
            export_type=self.export_type.value,
            md_export_kwargs=self.md_export_kwargs,
            chunker=None,
            meta_extractor=meta_extractor_data,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DoclingConverter":
        """
        Deserialize this component from a dictionary.

        The `converter` and `chunker` parameters are not serializable and are always ignored during
        deserialization; the restored instance will use the default `DocumentConverter` and `HybridChunker`
        respectively.

        :param data: Dictionary with keys `type` and `init_parameters`, as produced by `to_dict`.
        :returns: A new `DoclingConverter` instance.
        """
        init_params = data.get("init_parameters", {})

        meta_extractor_data = init_params.get("meta_extractor")
        if meta_extractor_data is not None:
            init_params["meta_extractor"] = deserialize_class_instance(meta_extractor_data)

        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        paths: list[str | Path] | None = None,
        sources: list[str | Path | ByteStream] | None = None,
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Run the DoclingConverter.

        :param paths: Deprecated. Use `sources` instead.
        :param sources: List of file paths, URLs, or ByteStream objects to convert.
        :param meta:
            Optional metadata to attach to the Documents.
            This value can be either a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced Documents.
            If it's a list, the length of the list must match the number of sources, because the two lists will
            be zipped.
            If a source is a ByteStream, its own metadata is also merged into the output.
        :returns:
            A dictionary with key `"documents"` containing the output Haystack Documents.
        :raises ValueError: If `meta` is a list whose length does not match the number of sources.
        :raises RuntimeError: If an unexpected `export_type` is encountered.
        """
        if paths is not None:
            warnings.warn(
                "The 'paths' parameter is deprecated. Use 'sources' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if sources is None:
                sources = list(paths)  # type: ignore[arg-type]

        if sources is None:
            msg = "Either 'sources' or the deprecated 'paths' parameter must be provided."
            raise ValueError(msg)

        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))

        documents: list[Document] = []
        for source, source_meta in zip(sources, meta_list, strict=True):
            if isinstance(source, ByteStream):
                doc_stream = _bytestream_to_document_stream(source)
                dl_doc = self._converter_instance.convert(source=doc_stream, **self.convert_kwargs).document
                # merge ByteStream meta (e.g. file_path, mime_type) with user-supplied meta
                merged_meta = {**(source.meta or {}), **source_meta}
            else:
                dl_doc = self._converter_instance.convert(source=source, **self.convert_kwargs).document
                merged_meta = source_meta

            if self.export_type == ExportType.DOC_CHUNKS:
                chunk_iter = self._chunker_instance.chunk(dl_doc=dl_doc)
                hs_docs = [
                    Document(
                        content=self._chunker_instance.contextualize(chunk=chunk),
                        meta={**self._meta_extractor_instance.extract_chunk_meta(chunk=chunk), **merged_meta},
                    )
                    for chunk in chunk_iter
                ]
                documents.extend(hs_docs)
            elif self.export_type == ExportType.MARKDOWN:
                hs_doc = Document(
                    content=dl_doc.export_to_markdown(**self.md_export_kwargs),
                    meta={**self._meta_extractor_instance.extract_dl_doc_meta(dl_doc=dl_doc), **merged_meta},
                )
                documents.append(hs_doc)
            elif self.export_type == ExportType.JSON:
                hs_doc = Document(
                    content=json.dumps(dl_doc.export_to_dict()),
                    meta={**self._meta_extractor_instance.extract_dl_doc_meta(dl_doc=dl_doc), **merged_meta},
                )
                documents.append(hs_doc)
            else:
                err_msg = f"Unexpected export type: {self.export_type}"
                raise RuntimeError(err_msg)

        return {"documents": documents}
