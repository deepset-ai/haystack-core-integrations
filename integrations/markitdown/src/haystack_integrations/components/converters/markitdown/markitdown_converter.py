# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path
from typing import Any

from haystack import Document, component, logging
from haystack.components.converters.utils import normalize_metadata
from haystack.dataclasses import ByteStream

from markitdown import MarkItDown, StreamInfo

logger = logging.getLogger(__name__)


@component
class MarkItDownConverter:
    """
    Converts files to Haystack Documents using [MarkItDown](https://github.com/microsoft/markitdown).

    MarkItDown is a Microsoft library that converts many file formats to Markdown,
    including PDF, Word (.docx), PowerPoint (.pptx), Excel (.xlsx), HTML, images,
    audio, and more. All processing is performed locally.

    ### Usage example

    ```python
    from haystack_integrations.components.converters.markitdown import MarkItDownConverter

    converter = MarkItDownConverter()
    result = converter.run(sources=["document.pdf", "report.docx"])
    documents = result["documents"]
    ```
    """

    def __init__(
        self,
        store_full_path: bool = False,
    ) -> None:
        """
        Initializes the MarkItDownConverter.

        :param store_full_path:
            If `True`, the full file path is stored in the Document metadata.
            If `False`, only the file name is stored. Defaults to `False`.
        """
        self.store_full_path = store_full_path
        self._converter = MarkItDown()

    @component.output_types(documents=list[Document])
    def run(
        self,
        sources: list[str | Path | ByteStream],
        meta: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Converts files to Documents using MarkItDown.

        :param sources:
            List of file paths or ByteStream objects to convert.
        :param meta:
            Optional metadata to attach to the Documents. Can be a single dict
            applied to all Documents, or a list of dicts aligned with `sources`.
        :returns:
            A dictionary with key `documents` containing the converted Documents.
        """
        meta_list = normalize_metadata(meta, sources_count=len(sources))
        documents: list[Document] = []

        for source, user_meta in zip(sources, meta_list, strict=True):
            try:
                if isinstance(source, ByteStream):
                    file_path_str = source.meta.get("file_path", "")
                    ext = Path(file_path_str).suffix if file_path_str else ""
                    result = self._converter.convert_stream(
                        io.BytesIO(source.data), stream_info=StreamInfo(extension=ext)
                    )
                else:
                    result = self._converter.convert(str(source))
            except Exception as e:
                logger.warning(
                    "Could not convert {source} to a Document. Skipping it. Error: {error}",
                    source=source,
                    error=e,
                )
                continue

            source_meta: dict[str, Any] = {}
            if isinstance(source, ByteStream):
                if "file_path" in source.meta:
                    file_path = source.meta["file_path"]
                    source_meta["file_path"] = file_path if self.store_full_path else Path(file_path).name
            else:
                source_meta["file_path"] = str(source) if self.store_full_path else Path(source).name

            documents.append(Document(content=result.text_content, meta={**source_meta, **user_meta}))

        return {"documents": documents}
