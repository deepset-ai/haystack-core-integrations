# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import chonkie
from haystack import Document, component, default_from_dict, default_to_dict


@component
class ChonkieTokenChunker:
    """
    A Document Splitter that uses Chonkie's TokenChunker to split documents.

    Usage::

        from haystack import Document
        from haystack_integrations.components.preprocessors.chonkie import ChonkieTokenChunker

        chunker = ChonkieTokenChunker(chunk_size=512, chunk_overlap=50)
        documents = [Document(content="Hello world. This is a test.")]
        result = chunker.run(documents=documents)
        print(result["documents"])
    """

    def __init__(
        self,
        tokenizer: str = "character",
        chunk_size: int = 2048,
        chunk_overlap: int = 0,
    ) -> None:
        """
        Initializes the ChonkieTokenChunker.

        :param tokenizer: The tokenizer to use for chunking. Defaults to "character".
        :param chunk_size: The maximum size of each chunk.
        :param chunk_overlap: The overlap between consecutive chunks.
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._chunker = chonkie.TokenChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Splits a list of documents into smaller token-based chunks.

        :param documents: The list of documents to split.
        :returns: A dictionary with the "documents" key containing the list of chunks.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = "ChonkieTokenChunker expects a list of Document objects."
            raise TypeError(msg)

        chunked_documents = []
        for doc in documents:
            if not doc.content:
                continue

            chunks = self._chunker.chunk(doc.content)
            for chunk in chunks:
                meta = doc.meta.copy() if doc.meta else {}
                meta["source_id"] = doc.id
                meta["start_index"] = getattr(chunk, "start_index", None)
                meta["end_index"] = getattr(chunk, "end_index", None)
                meta["token_count"] = getattr(chunk, "token_count", None)

                new_doc = Document(content=chunk.text, meta=meta)
                chunked_documents.append(new_doc)

        return {"documents": chunked_documents}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChonkieTokenChunker":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)
