# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict

import chonkie


@component
class ChonkieRecursiveChunker:
    """
    A Document Splitter that uses Chonkie's RecursiveChunker to split documents.

    Usage::

        from haystack import Document
        from haystack_integrations.components.preprocessors.chonkie import ChonkieRecursiveChunker

        chunker = ChonkieRecursiveChunker(chunk_size=512)
        documents = [Document(content="Hello world. This is a test.")]
        result = chunker.run(documents=documents)
        print(result["documents"])
    """

    def __init__(
        self,
        tokenizer: str = "character",
        chunk_size: int = 2048,
        min_characters_per_chunk: int = 24,
        rules: Any = None,
    ) -> None:
        """
        Initializes the ChonkieRecursiveChunker.

        :param tokenizer: The tokenizer to use for chunking. Defaults to "character".
        :param chunk_size: The maximum size of each chunk.
        :param min_characters_per_chunk: The minimum number of characters per chunk.
        :param rules: Custom rules for recursive chunking. If None, default rules are used.
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.min_characters_per_chunk = min_characters_per_chunk
        self.rules = rules

        kwargs = {
            "tokenizer": tokenizer,
            "chunk_size": chunk_size,
            "min_characters_per_chunk": min_characters_per_chunk,
        }
        if rules is not None:
            kwargs["rules"] = rules
            
        self._chunker = chonkie.RecursiveChunker(**kwargs)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Splits a list of documents into smaller chunks.

        :param documents: The list of documents to split.
        :returns: A dictionary with the "documents" key containing the list of chunks.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = "ChonkieRecursiveChunker expects a list of Document objects."
            raise TypeError(msg)

        chunked_documents = []
        for doc in documents:
            if not doc.content:
                continue

            chunks = self._chunker.chunk(doc.content)
            for chunk in chunks:
                meta = doc.meta.copy() if doc.meta else {}
                meta["source_id"] = doc.id
                meta["start_index"] = chunk.start_index
                meta["end_index"] = chunk.end_index
                meta["token_count"] = chunk.token_count

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
            min_characters_per_chunk=self.min_characters_per_chunk,
            rules=self.rules,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChonkieRecursiveChunker":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)



