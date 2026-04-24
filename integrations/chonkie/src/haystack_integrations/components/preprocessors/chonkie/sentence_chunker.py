# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import chonkie
from haystack import Document, component, default_from_dict, default_to_dict


@component
class ChonkieSentenceChunker:
    """
    A Document Splitter that uses Chonkie's SentenceChunker to split documents.

    Usage::

        from haystack import Document
        from haystack_integrations.components.preprocessors.chonkie import ChonkieSentenceChunker

        chunker = ChonkieSentenceChunker(chunk_size=512)
        documents = [Document(content="Hello world. This is a test.")]
        result = chunker.run(documents=documents)
        print(result["documents"])
    """

    def __init__(
        self,
        tokenizer: str = "character",
        chunk_size: int = 2048,
        chunk_overlap: int = 0,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 12,
        approximate: bool = False,
        delim: Any = None,
        include_delim: str = "prev",
    ) -> None:
        """
        Initializes the ChonkieSentenceChunker.

        :param tokenizer: The tokenizer to use for chunking. Defaults to "character".
        :param chunk_size: The maximum size of each chunk.
        :param chunk_overlap: The overlap between consecutive chunks.
        :param min_sentences_per_chunk: The minimum number of sentences per chunk.
        :param min_characters_per_sentence: The minimum number of characters per sentence.
        :param approximate: Whether to use approximate chunking.
        :param delim: Delimiters to use for splitting. If None, default delimiters are used.
        :param include_delim: Whether to include the delimiter in the chunks ("prev" or "next").
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.approximate = approximate
        self.delim = delim
        self.include_delim = include_delim

        kwargs = {
            "tokenizer": tokenizer,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "min_sentences_per_chunk": min_sentences_per_chunk,
            "min_characters_per_sentence": min_characters_per_sentence,
            "approximate": approximate,
            "include_delim": include_delim,
        }
        if delim is not None:
            kwargs["delim"] = delim

        self._chunker = chonkie.SentenceChunker(**kwargs)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Splits a list of documents into smaller sentence-based chunks.

        :param documents: The list of documents to split.
        :returns: A dictionary with the "documents" key containing the list of chunks.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = "ChonkieSentenceChunker expects a list of Document objects."
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
            min_sentences_per_chunk=self.min_sentences_per_chunk,
            min_characters_per_sentence=self.min_characters_per_sentence,
            approximate=self.approximate,
            delim=self.delim,
            include_delim=self.include_delim,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChonkieSentenceChunker":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)
