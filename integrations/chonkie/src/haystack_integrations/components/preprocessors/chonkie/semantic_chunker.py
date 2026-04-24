# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict

import chonkie

@component
class ChonkieSemanticChunker:
    """
    A Document Splitter that uses Chonkie's SemanticChunker to split documents.

    Usage::

        from haystack import Document
        from haystack_integrations.components.preprocessors.chonkie import ChonkieSemanticChunker

        chunker = ChonkieSemanticChunker(chunk_size=512)
        documents = [Document(content="Hello world. This is a test.")]
        result = chunker.run(documents=documents)
        print(result["documents"])
    """

    def __init__(
        self,
        embedding_model: Any = "minishlab/potion-base-32M",
        threshold: float = 0.8,
        chunk_size: int = 2048,
        similarity_window: int = 3,
        min_sentences_per_chunk: int = 1,
        min_characters_per_sentence: int = 24,
        delim: Any = None,
        include_delim: str = "prev",
        skip_window: int = 0,
        filter_window: int = 5,
        filter_polyorder: int = 3,
        filter_tolerance: float = 0.2,
    ) -> None:
        """
        Initializes the ChonkieSemanticChunker.

        :param embedding_model: The embedding model to use for semantic similarity.
        :param threshold: The semantic similarity threshold.
        :param chunk_size: The maximum size of each chunk.
        :param similarity_window: The window size for similarity calculations.
        :param min_sentences_per_chunk: The minimum number of sentences per chunk.
        :param min_characters_per_sentence: The minimum number of characters per sentence.
        :param delim: Delimiters to use for splitting. If None, default delimiters are used.
        :param include_delim: Whether to include the delimiter in the chunks.
        :param skip_window: The skip window for similarity calculations.
        :param filter_window: The filter window for similarity calculations.
        :param filter_polyorder: The polynomial order for similarity filtering.
        :param filter_tolerance: The tolerance for similarity filtering.
        """
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.similarity_window = similarity_window
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.min_characters_per_sentence = min_characters_per_sentence
        self.delim = delim
        self.include_delim = include_delim
        self.skip_window = skip_window
        self.filter_window = filter_window
        self.filter_polyorder = filter_polyorder
        self.filter_tolerance = filter_tolerance

        kwargs = {
            "embedding_model": embedding_model,
            "threshold": threshold,
            "chunk_size": chunk_size,
            "similarity_window": similarity_window,
            "min_sentences_per_chunk": min_sentences_per_chunk,
            "min_characters_per_sentence": min_characters_per_sentence,
            "include_delim": include_delim,
            "skip_window": skip_window,
            "filter_window": filter_window,
            "filter_polyorder": filter_polyorder,
            "filter_tolerance": filter_tolerance,
        }
        if delim is not None:
            kwargs["delim"] = delim

        self._chunker = chonkie.SemanticChunker(**kwargs)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Splits a list of documents into smaller semantic chunks.

        :param documents: The list of documents to split.
        :returns: A dictionary with the "documents" key containing the list of chunks.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = "ChonkieSemanticChunker expects a list of Document objects."
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
            embedding_model=self.embedding_model,
            threshold=self.threshold,
            chunk_size=self.chunk_size,
            similarity_window=self.similarity_window,
            min_sentences_per_chunk=self.min_sentences_per_chunk,
            min_characters_per_sentence=self.min_characters_per_sentence,
            delim=self.delim,
            include_delim=self.include_delim,
            skip_window=self.skip_window,
            filter_window=self.filter_window,
            filter_polyorder=self.filter_polyorder,
            filter_tolerance=self.filter_tolerance,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChonkieSemanticChunker":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)