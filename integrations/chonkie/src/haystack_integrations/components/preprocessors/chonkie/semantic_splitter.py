# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging

import chonkie

logger = logging.getLogger(__name__)


@component
class ChonkieSemanticDocumentSplitter:
    """
    A Document Splitter that uses Chonkie's SemanticChunker to split documents.

    ### Usage example
    ```python
    from haystack import Document
    from haystack_integrations.components.preprocessors.chonkie import ChonkieSemanticDocumentSplitter

    chunker = ChonkieSemanticDocumentSplitter(chunk_size=512)
    documents = [Document(content="Hello world. This is a test.")]
    result = chunker.run(documents=documents)
    print(result["documents"])
    ```
    """

    def __init__(
        self,
        *,
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
        skip_empty_documents: bool = True,
        page_break_character: str = "\f",
    ) -> None:
        """
        Initializes the ChonkieSemanticDocumentSplitter.

        :param embedding_model: The embedding model to use for semantic similarity.
            See the [Chonkie documentation](https://docs.chonkie.ai/) for more information on supported models.
        :param threshold: The semantic similarity threshold.
        :param chunk_size: The maximum number of tokens per chunk. The actual length depends on the
            embedding model's tokenizer.
        :param similarity_window: The window size for similarity calculations.
        :param min_sentences_per_chunk: The minimum number of sentences per chunk.
        :param min_characters_per_sentence: The minimum number of characters per sentence.
        :param delim: Delimiters to use for splitting. If None, default delimiters are used.
        :param include_delim: Whether to include the delimiter in the chunks.
        :param skip_window: The skip window for similarity calculations.
        :param filter_window: The filter window for similarity calculations.
        :param filter_polyorder: The polynomial order for similarity filtering.
        :param filter_tolerance: The tolerance for similarity filtering.
        :param skip_empty_documents: Whether to skip empty documents.
        :param page_break_character: The character to use for page breaks.
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
        self.skip_empty_documents = skip_empty_documents
        self.page_break_character = page_break_character
        self._chunker: chonkie.SemanticChunker | None = None

    def warm_up(self) -> None:
        """
        Initializes the component by loading the embedding model.
        """
        if self._chunker is not None:
            return

        kwargs = {
            "embedding_model": self.embedding_model,
            "threshold": self.threshold,
            "chunk_size": self.chunk_size,
            "similarity_window": self.similarity_window,
            "min_sentences_per_chunk": self.min_sentences_per_chunk,
            "min_characters_per_sentence": self.min_characters_per_sentence,
            "include_delim": self.include_delim,
            "skip_window": self.skip_window,
            "filter_window": self.filter_window,
            "filter_polyorder": self.filter_polyorder,
            "filter_tolerance": self.filter_tolerance,
        }
        if self.delim is not None:
            kwargs["delim"] = self.delim

        self._chunker = chonkie.SemanticChunker(**kwargs)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Splits a list of documents into smaller semantic chunks.

        :param documents: The list of documents to split.
        :returns: A dictionary with the "documents" key containing the list of chunks.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = "ChonkieSemanticDocumentSplitter expects a list of Document objects."
            raise TypeError(msg)

        if self._chunker is None:
            self.warm_up()
            assert self._chunker is not None  # noqa: S101

        chunked_documents = []
        for doc in documents:
            if doc.content is None:
                msg = f"ChonkieSemanticDocumentSplitter works only with text documents but doc ID {doc.id} is None."
                raise ValueError(msg)

            if doc.content == "" and self.skip_empty_documents:
                logger.warning(
                    "Document ID {doc_id} has an empty content. Skipping this document.",
                    doc_id=doc.id,
                )
                continue

            chunks = self._chunker.chunk(doc.content)
            base_page = doc.meta.get("page_number", 1) if doc.meta else 1
            for split_id, chunk in enumerate(chunks):
                current_page = base_page + doc.content[: chunk.start_index].count(self.page_break_character)
                meta = doc.meta.copy() if doc.meta else {}
                meta.update(
                    {
                        "source_id": doc.id,
                        "page_number": current_page,
                        "split_id": split_id,
                        "split_idx_start": chunk.start_index,
                        "split_idx_end": chunk.end_index,
                        "token_count": chunk.token_count,
                    }
                )
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
            skip_empty_documents=self.skip_empty_documents,
            page_break_character=self.page_break_character,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChonkieSemanticDocumentSplitter":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)
