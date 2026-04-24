# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging

import chonkie

logger = logging.getLogger(__name__)


@component
class ChonkieTokenDocumentSplitter:
    """
    A Document Splitter that uses Chonkie's TokenChunker to split documents.

    ### Usage example
    ```python
    from haystack import Document
    from haystack_integrations.components.preprocessors.chonkie import ChonkieTokenDocumentSplitter

    chunker = ChonkieTokenDocumentSplitter(chunk_size=512, chunk_overlap=50)
    documents = [Document(content="Hello world. This is a test.")]
    result = chunker.run(documents=documents)
    print(result["documents"])
    ```
    """

    def __init__(
        self,
        *,
        tokenizer: str = "character",
        chunk_size: int = 2048,
        chunk_overlap: int = 0,
        skip_empty_documents: bool = True,
        page_break_character: str = "\f",
    ) -> None:
        """
        Initializes the ChonkieTokenDocumentSplitter.

        :param tokenizer: The tokenizer to use for chunking. Defaults to "character".
            Common options include "character", "gpt2", and "cl100k_base".
            See the [Chonkie documentation](https://docs.chonkie.ai/) for more information on available tokenizers.
        :param chunk_size: The maximum number of tokens per chunk. The actual length depends on the chosen tokenizer.
        :param chunk_overlap: The overlap between consecutive chunks.
        :param skip_empty_documents: Whether to skip empty documents.
        :param page_break_character: The character to use for page breaks.
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.skip_empty_documents = skip_empty_documents
        self.page_break_character = page_break_character

        self._chunker = chonkie.TokenChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Splits a list of documents into smaller token-based chunks.

        :param documents: The list of documents to split.
        :returns: A dictionary with the "documents" key containing the list of chunks.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = "ChonkieTokenDocumentSplitter expects a list of Document objects."
            raise TypeError(msg)

        chunked_documents = []
        for doc in documents:
            if doc.content is None:
                msg = f"ChonkieTokenDocumentSplitter works only with text documents but doc ID {doc.id} is None."
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
            tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            skip_empty_documents=self.skip_empty_documents,
            page_break_character=self.page_break_character,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChonkieTokenDocumentSplitter":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)
