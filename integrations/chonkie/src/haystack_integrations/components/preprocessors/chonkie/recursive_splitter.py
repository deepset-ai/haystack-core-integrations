# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging

import chonkie
from chonkie.types.recursive import RecursiveLevel, RecursiveRules

logger = logging.getLogger(__name__)


@component
class ChonkieRecursiveDocumentSplitter:
    """
    A Document Splitter that uses Chonkie's RecursiveChunker to split documents.

    ### Usage example
    ```python
    from haystack import Document
    from haystack_integrations.components.preprocessors.chonkie import ChonkieRecursiveDocumentSplitter

    chunker = ChonkieRecursiveDocumentSplitter(chunk_size=512)
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
        min_characters_per_chunk: int = 24,
        rules: RecursiveRules | dict[str, Any] | None = None,
        skip_empty_documents: bool = True,
        page_break_character: str = "\f",
    ) -> None:
        """
        Initializes the ChonkieRecursiveDocumentSplitter.

        :param tokenizer: The tokenizer to use for chunking. Defaults to "character".
            Common options include "character", "gpt2", and "cl100k_base".
            See the [Chonkie documentation](https://docs.chonkie.ai/) for more information on available tokenizers.
        :param chunk_size: The maximum number of tokens per chunk. The actual length depends on the chosen tokenizer.
        :param min_characters_per_chunk: The minimum number of characters per chunk.
        :param rules: Custom rules for recursive chunking. If None, default rules are used.
            See the [Chonkie documentation](https://docs.chonkie.ai/) for more information.
        :param skip_empty_documents: Whether to skip empty documents.
        :param page_break_character: The character to use for page breaks.
        """
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.min_characters_per_chunk = min_characters_per_chunk
        self.skip_empty_documents = skip_empty_documents
        self.page_break_character = page_break_character

        if isinstance(rules, dict):
            levels = [RecursiveLevel(**level) for level in rules.get("levels", [])]
            self.rules: RecursiveRules | None = RecursiveRules(levels=levels)
        else:
            self.rules = rules

        kwargs: dict[str, Any] = {
            "tokenizer": tokenizer,
            "chunk_size": chunk_size,
            "min_characters_per_chunk": min_characters_per_chunk,
        }
        if self.rules is not None:
            kwargs["rules"] = self.rules

        self._chunker = chonkie.RecursiveChunker(**kwargs)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Splits a list of documents into smaller chunks.

        :param documents: The list of documents to split.
        :returns: A dictionary with the "documents" key containing the list of chunks.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = "ChonkieRecursiveDocumentSplitter expects a list of Document objects."
            raise TypeError(msg)

        chunked_documents = []
        for doc in documents:
            if doc.content is None:
                msg = f"ChonkieRecursiveDocumentSplitter works only with text documents but doc ID {doc.id} is None."
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
        rules_dict = asdict(self.rules) if self.rules else None
        return default_to_dict(
            self,
            tokenizer=self.tokenizer,
            chunk_size=self.chunk_size,
            min_characters_per_chunk=self.min_characters_per_chunk,
            rules=rules_dict,
            skip_empty_documents=self.skip_empty_documents,
            page_break_character=self.page_break_character,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChonkieRecursiveDocumentSplitter":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)
