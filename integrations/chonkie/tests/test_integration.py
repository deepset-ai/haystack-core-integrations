# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from chonkie.types.recursive import RecursiveLevel, RecursiveRules
from haystack import Document

from haystack_integrations.components.preprocessors.chonkie import (
    ChonkieRecursiveChunker,
    ChonkieSentenceChunker,
    ChonkieTokenChunker,
)


@pytest.mark.integration
class TestChonkieChunkersIntegration:
    def test_recursive_chunker(self):
        custom_rules = RecursiveRules(
            levels=[
                RecursiveLevel(delimiters=["\n\n", "\n"], include_delim="prev"),
                RecursiveLevel(delimiters=[". ", "? ", "! "], include_delim="prev"),
                RecursiveLevel(delimiters=[", ", "; "], include_delim="prev"),
            ]
        )
        chunker = ChonkieRecursiveChunker(chunk_size=50, rules=custom_rules)
        text = (
            "Haystack is an open-source AI framework. "
            "It allows you to build powerful, production-ready applications with LLMs.\n\n"
            "It is highly modular and flexible; "
            "you can use it for RAG, search, and much more. "
        )
        docs = [Document(content=text)]
        result = chunker.run(documents=docs)
        chunks = result["documents"]
        assert len(chunks) > 0
        assert chunks[0].content.startswith("Haystack")
        assert chunks[0].meta["source_id"] == docs[0].id

    def test_token_chunker(self):
        chunker = ChonkieTokenChunker(tokenizer="word", chunk_size=15, chunk_overlap=3)
        text = (
            "Tokenization is the process of breaking down a sequence of text into smaller pieces "
            "called tokens. These tokens can be words, characters, or subwords. Overlap is crucial."
        )
        docs = [Document(content=text)]
        result = chunker.run(documents=docs)
        chunks = result["documents"]
        assert len(chunks) > 0
        assert chunks[0].meta["source_id"] == docs[0].id

    def test_sentence_chunker(self):
        chunker = ChonkieSentenceChunker(
            chunk_size=25,
            chunk_overlap=5,
            min_sentences_per_chunk=1,
            min_characters_per_sentence=10,
            approximate=False,
            delim=[". ", "? ", "! "],
            include_delim="prev",
        )
        text = (
            "Sentence chunking attempts to keep semantic boundaries intact by splitting on punctuation. "
            "Is this method effective? Yes, it is! "
        )
        docs = [Document(content=text)]
        result = chunker.run(documents=docs)
        chunks = result["documents"]
        assert len(chunks) > 0
        assert chunks[0].meta["source_id"] == docs[0].id
