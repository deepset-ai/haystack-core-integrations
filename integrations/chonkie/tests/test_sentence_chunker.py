import pytest

# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack import Document

from haystack_integrations.components.preprocessors.chonkie import ChonkieSentenceChunker


class TestChonkieSentenceChunker:
    def test_init_default(self):
        chunker = ChonkieSentenceChunker()
        assert chunker.chunk_size == 2048
        assert chunker.tokenizer == "character"
        assert chunker.chunk_overlap == 0
        assert chunker.min_sentences_per_chunk == 1
        assert chunker.min_characters_per_sentence == 12
        assert chunker.approximate is False
        assert chunker.delim is None
        assert chunker.include_delim == "prev"

    def test_to_dict(self):
        chunker = ChonkieSentenceChunker(
            chunk_size=1024,
            tokenizer="word",
            chunk_overlap=50,
            min_sentences_per_chunk=2,
            min_characters_per_sentence=10,
            approximate=True,
            delim=[". ", "? "],
            include_delim="next",
        )
        data = chunker.to_dict()
        assert data == {
            "type": "haystack_integrations.components.preprocessors.chonkie.sentence_chunker.ChonkieSentenceChunker",
            "init_parameters": {
                "chunk_size": 1024,
                "tokenizer": "word",
                "chunk_overlap": 50,
                "min_sentences_per_chunk": 2,
                "min_characters_per_sentence": 10,
                "approximate": True,
                "delim": [". ", "? "],
                "include_delim": "next",
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.preprocessors.chonkie.sentence_chunker.ChonkieSentenceChunker",
            "init_parameters": {
                "chunk_size": 1024,
                "tokenizer": "word",
                "chunk_overlap": 50,
                "min_sentences_per_chunk": 2,
                "min_characters_per_sentence": 10,
                "approximate": True,
                "delim": [". ", "? "],
                "include_delim": "next",
            },
        }
        chunker = ChonkieSentenceChunker.from_dict(data)
        assert chunker.chunk_size == 1024
        assert chunker.tokenizer == "word"
        assert chunker.chunk_overlap == 50
        assert chunker.min_sentences_per_chunk == 2
        assert chunker.min_characters_per_sentence == 10
        assert chunker.approximate is True
        assert chunker.delim == [". ", "? "]
        assert chunker.include_delim == "next"

    def test_run(self):
        chunker = ChonkieSentenceChunker(chunk_size=15, chunk_overlap=2)
        doc = Document(content="Hello world! This is a test string for chunking. Here is a new sentence.")
        result = chunker.run(documents=[doc])

        assert "documents" in result
        chunks = result["documents"]
        assert len(chunks) > 0
        assert chunks[0].meta["source_id"] == doc.id
        assert "start_index" in chunks[0].meta
        assert "end_index" in chunks[0].meta

    def test_run_empty_document(self):
        chunker = ChonkieSentenceChunker()
        result = chunker.run(documents=[Document(content="")])
        assert result["documents"] == []

    def test_run_invalid_documents_type(self):

        chunker = ChonkieSentenceChunker()
        with pytest.raises(TypeError, match="expects a list of Document objects"):
            chunker.run(documents="invalid")
        with pytest.raises(TypeError, match="expects a list of Document objects"):
            chunker.run(documents=[1, 2, 3])
