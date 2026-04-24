import pytest

# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack import Document

from haystack_integrations.components.preprocessors.chonkie import ChonkieSentenceDocumentSplitter


class TestChonkieSentenceDocumentSplitter:
    def test_init_default(self):
        chunker = ChonkieSentenceDocumentSplitter()
        assert chunker.chunk_size == 2048
        assert chunker.tokenizer == "character"
        assert chunker.chunk_overlap == 0
        assert chunker.min_sentences_per_chunk == 1
        assert chunker.min_characters_per_sentence == 12
        assert chunker.approximate is False
        assert chunker.delim is None
        assert chunker.include_delim == "prev"

    def test_to_dict(self):
        chunker = ChonkieSentenceDocumentSplitter(
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
            "type": "haystack_integrations.components.preprocessors.chonkie.sentence_splitter.ChonkieSentenceDocumentSplitter",  # noqa: E501
            "init_parameters": {
                "chunk_size": 1024,
                "tokenizer": "word",
                "chunk_overlap": 50,
                "min_sentences_per_chunk": 2,
                "min_characters_per_sentence": 10,
                "approximate": True,
                "delim": [". ", "? "],
                "include_delim": "next",
                "skip_empty_documents": True,
                "page_break_character": "\f",
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.preprocessors.chonkie.sentence_splitter.ChonkieSentenceDocumentSplitter",  # noqa: E501
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
        chunker = ChonkieSentenceDocumentSplitter.from_dict(data)
        assert chunker.chunk_size == 1024
        assert chunker.tokenizer == "word"
        assert chunker.chunk_overlap == 50
        assert chunker.min_sentences_per_chunk == 2
        assert chunker.min_characters_per_sentence == 10
        assert chunker.approximate is True
        assert chunker.delim == [". ", "? "]
        assert chunker.include_delim == "next"

    def test_run(self):
        chunker = ChonkieSentenceDocumentSplitter(chunk_size=15, chunk_overlap=2)
        doc = Document(content="Hello world! This is a test string for chunking. Here is a new sentence.")
        result = chunker.run(documents=[doc])

        assert "documents" in result
        chunks = result["documents"]
        assert len(chunks) == 3
        for split_id, chunk in enumerate(chunks):
            assert chunk.meta["source_id"] == doc.id
            assert chunk.meta["split_id"] == split_id
            assert chunk.meta["page_number"] == 1
            assert chunk.meta["split_idx_start"] >= 0
            assert chunk.meta["split_idx_end"] > chunk.meta["split_idx_start"]
            assert chunk.meta["token_count"] > 0
            assert chunk.content == doc.content[chunk.meta["split_idx_start"] : chunk.meta["split_idx_end"]]
            assert len(chunk.meta) == 6

    def test_run_empty_document(self):
        chunker = ChonkieSentenceDocumentSplitter(skip_empty_documents=True)
        result = chunker.run(documents=[Document(content="")])
        assert result["documents"] == []

        chunker = ChonkieSentenceDocumentSplitter(skip_empty_documents=False)
        result = chunker.run(documents=[Document(content="")])
        assert result["documents"] == []

    def test_run_none_content(self):
        chunker = ChonkieSentenceDocumentSplitter()
        with pytest.raises(
            ValueError, match=r"ChonkieSentenceDocumentSplitter works only with text documents but doc ID .* is None"
        ):
            chunker.run(documents=[Document(content=None)])

    def test_run_page_number(self):
        chunker = ChonkieSentenceDocumentSplitter(chunk_size=20)
        # Add periods to ensure sentence splitting
        text = "Page 1 content.\f. Page 2 content."
        doc = Document(content=text, meta={"page_number": 1})
        result = chunker.run(documents=[doc])
        chunks = result["documents"]
        assert len(chunks) >= 2
        # Verify page numbers
        assert chunks[0].meta["page_number"] == 1
        # Find a chunk that contains "Page 2" and check its page number
        page_2_chunk = next(c for c in chunks if "Page 2" in c.content)
        assert page_2_chunk.meta["page_number"] == 2

    def test_run_invalid_documents_type(self):

        chunker = ChonkieSentenceDocumentSplitter()
        with pytest.raises(TypeError, match="expects a list of Document objects"):
            chunker.run(documents="invalid")
        with pytest.raises(TypeError, match="expects a list of Document objects"):
            chunker.run(documents=[1, 2, 3])

    @pytest.mark.integration
    def test_run_integration(self):
        chunker = ChonkieSentenceDocumentSplitter(
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
        assert len(chunks) == 3
        for split_id, chunk in enumerate(chunks):
            assert chunk.meta["source_id"] == docs[0].id
            assert chunk.meta["split_id"] == split_id
            assert chunk.meta["page_number"] == 1
            assert chunk.meta["split_idx_start"] >= 0
            assert chunk.meta["split_idx_end"] > chunk.meta["split_idx_start"]
            assert chunk.meta["token_count"] > 0
            assert chunk.content == text[chunk.meta["split_idx_start"] : chunk.meta["split_idx_end"]]
            assert len(chunk.meta) == 6
