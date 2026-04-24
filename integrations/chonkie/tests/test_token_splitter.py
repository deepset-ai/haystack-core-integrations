import pytest

# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack import Document

from haystack_integrations.components.preprocessors.chonkie import ChonkieTokenDocumentSplitter


class TestChonkieTokenDocumentSplitter:
    def test_init_default(self):
        chunker = ChonkieTokenDocumentSplitter()
        assert chunker.chunk_size == 2048
        assert chunker.tokenizer == "character"
        assert chunker.chunk_overlap == 0

    def test_to_dict(self):
        chunker = ChonkieTokenDocumentSplitter(chunk_size=1024, tokenizer="word", chunk_overlap=50)
        data = chunker.to_dict()
        assert data == {
            "type": "haystack_integrations.components.preprocessors.chonkie.token_splitter.ChonkieTokenDocumentSplitter",  # noqa: E501
            "init_parameters": {
                "chunk_size": 1024,
                "tokenizer": "word",
                "chunk_overlap": 50,
                "skip_empty_documents": True,
                "page_break_character": "\f",
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.preprocessors.chonkie.token_splitter.ChonkieTokenDocumentSplitter",  # noqa: E501
            "init_parameters": {
                "chunk_size": 1024,
                "tokenizer": "word",
                "chunk_overlap": 50,
            },
        }
        chunker = ChonkieTokenDocumentSplitter.from_dict(data)
        assert chunker.chunk_size == 1024
        assert chunker.tokenizer == "word"
        assert chunker.chunk_overlap == 50

    def test_run(self):
        chunker = ChonkieTokenDocumentSplitter(chunk_size=10, chunk_overlap=2)
        doc = Document(content="Hello world! This is a test string for chunking.")
        result = chunker.run(documents=[doc])

        assert "documents" in result
        chunks = result["documents"]
        assert len(chunks) == 6
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
        chunker = ChonkieTokenDocumentSplitter(skip_empty_documents=True)
        result = chunker.run(documents=[Document(content="")])
        assert result["documents"] == []

        chunker = ChonkieTokenDocumentSplitter(skip_empty_documents=False)
        result = chunker.run(documents=[Document(content="")])
        assert result["documents"] == []

    def test_run_none_content(self):
        chunker = ChonkieTokenDocumentSplitter()
        with pytest.raises(
            ValueError, match=r"ChonkieTokenDocumentSplitter works only with text documents but doc ID .* is None"
        ):
            chunker.run(documents=[Document(content=None)])

    def test_run_page_number(self):
        chunker = ChonkieTokenDocumentSplitter(chunk_size=20)
        text = "Page 1 content.\fPage 2 content."
        doc = Document(content=text, meta={"page_number": 1})
        result = chunker.run(documents=[doc])
        chunks = result["documents"]
        assert len(chunks) >= 2
        # Verify page numbers
        assert chunks[0].meta["page_number"] == 1
        # The second chunk should be on page 2 (since \f is in chunk 1)
        assert chunks[1].meta["page_number"] == 2

    def test_run_invalid_documents_type(self):

        chunker = ChonkieTokenDocumentSplitter()
        with pytest.raises(TypeError, match="expects a list of Document objects"):
            chunker.run(documents="invalid")
        with pytest.raises(TypeError, match="expects a list of Document objects"):
            chunker.run(documents=[1, 2, 3])

    @pytest.mark.integration
    def test_run_integration(self):
        chunker = ChonkieTokenDocumentSplitter(tokenizer="word", chunk_size=15, chunk_overlap=3)
        text = (
            "Tokenization is the process of breaking down a sequence of text into smaller pieces "
            "called tokens. These tokens can be words, characters, or subwords. Overlap is crucial."
        )
        docs = [Document(content=text)]
        result = chunker.run(documents=docs)
        chunks = result["documents"]
        assert len(chunks) == 2
        for split_id, chunk in enumerate(chunks):
            assert chunk.meta["source_id"] == docs[0].id
            assert chunk.meta["split_id"] == split_id
            assert chunk.meta["page_number"] == 1
            assert chunk.meta["split_idx_start"] >= 0
            assert chunk.meta["split_idx_end"] > chunk.meta["split_idx_start"]
            assert chunk.meta["token_count"] > 0
            assert chunk.content == text[chunk.meta["split_idx_start"] : chunk.meta["split_idx_end"]]
            assert len(chunk.meta) == 6
