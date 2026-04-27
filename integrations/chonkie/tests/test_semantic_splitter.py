# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import chonkie
import pytest
from haystack import Document

from haystack_integrations.components.preprocessors.chonkie import ChonkieSemanticDocumentSplitter


class TestChonkieSemanticDocumentSplitter:
    def test_init_default(self):
        chunker = ChonkieSemanticDocumentSplitter()
        assert chunker.embedding_model == "minishlab/potion-base-32M"
        assert chunker.threshold == 0.8
        assert chunker.chunk_size == 2048
        assert chunker.similarity_window == 3
        assert chunker.min_sentences_per_chunk == 1
        assert chunker.min_characters_per_sentence == 24
        assert chunker._chunker is None

    def test_to_dict(self):
        chunker = ChonkieSemanticDocumentSplitter(
            embedding_model="all-MiniLM-L6-v2",
            threshold=0.75,
            chunk_size=1024,
            similarity_window=2,
            min_sentences_per_chunk=2,
            min_characters_per_sentence=10,
        )
        data = chunker.to_dict()
        assert data == {
            "type": "haystack_integrations.components.preprocessors.chonkie.semantic_splitter.ChonkieSemanticDocumentSplitter",  # noqa: E501
            "init_parameters": {
                "embedding_model": "all-MiniLM-L6-v2",
                "threshold": 0.75,
                "chunk_size": 1024,
                "similarity_window": 2,
                "min_sentences_per_chunk": 2,
                "min_characters_per_sentence": 10,
                "delim": None,
                "include_delim": "prev",
                "skip_window": 0,
                "filter_window": 5,
                "filter_polyorder": 3,
                "filter_tolerance": 0.2,
                "skip_empty_documents": True,
                "page_break_character": "\f",
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.preprocessors.chonkie.semantic_splitter.ChonkieSemanticDocumentSplitter",  # noqa: E501
            "init_parameters": {
                "embedding_model": "all-MiniLM-L6-v2",
                "threshold": 0.75,
                "chunk_size": 1024,
                "similarity_window": 2,
                "min_sentences_per_chunk": 2,
                "min_characters_per_sentence": 10,
                "delim": [". ", "! "],
                "include_delim": "prev",
                "skip_window": 1,
                "filter_window": 3,
                "filter_polyorder": 2,
                "filter_tolerance": 0.1,
            },
        }
        chunker = ChonkieSemanticDocumentSplitter.from_dict(data)
        assert chunker.embedding_model == "all-MiniLM-L6-v2"
        assert chunker.threshold == 0.75
        assert chunker.chunk_size == 1024
        assert chunker.similarity_window == 2
        assert chunker.min_sentences_per_chunk == 2
        assert chunker.min_characters_per_sentence == 10
        assert chunker.delim == [". ", "! "]
        assert chunker.include_delim == "prev"
        assert chunker.skip_window == 1
        assert chunker.filter_window == 3
        assert chunker.filter_polyorder == 2
        assert chunker.filter_tolerance == 0.1

    @patch("haystack_integrations.components.preprocessors.chonkie.semantic_splitter.chonkie.SemanticChunker")
    def test_warm_up(self, mock_chunker):
        chunker = ChonkieSemanticDocumentSplitter()
        assert chunker._chunker is None
        chunker.warm_up()
        assert chunker._chunker is not None
        mock_chunker.assert_called_once()

        # Calling warm_up again should not re-initialize
        chunker.warm_up()
        mock_chunker.assert_called_once()

    @patch("haystack_integrations.components.preprocessors.chonkie.semantic_splitter.chonkie.SemanticChunker")
    def test_run(self, mock_chunker):
        # Setup mock return chunks
        mock_instance = mock_chunker.return_value
        mock_instance.chunk.return_value = [
            chonkie.types.base.Chunk(text="Hello world!", token_count=3, start_index=0, end_index=12),
            chonkie.types.base.Chunk(
                text="This is a semantic test string.", token_count=7, start_index=13, end_index=44
            ),
        ]

        chunker = ChonkieSemanticDocumentSplitter()
        chunker.warm_up()
        doc = Document(
            content="Hello world! This is a semantic test string. It contains multiple sentences. We will split it up."
        )
        result = chunker.run(documents=[doc])

        assert "documents" in result
        chunks = result["documents"]
        assert len(chunks) == 2
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
        chunker = ChonkieSemanticDocumentSplitter(skip_empty_documents=True)
        result = chunker.run(documents=[Document(content="")])
        assert result["documents"] == []

        chunker = ChonkieSemanticDocumentSplitter(skip_empty_documents=False)
        result = chunker.run(documents=[Document(content="")])
        assert result["documents"] == []

    def test_run_none_content(self):
        chunker = ChonkieSemanticDocumentSplitter()
        with pytest.raises(
            ValueError, match=r"ChonkieSemanticDocumentSplitter works only with text documents but doc ID .* is None"
        ):
            chunker.run(documents=[Document(content=None)])

    def test_run_page_number(self):
        chunker = ChonkieSemanticDocumentSplitter(chunk_size=50)
        text = "Page 1 content.\fPage 2 content."
        doc = Document(content=text, meta={"page_number": 1})
        result = chunker.run(documents=[doc])
        chunks = result["documents"]
        assert len(chunks) >= 1
        assert chunks[0].meta["page_number"] == 1

    def test_run_invalid_documents_type(self):

        chunker = ChonkieSemanticDocumentSplitter()
        with pytest.raises(TypeError, match="expects a list of Document objects"):
            chunker.run(documents="invalid")
        with pytest.raises(TypeError, match="expects a list of Document objects"):
            chunker.run(documents=[1, 2, 3])

    @pytest.mark.integration
    def test_run_integration(self):
        chunker = ChonkieSemanticDocumentSplitter(embedding_model="minishlab/potion-base-32M", chunk_size=64)
        chunker.warm_up()
        text = (
            "Semantic chunking is a method of splitting text based on the meaning of the content. "
            "It uses embedding models to determine boundaries. This ensures that chunks are semantically coherent."
        )
        docs = [Document(content=text)]
        result = chunker.run(documents=docs)
        chunks = result["documents"]
        assert len(chunks) == 1
        for split_id, chunk in enumerate(chunks):
            assert chunk.meta["source_id"] == docs[0].id
            assert chunk.meta["split_id"] == split_id
            assert chunk.meta["page_number"] == 1
            assert chunk.meta["split_idx_start"] >= 0
            assert chunk.meta["split_idx_end"] > chunk.meta["split_idx_start"]
            assert chunk.meta["token_count"] > 0
            assert chunk.content == text[chunk.meta["split_idx_start"] : chunk.meta["split_idx_end"]]
            assert len(chunk.meta) == 6
