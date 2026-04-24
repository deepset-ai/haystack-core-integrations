# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import chonkie
import pytest
from haystack import Document

from haystack_integrations.components.preprocessors.chonkie import ChonkieSemanticChunker


class TestChonkieSemanticChunker:
    @patch("haystack_integrations.components.preprocessors.chonkie.semantic_chunker.chonkie.SemanticChunker")
    def test_init_default(self, _mock_chunker):
        chunker = ChonkieSemanticChunker()
        assert chunker.embedding_model == "minishlab/potion-base-32M"
        assert chunker.threshold == 0.8
        assert chunker.chunk_size == 2048
        assert chunker.similarity_window == 3
        assert chunker.min_sentences_per_chunk == 1
        assert chunker.min_characters_per_sentence == 24

    @patch("haystack_integrations.components.preprocessors.chonkie.semantic_chunker.chonkie.SemanticChunker")
    def test_to_dict(self, _mock_chunker):
        chunker = ChonkieSemanticChunker(
            embedding_model="all-MiniLM-L6-v2",
            threshold=0.75,
            chunk_size=1024,
            similarity_window=2,
            min_sentences_per_chunk=2,
            min_characters_per_sentence=10,
        )
        data = chunker.to_dict()
        assert data == {
            "type": "haystack_integrations.components.preprocessors.chonkie.semantic_chunker.ChonkieSemanticChunker",
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
            },
        }

    @patch("haystack_integrations.components.preprocessors.chonkie.semantic_chunker.chonkie.SemanticChunker")
    def test_from_dict(self, _mock_chunker):
        data = {
            "type": "haystack_integrations.components.preprocessors.chonkie.semantic_chunker.ChonkieSemanticChunker",
            "init_parameters": {
                "embedding_model": "all-MiniLM-L6-v2",
                "threshold": 0.75,
                "chunk_size": 1024,
                "similarity_window": 2,
                "min_sentences_per_chunk": 2,
                "min_characters_per_sentence": 10,
            },
        }
        chunker = ChonkieSemanticChunker.from_dict(data)
        assert chunker.embedding_model == "all-MiniLM-L6-v2"
        assert chunker.threshold == 0.75
        assert chunker.chunk_size == 1024
        assert chunker.similarity_window == 2
        assert chunker.min_sentences_per_chunk == 2
        assert chunker.min_characters_per_sentence == 10

    @patch("haystack_integrations.components.preprocessors.chonkie.semantic_chunker.chonkie.SemanticChunker")
    def test_run(self, mock_chunker):
        # Setup mock return chunks
        mock_instance = mock_chunker.return_value
        mock_instance.chunk.return_value = [
            chonkie.types.base.Chunk(text="Hello world!", token_count=3, start_index=0, end_index=12),
            chonkie.types.base.Chunk(
                text="This is a semantic test string.", token_count=7, start_index=13, end_index=44
            ),
        ]

        chunker = ChonkieSemanticChunker()
        doc = Document(
            content="Hello world! This is a semantic test string. It contains multiple sentences. We will split it up."
        )
        result = chunker.run(documents=[doc])

        assert "documents" in result
        chunks = result["documents"]
        assert len(chunks) == 2
        assert chunks[0].meta["source_id"] == doc.id
        assert chunks[0].meta["start_index"] == 0
        assert chunks[0].meta["end_index"] == 12

    def test_run_empty_document(self):
        chunker = ChonkieSemanticChunker()
        result = chunker.run(documents=[Document(content="")])
        assert result["documents"] == []

    def test_run_invalid_documents_type(self):

        chunker = ChonkieSemanticChunker()
        with pytest.raises(TypeError, match="expects a list of Document objects"):
            chunker.run(documents="invalid")
        with pytest.raises(TypeError, match="expects a list of Document objects"):
            chunker.run(documents=[1, 2, 3])
