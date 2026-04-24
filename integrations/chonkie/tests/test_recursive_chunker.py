# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import Document

from haystack_integrations.components.preprocessors.chonkie import ChonkieRecursiveChunker


class TestChonkieRecursiveChunker:
    def test_init_default(self):
        chunker = ChonkieRecursiveChunker()
        assert chunker.chunk_size == 2048
        assert chunker.tokenizer == "character"
        assert chunker.min_characters_per_chunk == 24

    def test_to_dict(self):
        chunker = ChonkieRecursiveChunker(chunk_size=1024, tokenizer="word", min_characters_per_chunk=10)
        data = chunker.to_dict()
        assert data == {
            "type": "haystack_integrations.components.preprocessors.chonkie.recursive_chunker.ChonkieRecursiveChunker",
            "init_parameters": {
                "chunk_size": 1024,
                "tokenizer": "word",
                "min_characters_per_chunk": 10,
                "rules": None,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.preprocessors.chonkie.recursive_chunker.ChonkieRecursiveChunker",
            "init_parameters": {
                "chunk_size": 1024,
                "tokenizer": "word",
                "min_characters_per_chunk": 10,
            },
        }
        chunker = ChonkieRecursiveChunker.from_dict(data)
        assert chunker.chunk_size == 1024
        assert chunker.tokenizer == "word"
        assert chunker.min_characters_per_chunk == 10

    def test_run(self):
        chunker = ChonkieRecursiveChunker(chunk_size=10, min_characters_per_chunk=2)
        doc = Document(content="Hello world! This is a test string for chunking.")
        result = chunker.run(documents=[doc])

        assert "documents" in result
        chunks = result["documents"]
        assert len(chunks) > 0
        assert chunks[0].meta["source_id"] == doc.id
        assert "start_index" in chunks[0].meta
        assert "end_index" in chunks[0].meta
