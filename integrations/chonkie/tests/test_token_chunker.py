# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import Document
from haystack_integrations.components.preprocessors.chonkie import ChonkieTokenChunker


class TestChonkieTokenChunker:
    def test_init_default(self):
        chunker = ChonkieTokenChunker()
        assert chunker.chunk_size == 2048
        assert chunker.tokenizer == "character"
        assert chunker.chunk_overlap == 0

    def test_to_dict(self):
        chunker = ChonkieTokenChunker(chunk_size=1024, tokenizer="word", chunk_overlap=50)
        data = chunker.to_dict()
        assert data == {
            "type": "haystack_integrations.components.preprocessors.chonkie.token_chunker.ChonkieTokenChunker",
            "init_parameters": {
                "chunk_size": 1024,
                "tokenizer": "word",
                "chunk_overlap": 50,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.preprocessors.chonkie.token_chunker.ChonkieTokenChunker",
            "init_parameters": {
                "chunk_size": 1024,
                "tokenizer": "word",
                "chunk_overlap": 50,
            },
        }
        chunker = ChonkieTokenChunker.from_dict(data)
        assert chunker.chunk_size == 1024
        assert chunker.tokenizer == "word"
        assert chunker.chunk_overlap == 50

    def test_run(self):
        chunker = ChonkieTokenChunker(chunk_size=10, chunk_overlap=2)
        doc = Document(content="Hello world! This is a test string for chunking.")
        result = chunker.run(documents=[doc])

        assert "documents" in result
        chunks = result["documents"]
        assert len(chunks) > 0
        assert chunks[0].meta["source_id"] == doc.id
        assert "start_index" in chunks[0].meta
        assert "end_index" in chunks[0].meta
