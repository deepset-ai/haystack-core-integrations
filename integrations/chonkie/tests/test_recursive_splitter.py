# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from chonkie.types.recursive import RecursiveLevel, RecursiveRules
from haystack import Document

from haystack_integrations.components.preprocessors.chonkie import ChonkieRecursiveDocumentSplitter


class TestChonkieRecursiveDocumentSplitter:
    def test_init_default(self):
        chunker = ChonkieRecursiveDocumentSplitter()
        assert chunker.chunk_size == 2048
        assert chunker.tokenizer == "character"
        assert chunker.min_characters_per_chunk == 24

    def test_to_dict(self):
        chunker = ChonkieRecursiveDocumentSplitter(chunk_size=1024, tokenizer="word", min_characters_per_chunk=10)
        data = chunker.to_dict()
        assert data == {
            "type": "haystack_integrations.components.preprocessors.chonkie.recursive_splitter.ChonkieRecursiveDocumentSplitter",  # noqa: E501
            "init_parameters": {
                "chunk_size": 1024,
                "tokenizer": "word",
                "min_characters_per_chunk": 10,
                "rules": None,
                "skip_empty_documents": True,
                "page_break_character": "\f",
            },
        }

    def test_to_dict_with_rules(self):

        rules = RecursiveRules(levels=[RecursiveLevel(delimiters=["\n\n"], include_delim="prev")])
        chunker = ChonkieRecursiveDocumentSplitter(rules=rules)
        data = chunker.to_dict()
        assert data == {
            "type": "haystack_integrations.components.preprocessors.chonkie.recursive_splitter.ChonkieRecursiveDocumentSplitter",  # noqa: E501
            "init_parameters": {
                "chunk_size": 2048,
                "tokenizer": "character",
                "min_characters_per_chunk": 24,
                "rules": {
                    "levels": [
                        {
                            "delimiters": ["\n\n"],
                            "whitespace": False,
                            "include_delim": "prev",
                            "pattern": None,
                            "pattern_mode": "split",
                        }
                    ]
                },
                "skip_empty_documents": True,
                "page_break_character": "\f",
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.preprocessors.chonkie.recursive_splitter.ChonkieRecursiveDocumentSplitter",  # noqa: E501
            "init_parameters": {
                "chunk_size": 1024,
                "tokenizer": "word",
                "min_characters_per_chunk": 10,
            },
        }
        chunker = ChonkieRecursiveDocumentSplitter.from_dict(data)
        assert chunker.chunk_size == 1024
        assert chunker.tokenizer == "word"
        assert chunker.min_characters_per_chunk == 10

    def test_from_dict_with_rules(self):

        data = {
            "type": "haystack_integrations.components.preprocessors.chonkie.recursive_splitter.ChonkieRecursiveDocumentSplitter",  # noqa: E501
            "init_parameters": {
                "chunk_size": 1024,
                "tokenizer": "word",
                "min_characters_per_chunk": 10,
                "rules": {
                    "levels": [
                        {
                            "delimiters": ["\n\n"],
                            "whitespace": False,
                            "include_delim": "prev",
                            "pattern": None,
                            "pattern_mode": "split",
                        }
                    ]
                },
            },
        }
        chunker = ChonkieRecursiveDocumentSplitter.from_dict(data)
        assert chunker.chunk_size == 1024
        assert chunker.tokenizer == "word"
        assert chunker.min_characters_per_chunk == 10
        assert isinstance(chunker.rules, RecursiveRules)
        assert chunker.rules.levels[0].delimiters == ["\n\n"]

    def test_run(self):
        chunker = ChonkieRecursiveDocumentSplitter(chunk_size=10, min_characters_per_chunk=2)
        doc = Document(content="Hello world! This is a test string for chunking.")
        result = chunker.run(documents=[doc])

        assert "documents" in result
        chunks = result["documents"]
        assert len(chunks) == 7
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
        chunker = ChonkieRecursiveDocumentSplitter(skip_empty_documents=True)
        result = chunker.run(documents=[Document(content="")])
        assert result["documents"] == []

        chunker = ChonkieRecursiveDocumentSplitter(skip_empty_documents=False)
        result = chunker.run(documents=[Document(content="")])
        # Chonkie returns no chunks for empty string even if not skipped
        assert result["documents"] == []

    def test_run_none_content(self):
        chunker = ChonkieRecursiveDocumentSplitter()
        with pytest.raises(
            ValueError, match=r"ChonkieRecursiveDocumentSplitter works only with text documents but doc ID .* is None"
        ):
            chunker.run(documents=[Document(content=None)])

    def test_run_page_number_complex(self):
        chunker = ChonkieRecursiveDocumentSplitter(chunk_size=15)
        text = "Page 1 content.\fPage 2 content.\fPage 3 content.\fPage 4 content."
        doc = Document(content=text, meta={"page_number": 10})
        result = chunker.run(documents=[doc])
        chunks = result["documents"]

        pages_seen = sorted({chunk.meta["page_number"] for chunk in chunks})

        assert 10 in pages_seen
        assert pages_seen[-1] >= 12
        last_chunk = chunks[-1]
        text_before_last_chunk = text[: last_chunk.meta["split_idx_start"]]
        expected_last_page = 10 + text_before_last_chunk.count("\f")
        assert last_chunk.meta["page_number"] == expected_last_page

    def test_run_invalid_documents_type(self):

        chunker = ChonkieRecursiveDocumentSplitter()
        with pytest.raises(TypeError, match="expects a list of Document objects"):
            chunker.run(documents="invalid")
        with pytest.raises(TypeError, match="expects a list of Document objects"):
            chunker.run(documents=[1, 2, 3])

    @pytest.mark.integration
    def test_run_integration(self):

        custom_rules = RecursiveRules(
            levels=[
                RecursiveLevel(delimiters=["\n\n", "\n"], include_delim="prev"),
                RecursiveLevel(delimiters=[". ", "? ", "! "], include_delim="prev"),
                RecursiveLevel(delimiters=[", ", "; "], include_delim="prev"),
            ]
        )
        chunker = ChonkieRecursiveDocumentSplitter(chunk_size=50, rules=custom_rules)
        text = (
            "Haystack is an open-source AI framework. "
            "It allows you to build powerful, production-ready applications with LLMs.\n\n"
            "It is highly modular and flexible; "
            "you can use it for RAG, search, and much more. "
        )
        docs = [Document(content=text)]
        result = chunker.run(documents=docs)
        chunks = result["documents"]
        assert len(chunks) == 5
        for split_id, chunk in enumerate(chunks):
            assert chunk.meta["source_id"] == docs[0].id
            assert chunk.meta["split_id"] == split_id
            assert chunk.meta["page_number"] == 1
            assert chunk.meta["split_idx_start"] >= 0
            assert chunk.meta["split_idx_end"] > chunk.meta["split_idx_start"]
            assert chunk.meta["token_count"] > 0
            assert chunk.content == text[chunk.meta["split_idx_start"] : chunk.meta["split_idx_end"]]
            assert len(chunk.meta) == 6
