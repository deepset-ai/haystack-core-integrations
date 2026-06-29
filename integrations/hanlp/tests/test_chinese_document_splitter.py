# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.utils import deserialize_callable, serialize_callable

from haystack_integrations.components.preprocessors.hanlp import ChineseDocumentSplitter


# custom split function for testing
def custom_split(text: str) -> list[str]:
    return text.split(".")


class TestChineseDocumentSplitter:
    @pytest.fixture
    def sample_text(self) -> str:
        return (
            "这是第一句话，也是故事的开端，紧接着是第二句话，渐渐引出了背景；随后，翻开新/f的一页，"
            "我们读到了这一页的第一句话，继续延展出情节的发展，直到这页的第二句话将整段文字温柔地收束于平静之中。"
        )

    def test_to_dict(self):
        """
        Test the to_dict method of the DocumentSplitter class.
        """
        splitter = ChineseDocumentSplitter(split_by="word", split_length=10, split_overlap=2, split_threshold=5)
        serialized = splitter.to_dict()

        expected_type = (
            "haystack_integrations.components.preprocessors.hanlp.chinese_document_splitter.ChineseDocumentSplitter"
        )
        assert serialized["type"] == expected_type
        assert serialized["init_parameters"]["split_by"] == "word"
        assert serialized["init_parameters"]["split_length"] == 10
        assert serialized["init_parameters"]["split_overlap"] == 2
        assert serialized["init_parameters"]["split_threshold"] == 5
        assert "splitting_function" not in serialized["init_parameters"]

    def test_to_dict_with_splitting_function(self):
        """
        Test the to_dict method of the DocumentSplitter class when a custom splitting function is provided.
        """

        splitter = ChineseDocumentSplitter(split_by="function", splitting_function=custom_split)
        serialized = splitter.to_dict()

        expected_type = (
            "haystack_integrations.components.preprocessors.hanlp.chinese_document_splitter.ChineseDocumentSplitter"
        )
        assert serialized["type"] == expected_type
        assert serialized["init_parameters"]["split_by"] == "function"
        assert "splitting_function" in serialized["init_parameters"]
        assert callable(deserialize_callable(serialized["init_parameters"]["splitting_function"]))

    def test_from_dict(self):
        """
        Test the from_dict class method of the DocumentSplitter class.
        """
        data = {
            "type": (
                "haystack_integrations.components.preprocessors.hanlp.chinese_document_splitter.ChineseDocumentSplitter"
            ),
            "init_parameters": {"split_by": "word", "split_length": 10, "split_overlap": 2, "split_threshold": 5},
        }
        splitter = ChineseDocumentSplitter.from_dict(data)

        assert splitter.split_by == "word"
        assert splitter.split_length == 10
        assert splitter.split_overlap == 2
        assert splitter.split_threshold == 5
        assert splitter.splitting_function is None

    def test_from_dict_with_splitting_function(self):
        """
        Test the from_dict class method of the DocumentSplitter class when a custom splitting function is provided.
        """

        data = {
            "type": (
                "haystack_integrations.components.preprocessors.hanlp.chinese_document_splitter.ChineseDocumentSplitter"
            ),
            "init_parameters": {"split_by": "function", "splitting_function": serialize_callable(custom_split)},
        }
        splitter = ChineseDocumentSplitter.from_dict(data)

        assert splitter.split_by == "function"
        assert callable(splitter.splitting_function)
        assert splitter.splitting_function("a.b.c") == ["a", "b", "c"]

    def test_validate_init_parameters(self):
        ChineseDocumentSplitter._validate_init_parameters(
            split_by="word",
            split_length=1000,
            split_overlap=200,
            split_threshold=0,
            granularity="coarse",
        )

        with pytest.raises(ValueError, match="split_length must be positive"):
            ChineseDocumentSplitter._validate_init_parameters(split_length=0)

        with pytest.raises(ValueError, match="split_overlap must be non-negative"):
            ChineseDocumentSplitter._validate_init_parameters(split_overlap=-1)

        with pytest.raises(ValueError, match="split_overlap must be less than split_length"):
            ChineseDocumentSplitter._validate_init_parameters(split_overlap=1000, split_length=500)

        with pytest.raises(ValueError, match="split_threshold must be non-negative"):
            ChineseDocumentSplitter._validate_init_parameters(split_threshold=-1)

        with pytest.raises(ValueError, match="split_threshold must be less than split_length"):
            ChineseDocumentSplitter._validate_init_parameters(split_threshold=1001, split_length=1000)

        with pytest.raises(
            ValueError,
            match="split_by must be one of 'word', 'sentence', 'passage', 'page', 'line', 'period', 'function'",
        ):
            ChineseDocumentSplitter._validate_init_parameters(split_by="invalid")

        with pytest.raises(ValueError, match="granularity must be one of 'coarse', 'fine'"):
            ChineseDocumentSplitter._validate_init_parameters(granularity="invalid")

    def test_split_by_character_returns_empty_for_none_content(self):
        splitter = ChineseDocumentSplitter(split_by="word")
        assert splitter._split_by_character(Document(content=None)) == []

    def test_split_by_hanlp_sentence_returns_empty_for_none_content(self):
        splitter = ChineseDocumentSplitter(split_by="sentence")
        assert splitter._split_by_hanlp_sentence(Document(content=None)) == []

    def test_split_document_dispatches_to_function_splitter(self):
        splitter = ChineseDocumentSplitter(split_by="function", splitting_function=custom_split)
        doc = Document(content="a.b.c")
        result = splitter._split_document(doc)
        assert [d.content for d in result] == ["a", "b", "c"]

    def test_split_by_function_returns_empty_for_none_content(self):
        splitter = ChineseDocumentSplitter(split_by="function", splitting_function=custom_split)
        assert splitter._split_by_function(Document(content=None)) == []

    def test_split_by_function_raises_when_no_function_provided(self):
        splitter = ChineseDocumentSplitter(split_by="function", splitting_function=custom_split)
        splitter.splitting_function = None
        with pytest.raises(ValueError, match=r"No splitting function provided\."):
            splitter._split_by_function(Document(content="a.b"))

    def test_split_by_function_raises_when_function_returns_non_list(self):
        splitter = ChineseDocumentSplitter(split_by="function", splitting_function=lambda t: "not a list")
        with pytest.raises(ValueError, match="must return a list of strings"):
            splitter._split_by_function(Document(content="a.b"))

    def test_split_by_function_produces_documents_with_metadata(self):
        splitter = ChineseDocumentSplitter(split_by="function", splitting_function=custom_split)
        doc = Document(content="a.b.c", meta={"name": "src"})
        result = splitter._split_by_function(doc)
        assert len(result) == 3
        assert [d.content for d in result] == ["a", "b", "c"]
        assert all(d.meta["source_id"] == doc.id for d in result)
        assert all(d.meta["name"] == "src" for d in result)
        assert [d.meta["split_id"] for d in result] == [0, 1, 2]

    @pytest.mark.parametrize(
        ("previous_content", "current_content"),
        [(None, "abc"), ("abc", None)],
    )
    def test_add_split_overlap_information_skips_when_content_is_none(self, previous_content, current_content):
        previous_doc = Document(content=previous_content, meta={"_split_overlap": []})
        current_doc = Document(content=current_content, meta={"_split_overlap": []})
        ChineseDocumentSplitter._add_split_overlap_information(current_doc, 0, previous_doc, 0)
        assert current_doc.meta["_split_overlap"] == []
        assert previous_doc.meta["_split_overlap"] == []

    def test_add_split_overlap_information_skips_when_no_matching_prefix(self):
        previous_doc = Document(content="abcdef", meta={"_split_overlap": []})
        current_doc = Document(content="xyz", meta={"_split_overlap": []})
        ChineseDocumentSplitter._add_split_overlap_information(current_doc, 2, previous_doc, 0)
        assert current_doc.meta["_split_overlap"] == []
        assert previous_doc.meta["_split_overlap"] == []

    @pytest.mark.parametrize("elements", [[], ["", "   ", "\t"]])
    def test_concatenate_units_returns_empty_for_empty_or_whitespace(self, elements):
        splitter = ChineseDocumentSplitter(split_by="word", split_length=5, split_overlap=1)
        assert splitter._concatenate_units(elements, split_length=5, split_overlap=1, split_threshold=0) == ([], [], [])

    def test_concatenate_units_returns_single_chunk_for_short_input(self):
        splitter = ChineseDocumentSplitter(split_by="word", split_length=10, split_overlap=0)
        text_splits, pages, start_idxs = splitter._concatenate_units(
            ["a", "b", "c"], split_length=10, split_overlap=0, split_threshold=0
        )
        assert text_splits == ["abc"]
        assert pages == [1]
        assert start_idxs == [0]

    def test_concatenate_units_applies_split_threshold(self):
        splitter = ChineseDocumentSplitter(split_by="word", split_length=2, split_overlap=0)
        text_splits, _, _ = splitter._concatenate_units(
            ["a", "b", "c"], split_length=2, split_overlap=0, split_threshold=2
        )
        # the trailing "c" chunk is smaller than split_threshold=2 and gets attached to previous
        assert text_splits == ["abc"]

    def test_number_of_sentences_to_keep_returns_zero_when_overlap_is_zero(self):
        splitter = ChineseDocumentSplitter(split_by="word", split_length=10, split_overlap=0)
        assert splitter._number_of_sentences_to_keep(["s1", "s2", "s3"], split_length=10, split_overlap=0) == 0

    def test_number_of_sentences_to_keep_stops_when_overlap_reached(self):
        splitter = ChineseDocumentSplitter(split_by="word", split_length=10, split_overlap=2)
        # one word per sentence, so we need at least 3 sentences to exceed split_overlap=2
        splitter.chinese_tokenizer = MagicMock(side_effect=lambda s: [s])
        result = splitter._number_of_sentences_to_keep(["s1", "s2", "s3", "s4"], split_length=10, split_overlap=2)
        assert result == 3

    def test_number_of_sentences_to_keep_stops_when_split_length_exceeded(self):
        splitter = ChineseDocumentSplitter(split_by="word", split_length=2, split_overlap=1)
        # each sentence tokenizes to 5 "words", so first iteration already exceeds split_length
        splitter.chinese_tokenizer = MagicMock(return_value=["a", "b", "c", "d", "e"])
        assert splitter._number_of_sentences_to_keep(["s1", "s2", "s3"], split_length=2, split_overlap=1) == 0

    def test_warm_up_is_idempotent(self):
        splitter = ChineseDocumentSplitter(split_by="word")
        with patch(
            "haystack_integrations.components.preprocessors.hanlp.chinese_document_splitter.hanlp"
        ) as mock_hanlp:
            mock_hanlp.load.return_value = MagicMock()
            splitter.warm_up()
            splitter.warm_up()
            assert mock_hanlp.load.call_count == 2  # once for tokenizer, once for split_sent

    @pytest.mark.integration
    def test_empty_list(self):
        splitter = ChineseDocumentSplitter()
        results = splitter.run(documents=[])
        assert results == {"documents": []}

    @pytest.mark.integration
    def test_empty_document(self):
        splitter = ChineseDocumentSplitter()
        documents = [Document(content="")]
        results = splitter.run(documents=documents)
        assert results == {"documents": []}

    @pytest.mark.integration
    def test_whitespace_only_document(self):
        splitter = ChineseDocumentSplitter()
        documents = [Document(content="  ")]
        results = splitter.run(documents=documents)
        assert len(results["documents"]) == 0

    @pytest.mark.integration
    def test_metadata_copied_to_split_documents(self):
        documents = [
            Document(content="这是测试文本。", meta={"name": "doc 0"}),
            Document(content="这是另一个测试文本。", meta={"name": "doc 1"}),
        ]
        splitter = ChineseDocumentSplitter(split_by="word", split_length=5, split_overlap=2)
        result = splitter.run(documents=documents)
        assert len(result["documents"]) == 2
        for doc, split_doc in zip(documents, result["documents"], strict=True):
            assert doc.meta.items() <= split_doc.meta.items()

    @pytest.mark.integration
    def test_source_id_stored_in_metadata(self):
        documents = [
            Document(content="这是第一个测试文本。"),
            Document(content="这是第二个测试文本。"),
        ]
        splitter = ChineseDocumentSplitter(split_by="word", split_length=5, split_overlap=2)
        result = splitter.run(documents=documents)
        assert len(result["documents"]) == 2
        for doc, split_doc in zip(documents, result["documents"], strict=True):
            assert doc.id == split_doc.meta["source_id"]

    @pytest.mark.integration
    def test_split_by_word(self, sample_text):
        splitter = ChineseDocumentSplitter(split_by="word", granularity="coarse", split_length=5, split_overlap=0)
        result = splitter.run(documents=[Document(content=sample_text)])
        docs = result["documents"]
        assert all(isinstance(doc, Document) for doc in docs)
        expected_lengths = [9, 9, 8, 8, 6, 6, 6, 8, 8, 6, 7, 8, 5]
        actual_lengths = [len(doc.content) for doc in docs]
        assert actual_lengths == expected_lengths

    @pytest.mark.integration
    def test_split_by_sentence(self, sample_text):
        splitter = ChineseDocumentSplitter(split_by="sentence", granularity="coarse", split_length=10, split_overlap=0)
        result = splitter.run(documents=[Document(content=sample_text)])
        docs = result["documents"]
        assert all(isinstance(doc, Document) for doc in docs), "All docs should be instances of Document"
        assert all(doc.content != "" for doc in docs), "All docs should have content"
        assert docs[-1].content.endswith("。"), "Last chunk should end with '。'"

    @pytest.mark.integration
    def test_respect_sentence_boundary(self):
        doc = Document(
            content="这是第一句话，这是第二句话，这是第三句话。"
            "这是第四句话，这是第五句话，这是第六句话！"
            "这是第七句话，这是第八句话，这是第九句话？"
        )
        splitter = ChineseDocumentSplitter(
            split_by="word", split_length=10, split_overlap=3, respect_sentence_boundary=True
        )
        result = splitter.run(documents=[doc])
        docs = result["documents"]

        assert len(docs) == 3
        assert all(doc.content.endswith(("。", "！", "？")) for doc in docs), "Sentence was cut off!"

    @pytest.mark.integration
    def test_overlap_chunks_with_long_text(self):
        doc = Document(
            content="月光轻轻洒落，林中传来阵阵狼嚎，夜色悄然笼罩一切。"
            "树叶在微风中沙沙作响，影子在地面上摇曳不定。"
            "一只猫头鹰静静地眨了眨眼，从枝头注视着四周……"
            "远处的小溪哗啦啦地流淌，仿佛在向石头倾诉着什么。"
            "“咔嚓”一声，某处的树枝突然断裂，然后恢复了寂静。"
            "空气中弥漫着松树与湿土的气息，令人心安。"
            "一只狐狸悄然出现，又迅速消失在灌木丛中。"
            "天上的星星闪烁着，仿佛在诉说古老的故事。"
            "时间仿佛停滞了……"
            "万物静候，聆听着夜的呼吸！"
        )

        splitter = ChineseDocumentSplitter(split_by="word", split_length=30, split_overlap=10, granularity="coarse")
        result = splitter.run(documents=[doc])
        docs = result["documents"]
        assert len(docs) == 6
        expected_lengths = [48, 46, 47, 45, 46, 47]
        actual_lengths = [len(doc.content) for doc in docs]
        assert actual_lengths == expected_lengths

        def has_any_overlap(suffix: str, prefix: str) -> bool:
            """
            Check if suffix and prefix have at least one continuous overlapping character sequence.
            Tries from the longest possible overlap down to 1 character.
            Returns True if any overlap found.
            """
            max_check_len = min(len(suffix), len(prefix))
            for length in range(max_check_len, 0, -1):
                if suffix[-length:] == prefix[:length]:
                    return True
            return False

        for i in range(1, len(docs)):
            prev_chunk = docs[i - 1].content
            curr_chunk = docs[i].content

            # Take last 20 chars of prev chunk and first 20 chars of current chunk to check overlap
            overlap_prev = prev_chunk[-20:]
            overlap_curr = curr_chunk[:20]

            assert has_any_overlap(overlap_prev, overlap_curr), (
                f"Chunks {i} and {i + 1} do not overlap. "
                f"Tail (up to 20 chars): '{overlap_prev}' vs Head (up to 20 chars): '{overlap_curr}'"
            )
