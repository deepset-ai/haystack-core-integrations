# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack import Document

from haystack_integrations.components.preprocessors.hanlp import ChineseDocumentSplitter


class TestChineseDocumentSplitter:
    @pytest.fixture
    def sample_text(self) -> str:
        return (
            "这是第一句话，也是故事的开端，紧接着是第二句话，渐渐引出了背景；随后，翻开新/f的一页，"
            "我们读到了这一页的第一句话，继续延展出情节的发展，直到这页的第二句话将整段文字温柔地收束于平静之中。"
        )

    @pytest.mark.integration
    def test_empty_list(self):
        splitter = ChineseDocumentSplitter()
        splitter.warm_up()
        results = splitter.run(documents=[])
        assert results == {"documents": []}

    @pytest.mark.integration
    def test_empty_document(self):
        splitter = ChineseDocumentSplitter()
        documents = [Document(content="")]
        splitter.warm_up()
        results = splitter.run(documents=documents)
        assert results == {"documents": []}

    @pytest.mark.integration
    def test_whitespace_only_document(self):
        splitter = ChineseDocumentSplitter()
        documents = [Document(content="  ")]
        splitter.warm_up()
        results = splitter.run(documents=documents)
        assert len(results["documents"]) == 0

    @pytest.mark.integration
    def test_metadata_copied_to_split_documents(self):
        documents = [
            Document(content="这是测试文本。", meta={"name": "doc 0"}),
            Document(content="这是另一个测试文本。", meta={"name": "doc 1"}),
        ]
        splitter = ChineseDocumentSplitter(split_by="word", split_length=5)
        splitter.warm_up()
        result = splitter.run(documents=documents)
        assert len(result["documents"]) == 2
        for doc, split_doc in zip(documents, result["documents"]):
            assert doc.meta.items() <= split_doc.meta.items()

    @pytest.mark.integration
    def test_source_id_stored_in_metadata(self):
        documents = [
            Document(content="这是第一个测试文本。"),
            Document(content="这是第二个测试文本。"),
        ]
        splitter = ChineseDocumentSplitter(split_by="word", split_length=5)
        splitter.warm_up()
        result = splitter.run(documents=documents)
        assert len(result["documents"]) == 2
        for doc, split_doc in zip(documents, result["documents"]):
            assert doc.id == split_doc.meta["source_id"]

    @pytest.mark.integration
    def test_split_by_word(self, sample_text):
        splitter = ChineseDocumentSplitter(split_by="word", particle_size="coarse", split_length=5, split_overlap=0)
        splitter.warm_up()
        result = splitter.run(documents=[Document(content=sample_text)])
        docs = result["documents"]
        assert all(isinstance(doc, Document) for doc in docs)
        expected_lengths = [9, 9, 8, 8, 6, 6, 6, 8, 8, 6, 7, 8, 5]
        actual_lengths = [len(doc.content.strip()) for doc in docs]
        assert actual_lengths == expected_lengths

    @pytest.mark.integration
    def test_split_by_sentence(self, sample_text):
        splitter = ChineseDocumentSplitter(
            split_by="sentence", particle_size="coarse", split_length=10, split_overlap=0
        )
        splitter.warm_up()
        result = splitter.run(documents=[Document(content=sample_text)])
        docs = result["documents"]
        assert all(isinstance(doc, Document) for doc in docs)
        assert all(doc.content.strip() != "" for doc in docs)
        assert any("。" in doc.content for doc in docs), "Expected at least one chunk containing a full stop."

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
        splitter.warm_up()
        result = splitter.run(documents=[doc])
        docs = result["documents"]
        assert all(doc.content.strip().endswith(("。", "！", "？")) for doc in docs), "Sentence was cut off!"

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

        splitter = ChineseDocumentSplitter(split_by="word", split_length=30, split_overlap=10, particle_size="coarse")
        splitter.warm_up()
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
