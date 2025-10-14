# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from haystack import Document, component, logging
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.utils import deserialize_callable, serialize_callable
from more_itertools import windowed

import hanlp

logger = logging.getLogger(__name__)

# mapping of split by character, 'function' and 'sentence' don't split by character
_CHARACTER_SPLIT_BY_MAPPING = {"page": "\f", "passage": "\n\n", "period": ".", "word": " ", "line": "\n"}


@component
class ChineseDocumentSplitter:
    """
    A DocumentSplitter for Chinese text.

    'coarse' represents coarse granularity Chinese word segmentation, 'fine' represents fine granularity word
    segmentation, default is coarse granularity word segmentation.

    Unlike English where words are usually separated by spaces,
    Chinese text is written continuously without spaces between words.
    Chinese words can consist of multiple characters.
    For example, the English word "America" is translated to "美国" in Chinese,
    which consists of two characters but is treated as a single word.
    Similarly, "Portugal" is "葡萄牙" in Chinese, three characters but one word.
    Therefore, splitting by word means splitting by these multi-character tokens,
    not simply by single characters or spaces.

    ### Usage example
    ```python
    doc = Document(content=
        "这是第一句话，这是第二句话，这是第三句话。"
        "这是第四句话，这是第五句话，这是第六句话！"
        "这是第七句话，这是第八句话，这是第九句话？"
    )

    splitter = ChineseDocumentSplitter(
        split_by="word", split_length=10, split_overlap=3, respect_sentence_boundary=True
    )
    splitter.warm_up()
    result = splitter.run(documents=[doc])
    print(result["documents"])
    ```
    """

    def __init__(
        self,
        split_by: Literal["word", "sentence", "passage", "page", "line", "period", "function"] = "word",
        split_length: int = 1000,
        split_overlap: int = 200,
        split_threshold: int = 0,
        respect_sentence_boundary: bool = False,
        splitting_function: Optional[Callable] = None,
        granularity: Literal["coarse", "fine"] = "coarse",
    ):
        """
        Initialize the ChineseDocumentSplitter component.

        :param split_by: The unit for splitting your documents. Choose from:
            - `word` for splitting by spaces (" ")
            - `period` for splitting by periods (".")
            - `page` for splitting by form feed ("\\f")
            - `passage` for splitting by double line breaks ("\\n\\n")
            - `line` for splitting each line ("\\n")
            - `sentence` for splitting by HanLP sentence tokenizer

        :param split_length: The maximum number of units in each split.
        :param split_overlap: The number of overlapping units for each split.
        :param split_threshold: The minimum number of units per split. If a split has fewer units
            than the threshold, it's attached to the previous split.
        :param respect_sentence_boundary: Choose whether to respect sentence boundaries when splitting by "word".
            If True, uses HanLP to detect sentence boundaries, ensuring splits occur only between sentences.
        :param splitting_function: Necessary when `split_by` is set to "function".
            This is a function which must accept a single `str` as input and return a `list` of `str` as output,
            representing the chunks after splitting.
        :param granularity: The granularity of Chinese word segmentation, either 'coarse' or 'fine'.

        :raises ValueError: If the granularity is not 'coarse' or 'fine'.
        """
        self._validate_init_parameters(split_by, split_length, split_overlap, split_threshold, granularity)
        self.split_by = split_by
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_threshold = split_threshold
        self.respect_sentence_boundary = respect_sentence_boundary
        self.splitting_function = splitting_function
        self.granularity = granularity

    @staticmethod
    def _validate_init_parameters(
        split_by: Literal["word", "sentence", "passage", "page", "line", "period", "function"] = "word",
        split_length: int = 1000,
        split_overlap: int = 200,
        split_threshold: int = 0,
        granularity: Literal["coarse", "fine"] = "coarse",
    ) -> None:
        """
        Validate the init parameters.

        :param split_by: The unit for splitting your documents.
        :param split_length: The maximum number of units in each split.
        :param split_overlap: The number of overlapping units for each split.
        :param split_threshold: The minimum number of units per split. If a split has fewer units
            than the threshold, it's attached to the previous split.
        :param granularity: The granularity of Chinese word segmentation, either 'coarse' or 'fine'.

        :raises ValueError:
            If the split_length is not positive.
            If the split_overlap is negative.
            If the split_overlap is greater than or equal to the split_length.
            If the split_threshold is negative.
            If the split_threshold is greater than the split_length.
            If the split_by is not one of 'word', 'sentence', 'passage', 'page', 'line', 'period', 'function'.
            If the granularity is not one of 'coarse', 'fine'.
        """

        if split_length <= 0:
            msg = f"split_length must be positive, but got {split_length}"
            raise ValueError(msg)
        if split_overlap < 0:
            msg = f"split_overlap must be non-negative, but got {split_overlap}"
            raise ValueError(msg)
        if split_overlap >= split_length:
            msg = f"split_overlap must be less than split_length, but got {split_overlap} >= {split_length}"
            raise ValueError(msg)
        if split_threshold < 0:
            msg = f"split_threshold must be non-negative, but got {split_threshold}"
            raise ValueError(msg)
        if split_threshold > split_length:
            msg = f"split_threshold must be less than split_length, but got {split_threshold} > {split_length}"
            raise ValueError(msg)
        if split_by not in {"word", "sentence", "passage", "page", "line", "period", "function"}:
            msg = (
                f"split_by must be one of 'word', 'sentence', 'passage', 'page', 'line', 'period', 'function', "
                f"but got {split_by}"
            )
            raise ValueError(msg)
        if granularity not in {"coarse", "fine"}:
            msg = f"granularity must be one of 'coarse', 'fine', but got {granularity}"
            raise ValueError(msg)

    @component.output_types(documents=list[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Split documents into smaller chunks.

        :param documents: The documents to split.
        :return: A dictionary containing the split documents.

        :raises RuntimeError: If the Chinese word segmentation model is not loaded.
        """
        if self.split_sent is None:
            msg = "The Chinese word segmentation model is not loaded. Please run 'warm_up()' before calling 'run()'."
            raise RuntimeError(msg)

        split_docs = []
        for doc in documents:
            split_docs.extend(self._split_document(doc))
        return {"documents": split_docs}

    def warm_up(self) -> None:
        """Warm up the component by loading the necessary models."""
        if self.granularity == "coarse":
            self.chinese_tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        if self.granularity == "fine":
            self.chinese_tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
        self.split_sent = hanlp.load(hanlp.pretrained.eos.UD_CTB_EOS_MUL)

    def _split_by_character(self, doc: Document) -> List[Document]:
        """
        Define a function to handle Chinese clauses

        :param doc: The document to split.
        :return: A list of split documents.
        """
        if doc.content is None:
            return []

        split_at = _CHARACTER_SPLIT_BY_MAPPING[self.split_by]

        units = self.chinese_tokenizer(doc.content)

        for i in range(len(units) - 1):
            units[i] += split_at
        text_splits, splits_pages, splits_start_idxs = self._concatenate_units(
            units, self.split_length, self.split_overlap, self.split_threshold
        )
        metadata = deepcopy(doc.meta)
        metadata["source_id"] = doc.id

        return self._create_docs_from_splits(
            text_splits=text_splits, splits_pages=splits_pages, splits_start_idxs=splits_start_idxs, meta=metadata
        )

    # Define a function to handle Chinese clauses
    def chinese_sentence_split(self, text: str) -> List[Dict[str, Any]]:
        """
        Split Chinese text into sentences.

        :param text: The text to split.
        :return: A list of split sentences.
        """
        # Split sentences
        sentences = self.split_sent(text)

        # Organize the format of segmented sentences
        results = []
        start = 0
        for sentence in sentences:
            start = text.find(sentence, start)
            end = start + len(sentence)
            results.append({"sentence": sentence, "start": start, "end": end})
            start = end

        return results

    def _split_document(self, doc: Document) -> List[Document]:
        if self.split_by == "sentence" or self.respect_sentence_boundary:
            return self._split_by_hanlp_sentence(doc)

        if self.split_by == "function" and self.splitting_function is not None:
            return self._split_by_function(doc)

        return self._split_by_character(doc)

    def _concatenate_sentences_based_on_word_amount(
        self, sentences: List[str], split_length: int, split_overlap: int, granularity: str
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Groups the sentences into chunks of `split_length` words while respecting sentence boundaries.

        This function is only used when splitting by `word` and `respect_sentence_boundary` is set to `True`, i.e.:
        with HanLP sentence tokenizer.

        :param sentences: The list of sentences to split.
        :param split_length: The maximum number of words in each split.
        :param split_overlap: The number of overlapping words in each split.
        :returns: A tuple containing the concatenated sentences, the start page numbers, and the start indices.
        """

        # chunk information
        chunk_word_count = 0
        chunk_starting_page_number = 1
        chunk_start_idx = 0
        current_chunk: List[str] = []

        # output lists
        split_start_page_numbers = []
        list_of_splits: List[List[str]] = []
        split_start_indices = []

        for sentence_idx, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            chunk_word_count += len(self.chinese_tokenizer(sentence))
            next_sentence_word_count = (
                len(self.chinese_tokenizer(sentences[sentence_idx + 1])) if sentence_idx < len(sentences) - 1 else 0
            )

            # Number of words in the current chunk plus the next sentence is larger than the split_length,
            # or we reached the last sentence
            if (chunk_word_count + next_sentence_word_count) > split_length or sentence_idx == len(sentences) - 1:
                #  Save current chunk and start a new one
                list_of_splits.append(current_chunk)
                split_start_page_numbers.append(chunk_starting_page_number)
                split_start_indices.append(chunk_start_idx)

                # Get the number of sentences that overlap with the next chunk
                num_sentences_to_keep = self._number_of_sentences_to_keep(
                    sentences=current_chunk,
                    split_length=split_length,
                    split_overlap=split_overlap,
                )
                # Set up information for the new chunk
                if num_sentences_to_keep > 0:
                    # Processed sentences are the ones that are not overlapping with the next chunk
                    processed_sentences = current_chunk[:-num_sentences_to_keep]
                    chunk_starting_page_number += sum(sent.count("\f") for sent in processed_sentences)
                    chunk_start_idx += len("".join(processed_sentences))

                    # Next chunk starts with the sentences that were overlapping with the previous chunk
                    current_chunk = current_chunk[-num_sentences_to_keep:]
                    chunk_word_count = sum(len(s.split()) for s in current_chunk)
                else:
                    # Here processed_sentences is the same as current_chunk since there is no overlap
                    chunk_starting_page_number += sum(sent.count("\f") for sent in current_chunk)
                    chunk_start_idx += len("".join(current_chunk))
                    current_chunk = []
                    chunk_word_count = 0

        # Concatenate the sentences together within each split
        text_splits = []
        for split in list_of_splits:
            text = "".join(split)
            if len(text) > 0:
                text_splits.append(text)

        return text_splits, split_start_page_numbers, split_start_indices

    def _split_by_hanlp_sentence(self, doc: Document) -> List[Document]:
        """
        Split Chinese text into sentences.

        :param doc: The document to split.
        :return: A list of split documents.
        """
        if doc.content is None:
            return []

        split_docs = []
        result = self.chinese_sentence_split(doc.content)
        units = [sentence["sentence"] for sentence in result]

        if self.respect_sentence_boundary:
            text_splits, splits_pages, splits_start_idxs = self._concatenate_sentences_based_on_word_amount(
                sentences=units,
                split_length=self.split_length,
                split_overlap=self.split_overlap,
                granularity=self.granularity,
            )
        else:
            text_splits, splits_pages, splits_start_idxs = self._concatenate_units(
                elements=units,
                split_length=self.split_length,
                split_overlap=self.split_overlap,
                split_threshold=self.split_threshold,
            )
        metadata = deepcopy(doc.meta)
        metadata["source_id"] = doc.id
        split_docs += self._create_docs_from_splits(
            text_splits=text_splits, splits_pages=splits_pages, splits_start_idxs=splits_start_idxs, meta=metadata
        )

        return split_docs

    def _concatenate_units(
        self, elements: List[str], split_length: int, split_overlap: int, split_threshold: int
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Concatenates the elements into parts of split_length units.

        Keeps track of the original page number that each element belongs. If the length of the current units is less
        than the pre-defined `split_threshold`, it does not create a new split. Instead, it concatenates the current
        units with the last split, preventing the creation of excessively small splits.
        """

        # If the text is empty or consists only of whitespace, return empty lists
        if not elements or all(not elem.strip() for elem in elements):
            return [], [], []

        # If the text is too short to split, return it as a single chunk
        if len(elements) <= split_length:
            text = "".join(elements)
            return [text], [1], [0]

        # Otherwise, proceed as before
        step = split_length - split_overlap
        step = max(step, 1)
        text_splits: List[str] = []
        splits_pages: List[int] = []
        splits_start_idxs: List[int] = []
        cur_start_idx = 0
        cur_page = 1
        segments = windowed(elements, n=split_length, step=step)

        for seg in segments:
            current_units = [unit for unit in seg if unit is not None]
            txt = "".join(current_units)

            # check if length of current units is below split_threshold
            if len(current_units) < split_threshold and len(text_splits) > 0:
                # concatenate the last split with the current one
                text_splits[-1] += txt

            # NOTE: This line skips documents that have content=""
            elif len(txt) > 0:
                text_splits.append(txt)
                splits_pages.append(cur_page)
                splits_start_idxs.append(cur_start_idx)

            processed_units = current_units[: split_length - split_overlap]
            cur_start_idx += len("".join(processed_units))

            if self.split_by == "page":
                num_page_breaks = len(processed_units)
            else:
                num_page_breaks = sum(processed_unit.count("\f") for processed_unit in processed_units)

            cur_page += num_page_breaks

        return text_splits, splits_pages, splits_start_idxs

    def _create_docs_from_splits(
        self, text_splits: List[str], splits_pages: List[int], splits_start_idxs: List[int], meta: Dict[str, Any]
    ) -> List[Document]:
        """
        Creates Document objects from splits enriching them with page number and the metadata of the original document.
        """
        documents: List[Document] = []

        for i, (txt, split_idx) in enumerate(zip(text_splits, splits_start_idxs)):
            copied_meta = deepcopy(meta)
            copied_meta["page_number"] = splits_pages[i]
            copied_meta["split_id"] = i
            copied_meta["split_idx_start"] = split_idx
            doc = Document(content=txt, meta=copied_meta)
            documents.append(doc)

            if self.split_overlap <= 0:
                continue

            doc.meta["_split_overlap"] = []

            if i == 0:
                continue

            doc_start_idx = splits_start_idxs[i]
            previous_doc = documents[i - 1]
            previous_doc_start_idx = splits_start_idxs[i - 1]
            self._add_split_overlap_information(doc, doc_start_idx, previous_doc, previous_doc_start_idx)

        for d in documents:
            if d.content is not None:
                d.content = d.content.replace(" ", "")
        return documents

    @staticmethod
    def _add_split_overlap_information(
        current_doc: Document, current_doc_start_idx: int, previous_doc: Document, previous_doc_start_idx: int
    ) -> None:
        """
        Adds split overlap information to the current and previous Document's meta.

        :param current_doc: The Document that is being split.
        :param current_doc_start_idx: The starting index of the current Document.
        :param previous_doc: The Document that was split before the current Document.
        :param previous_doc_start_idx: The starting index of the previous Document.
        """
        if previous_doc.content is None or current_doc.content is None:
            return

        overlapping_range = (current_doc_start_idx - previous_doc_start_idx, len(previous_doc.content))

        if overlapping_range[0] < overlapping_range[1]:
            overlapping_str = previous_doc.content[overlapping_range[0] : overlapping_range[1]]

            if current_doc.content.startswith(overlapping_str):
                # add split overlap information to this Document regarding the previous Document
                current_doc.meta["_split_overlap"].append({"doc_id": previous_doc.id, "range": overlapping_range})

                # add split overlap information to previous Document regarding this Document
                overlapping_range = (0, overlapping_range[1] - overlapping_range[0])
                previous_doc.meta["_split_overlap"].append({"doc_id": current_doc.id, "range": overlapping_range})

    def _number_of_sentences_to_keep(self, sentences: List[str], split_length: int, split_overlap: int) -> int:
        """
        Returns the number of sentences to keep in the next chunk based on the `split_overlap` and `split_length`.

        :param sentences: The list of sentences to split.
        :param split_length: The maximum number of words in each split.
        :param split_overlap: The number of overlapping words in each split.
        :returns: The number of sentences to keep in the next chunk.
        """
        # If the split_overlap is 0, we don't need to keep any sentences
        if split_overlap == 0:
            return 0

        num_sentences_to_keep = 0
        num_words = 0

        for sent in reversed(sentences[1:]):
            num_words += len(self.chinese_tokenizer(sent))
            # If the number of words is larger than the split_length then don't add any more sentences
            if num_words > split_length:
                break
            num_sentences_to_keep += 1
            if num_words > split_overlap:
                break
        return num_sentences_to_keep

    def _split_by_function(self, doc: Document) -> List[Document]:
        """
        Split a document using a custom splitting function.

        :param doc: The document to split.
        :return: A list of split documents.
        """
        if doc.content is None:
            return []

        if self.splitting_function is None:
            msg = "No splitting function provided."
            raise ValueError(msg)

        splits = self.splitting_function(doc.content)
        if not isinstance(splits, list):
            msg = "The splitting function must return a list of strings."
            raise ValueError(msg)

        metadata = deepcopy(doc.meta)
        metadata["source_id"] = doc.id

        return self._create_docs_from_splits(
            text_splits=splits,
            splits_pages=[1] * len(splits),
            splits_start_idxs=[0] * len(splits),
            meta=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.
        """
        serialized = default_to_dict(
            self,
            split_by=self.split_by,
            split_length=self.split_length,
            split_overlap=self.split_overlap,
            split_threshold=self.split_threshold,
            respect_sentence_boundary=self.respect_sentence_boundary,
            granularity=self.granularity,
        )
        if self.splitting_function:
            serialized["init_parameters"]["splitting_function"] = serialize_callable(self.splitting_function)
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChineseDocumentSplitter":
        """
        Deserializes the component from a dictionary.
        """
        init_params = data.get("init_parameters", {})

        splitting_function = init_params.get("splitting_function", None)
        if splitting_function:
            init_params["splitting_function"] = deserialize_callable(splitting_function)

        return default_from_dict(cls, data)
