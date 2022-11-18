from typing import List

from abc import abstractmethod
import logging
from dataclasses import dataclass

from tqdm import tqdm

from haystack import Span, Document
from haystack.nodes import BaseComponent


logger = logging.getLogger(__name__)


@dataclass
class AudioTranscriptionAlignment:
    """
    Dataclass representing an alignment unit with its position in the text, in the audio, and the aligned text itself
    """

    position_audio: Span
    position_text: Span
    aligned_text: str


class BaseTranscriptAligner(BaseComponent):
    """
    Aligns an audio file containing speech with its transcription.
    """

    outgoing_edges = 1

    def run(self, documents: List[Document]):  # type: ignore  # pylint: disable=arguments-differ
        for document in tqdm(documents):
            document.meta["alignment"] = self.align(document)
        return {"documents": documents}, "output_1"

    def run_batch(self, documents: List[List[Document]]):  # type: ignore  # pylint: disable=arguments-differ
        for document_list in documents:
            for document in tqdm(document_list):
                document.meta["alignment"] = self.align(document)
        return {"documents": documents}, "output_1"

    @abstractmethod
    def align(self, document: Document):
        """
        Create a list of alignment units matching each word with its position in the original audio.
        """
