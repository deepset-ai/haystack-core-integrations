from typing import List

from abc import abstractmethod
import logging

from tqdm import tqdm
from pydantic import dataclass

from haystack import Span, Document
from haystack.nodes import BaseComponent


logger = logging.getLogger(__name__)


@dataclass
class AudioTranscriptionAlignment:
    offset_audio: Span
    offset_text: Span
    aligned_string: str



class BaseTranscriptAligner(BaseComponent):
    """
    Aligns an audio file containing speech with its transcription.
    """
    outgoing_edges = 1

    def run(self, documents: List[Document]):  # type: ignore
        for document in tqdm(documents):
            document.meta["alignment"] = self.align(document)
        return {"documents": documents}, "output_1"

    def run_batch(self, documents: List[List[Document]]):
        for document_list in documents:
            for document in tqdm(document_list):
                document.meta["alignment"] = self.align(document)
        return {"documents": documents}, "output_1"


    @abstractmethod
    def align(self, document: Document):
        pass

