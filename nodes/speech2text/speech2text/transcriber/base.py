from typing import List

from abc import abstractmethod
import logging
from pathlib import Path

from haystack import Document
from haystack.nodes import BaseComponent


logger = logging.getLogger(__name__)


class BaseSpeechTranscriber(BaseComponent):
    """
    Transcribes audio files into Documents using a speech-to-text model.
    """

    outgoing_edges = 1

    def run(self, file_paths: List[Path]):  # type: ignore  # pylint: disable=arguments-differ
        documents = []
        for audio_file in file_paths:

            complete_transcript = ""
            logger.info("Processing %s", audio_file)
            for fragment_file in self.chunk(audio_file):
                complete_transcript += self.transcribe(fragment_file)

            documents.append(
                Document(
                    content=complete_transcript,
                    content_type="text",
                    meta={
                        "name": str(audio_file),
                        "audio": {"content": {"path": audio_file}},
                    },
                )
            )

        return {"documents": documents}, "output_1"

    def run_batch(self, *args, **kwargs):  # pylint: disable=arguments-differ
        raise NotImplementedError()

    @abstractmethod
    def chunk(self, path: Path):
        """
        Chunks the audio into smaller chunks that can be processed by the transcriber model.
        If the model can handle any length, implement with `yield path`.
        """

    @abstractmethod
    def transcribe(self, audio_file: Path, sample_rate=16000) -> str:
        """
        Performs the actual transcription.
        """
