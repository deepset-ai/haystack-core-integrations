from typing import Union, List

import math
from abc import abstractmethod
import logging
from pathlib import Path

from tqdm import tqdm
from pydub import AudioSegment
import librosa
import torch
from haystack import Document
from haystack.nodes import BaseComponent



logger = logging.getLogger(__name__)


class BaseSpeechTranscriber(BaseComponent):
    """
    Transcribes audio files into Documents using a speech-to-text model.
    """

    outgoing_edges = 1

    def run(self, file_paths: List[Path]):  # type: ignore
        documents = []
        for audio_file in file_paths:

            complete_transcript = ""
            logger.info(f"Processing {audio_file}")
            for fragment_file in self.chunk(audio_file):
                complete_transcript += self.transcribe(fragment_file)

            documents.append(
                Document(
                    content=complete_transcript,
                    content_type="text",
                    meta={
                        "name": str(audio_file),
                        "audio": {
                            "content": audio_file
                        }
                    }
                )
            )

        return {"documents": documents}, "output_1"

    def run_batch(self):
        raise NotImplemented

    def chunk(self, path: Path):
        """
        Returns the input audio in chunks that can be processed by the transcriber model.

        Override to a no-op if the transcriber implementation does not need it
        """
        audio_format = path.suffix.replace(".", "")
        audio = AudioSegment.from_file(path, format=audio_format)
        n_fragments = math.ceil(audio.duration_seconds / self.fragment_length)
        fragment_path = Path(f"/tmp/[frag]__{path.name}")

        for fragment_id in tqdm(total=n_fragments):
            fragment = audio[
                fragment_id * self.fragment_length * 1000 : (fragment_id + 1) * self.fragment_length * 1000
            ]
            fragment.export(fragment_path, format=audio_format)
            yield fragment_path

    @abstractmethod
    def transcribe(self, audio_file: Path, sample_rate=16000) -> str:
        pass
