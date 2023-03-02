# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Dict, Any, Union

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

            logger.info("Processing %s", audio_file)
            output = self.transcribe(audio_file)
            transcript = output.pop("text")

            documents.append(
                Document(
                    content=transcript,
                    content_type="text",
                    meta={
                        "name": str(audio_file),
                        "audio": {"path": audio_file, **output},
                    },
                )
            )

        return {"documents": documents}, "output_1"

    def run_batch(self):  # pylint: disable=arguments-differ
        raise NotImplementedError()

    @abstractmethod
    def transcribe(self, audio_file: Union[Path, str], sample_rate=16000) -> Dict[str, Any]:
        """
        Performs the actual transcription.
        """
        raise NotImplementedError()
