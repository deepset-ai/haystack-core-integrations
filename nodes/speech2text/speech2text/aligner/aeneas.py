import json
import logging
import datetime
from copy import deepcopy
from pathlib import Path

try:
    from aeneas.executetask import ExecuteTask
    from aeneas.task import Task
except ImportError as e:
    logging.exception(
        "'aeneas' not found. To use AeneasTranscriptAligner, please install aeneas with `pip install aeneas`. "
        "Make sure you install also the additional dependencies as explained here: "
        "https://github.com/readbeyond/aeneas/blob/master/wiki/INSTALL.md#manual-procedure", e
    )

from haystack import Span, Document

from speech2text.aligner.base import BaseTranscriptAligner, AudioTranscriptionAlignment


logger = logging.getLogger(__name__)


class AeneasTranscriptAligner(BaseTranscriptAligner):
    """
    Aligns an audio file with its transcription using the AENEAS forced aligmnent implementation.
    """

    def __init__(self):
        super().__init__()
        self.task = Task(config_string="task_language=eng|is_text_type=plain|os_task_file_format=json")

    def align(self, document: Document):
        # This step seems to be unavoidable :(
        transcript_path = Path(f"/tmp/aeneas_transcript.txt")
        with open(transcript_path, "w") as tf:
            tf.write(deepcopy(document.content).replace(" ", "\n"))

        print(document.meta)
        raw_alignments = self._align(audio_file=document.meta["audio"]["content"]["path"], transcript_file=transcript_path)

        accumulator = 0
        alignments = []
        for raw_alignment in raw_alignments:
            if raw_alignment["lines"]:
                word = raw_alignment["lines"][0]
                word_len = len(word) + 1  # 1 for the whitespace
                alignment = AudioTranscriptionAlignment(
                    Span(int(float(raw_alignment["begin"]) * 1000), int(float(raw_alignment["end"]) * 1000)),
                    Span(accumulator, accumulator + word_len),
                    word,
                )
                alignments.append(alignment)
                accumulator += word_len

        return alignments

    def _align(self, audio_file: Path, transcript_file: Path):
        """
        Generates the alignments and returns a list of AudioAlignment objects.
        """
        logger.debug(f"Aligning {audio_file} with {transcript_file}...")
        self.task.audio_file_path_absolute = str(audio_file.absolute())
        self.task.text_file_path_absolute = str(transcript_file.absolute())

        start = datetime.datetime.now()
        ExecuteTask(self.task).execute()

        logger.debug(f"Alignment complete. It required %s sec.", datetime.datetime.now() - start)
        return json.loads(self.task.sync_map.json_string).get("fragments", [])
