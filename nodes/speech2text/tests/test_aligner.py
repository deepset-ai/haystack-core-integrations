# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from haystack import Document

from speech2text.aligner import AeneasTranscriptAligner
from speech2text.errors import SpeechToTextNodeError


SAMPLES_PATH = Path(__file__).parent / "samples"


@pytest.mark.integration
class TestAligner:
    def test_aligner(self):
        doc = Document(
            content="this is the content of the document",
            content_type="text",
            meta={
                "name": "this is the content of the document.wav",
                "audio": {"content": {"path": SAMPLES_PATH / "this is the content of the document.wav"}},
            },
        )
        aligner = AeneasTranscriptAligner()
        results, _ = aligner.run(documents=[doc])

        assert results["documents"][0].meta.get("alignment", None)
