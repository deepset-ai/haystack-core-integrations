# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from haystack.utils.export_utils import print_answers


def print_audio_answers(results: dict, details: str = "all", max_text_len: Optional[int] = None):
    """
    Utility function to print results of Haystack pipelines in case they contain speech document/answers
    :param results: Results that the pipeline returned.
    :param details: Defines the level of details to print. Possible values: minimum, medium, all.
    :param max_text_len: Specifies the maximum allowed length for a text field.
        If you don't want to shorten the text, set this value to None.
    :return: None
    """
    # Defines the fields to keep in the Answer for each detail level
    fields_to_keep_by_level = {
        "minimum": ["answer", "answer_audio", "context", "context_audio"],
        "medium": ["answer", "answer_audio", "context", "context_audio", "score"],
        "all": [],
    }
    print_answers(
        results=results,
        details=details,
        max_text_len=max_text_len,  # FIXME once the Haystack PR is merged, add this:   _fields=fields_to_keep_by_level["details"]
    )
