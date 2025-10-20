# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class EmbeddingTruncateMode(Enum):
    """
    Specifies how inputs to the NVIDIA embedding components are truncated.
    If START, the input will be truncated from the start.
    If END, the input will be truncated from the end.
    If NONE, an error will be returned (if the input is too long).
    """

    START = "START"
    END = "END"
    NONE = "NONE"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str) -> "EmbeddingTruncateMode":
        """
        Create an truncate mode from a string.

        :param string:
            String to convert.
        :returns:
            Truncate mode.
        """
        enum_map = {e.value: e for e in EmbeddingTruncateMode}
        opt_mode = enum_map.get(string)
        if opt_mode is None:
            msg = f"Unknown truncate mode '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return opt_mode
