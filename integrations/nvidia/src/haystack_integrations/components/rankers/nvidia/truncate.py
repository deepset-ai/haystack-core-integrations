# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class RankerTruncateMode(str, Enum):
    """
    Specifies how inputs to the NVIDIA ranker components are truncated.
    If NONE, the input will not be truncated and an error returned instead.
    If END, the input will be truncated from the end.
    """

    NONE = "NONE"
    END = "END"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str) -> "RankerTruncateMode":
        """
        Create an truncate mode from a string.

        :param string:
            String to convert.
        :returns:
            Truncate mode.
        """
        return cls(string)
