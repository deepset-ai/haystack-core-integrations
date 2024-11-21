# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum


class JinaReaderMode(Enum):
    """
    Enum representing modes for the Jina Reader.

    Modes:
        READ: Process a URL and return the textual content of the page.
        SEARCH: Search the web and return the textual content of the most relevant pages.
        GROUND: Call the grounding engine to perform fact checking.

    """

    READ = "read"
    SEARCH = "search"
    GROUND = "ground"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str) -> "JinaReaderMode":
        """
        Create the reader mode from a string.

        :param string:
            String to convert.
        :returns:
            Reader mode.
        """
        enum_map = {e.value: e for e in JinaReaderMode}
        reader_mode = enum_map.get(string)
        if reader_mode is None:
            msg = f"Unknown reader mode '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return reader_mode
