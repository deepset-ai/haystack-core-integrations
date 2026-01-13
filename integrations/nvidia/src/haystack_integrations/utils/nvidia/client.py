# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class Client(Enum):
    """
    Client to use for NVIDIA NIMs.
    """

    NVIDIA_GENERATOR = "NvidiaGenerator"
    NVIDIA_TEXT_EMBEDDER = "NvidiaTextEmbedder"
    NVIDIA_DOCUMENT_EMBEDDER = "NvidiaDocumentEmbedder"
    NVIDIA_RANKER = "NvidiaRanker"

    def __str__(self) -> str:
        """Convert a Client enum to a string."""
        return self.value

    @staticmethod
    def from_str(string: str) -> "Client":
        """Convert a string to a Client enum."""
        enum_map = {e.value: e for e in Client}
        mode = enum_map.get(string)
        if mode is None:
            msg = f"Unknown client '{string}' to use for NVIDIA NIMs. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return mode
