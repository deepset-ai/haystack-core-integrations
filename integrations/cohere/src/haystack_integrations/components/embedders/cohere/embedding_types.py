# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum


class EmbeddingTypes(Enum):
    """
    Supported types for Cohere embeddings.

    FLOAT: Default float embeddings. Valid for all models.
    INT8: Signed int8 embeddings. Valid for only v3 models.
    UINT8: Unsigned int8 embeddings. Valid for only v3 models.
    BINARY: Signed binary embeddings. Valid for only v3 models.
    UBINARY: Unsigned binary embeddings. Valid for only v3 models.
    """

    FLOAT = "float"
    INT8 = "int8"
    UINT8 = "uint8"
    BINARY = "binary"
    UBINARY = "ubinary"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(string: str) -> "EmbeddingTypes":
        """
        Convert a string to an EmbeddingTypes enum.
        """
        enum_map = {e.value: e for e in EmbeddingTypes}
        embedding_type = enum_map.get(string.lower())
        if embedding_type is None:
            msg = f"Unknown embedding type '{string}'. Supported types are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return embedding_type
