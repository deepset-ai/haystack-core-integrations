# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64

import numpy as np

PERPLEXITY_FLOAT_ENCODING_FORMAT_ERROR = (
    "Perplexity's /v1/embeddings does not support encoding_format='float'; use 'base64_int8' or 'base64_binary'."
)

SUPPORTED_ENCODING_FORMATS = {"base64_int8", "base64_binary"}


def validate_encoding_format(encoding_format: str) -> str:
    """
    Validate Perplexity's embedding encoding format.
    """
    if encoding_format == "float":
        raise ValueError(PERPLEXITY_FLOAT_ENCODING_FORMAT_ERROR)
    if encoding_format not in SUPPORTED_ENCODING_FORMATS:
        supported_formats = "', '".join(sorted(SUPPORTED_ENCODING_FORMATS))
        msg = f"Unsupported encoding_format='{encoding_format}'. Use '{supported_formats}'."
        raise ValueError(msg)
    return encoding_format


def decode_embedding(embedding: str, encoding_format: str) -> list[float]:
    """
    Decode a Perplexity base64 embedding into Haystack's list[float] representation.
    """
    raw_embedding = base64.b64decode(embedding)
    if encoding_format == "base64_int8":
        return np.frombuffer(raw_embedding, dtype=np.int8).astype(np.float32).tolist()
    if encoding_format == "base64_binary":
        return np.unpackbits(np.frombuffer(raw_embedding, dtype=np.uint8)).astype(np.float32).tolist()

    msg = f"Unsupported encoding_format='{encoding_format}'."
    raise ValueError(msg)
