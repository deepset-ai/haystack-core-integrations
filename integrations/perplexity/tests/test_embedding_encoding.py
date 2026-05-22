# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack_integrations.components.embedders.perplexity.embedding_encoding import (
    decode_embedding,
    validate_encoding_format,
)


def test_validate_encoding_format_rejects_unsupported_format():
    with pytest.raises(ValueError) as exc_info:
        validate_encoding_format("base64_float16")

    assert str(exc_info.value) == "Unsupported encoding_format='base64_float16'. Use 'base64_binary', 'base64_int8'."


def test_decode_embedding_rejects_unsupported_format():
    with pytest.raises(ValueError) as exc_info:
        decode_embedding("", "base64_float16")

    assert str(exc_info.value) == "Unsupported encoding_format='base64_float16'."
