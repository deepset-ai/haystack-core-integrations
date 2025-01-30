# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack_integrations.components.embedders.nvidia import EmbeddingTruncateMode


class TestEmbeddingTruncateMode:
    @pytest.mark.parametrize(
        "mode, expected",
        [
            ("START", EmbeddingTruncateMode.START),
            ("END", EmbeddingTruncateMode.END),
            ("NONE", EmbeddingTruncateMode.NONE),
            (EmbeddingTruncateMode.START, EmbeddingTruncateMode.START),
            (EmbeddingTruncateMode.END, EmbeddingTruncateMode.END),
            (EmbeddingTruncateMode.NONE, EmbeddingTruncateMode.NONE),
        ],
    )
    def test_init_with_valid_mode(self, mode, expected):
        assert EmbeddingTruncateMode(mode) == expected

    def test_init_with_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError):
            invalid_mode = "INVALID"
            EmbeddingTruncateMode(invalid_mode)

    @pytest.mark.parametrize(
        "mode, expected",
        [
            ("START", EmbeddingTruncateMode.START),
            ("END", EmbeddingTruncateMode.END),
            ("NONE", EmbeddingTruncateMode.NONE),
        ],
    )
    def test_from_str_with_valid_mode(self, mode, expected):
        assert EmbeddingTruncateMode.from_str(mode) == expected

    def test_from_str_with_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError):
            invalid_mode = "INVALID"
            EmbeddingTruncateMode.from_str(invalid_mode)
