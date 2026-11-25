# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack_integrations.components.embedders.cohere.utils import _merge_meta


def test_merge_meta_first_batch_unchanged():
    metadata = {
        "api_version": {"version": "2"},
        "billed_units": {"input_tokens": 7, "output_tokens": None},
        "tokens": None,
        "warnings": None,
    }

    assert _merge_meta({}, metadata) == metadata


def test_merge_meta_sums_usage_and_keeps_latest_metadata():
    accumulated = {
        "billed_units": {"input_tokens": 7, "output_tokens": None, "images": 2},
        "tokens": {"input_tokens": None, "output_tokens": 4},
        "api_version": {"version": "2", "is_experimental": True},
        "warnings": ["old warning"],
    }
    new = {
        "billed_units": {"input_tokens": 3, "output_tokens": 0, "images": None},
        "tokens": {"input_tokens": 6, "output_tokens": None},
        "api_version": {"version": "2", "is_experimental": None},
        "warnings": None,
    }

    assert _merge_meta(accumulated, new) == {
        "billed_units": {"input_tokens": 10, "output_tokens": 0, "images": 2},
        "tokens": {"input_tokens": 6, "output_tokens": 4},
        "api_version": {"version": "2", "is_experimental": None},
        "warnings": None,
    }


@pytest.mark.parametrize("new", [{}, {"billed_units": None, "tokens": None}])
def test_merge_meta_preserves_usage_when_not_reported(new):
    accumulated = {"billed_units": {"input_tokens": 7}, "tokens": {"input_tokens": 9}}

    assert _merge_meta(accumulated, new) == accumulated
