# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from haystack_integrations.components.embedders.cohere.utils import _merge_meta, get_async_response, get_response


def _mock_response():
    response = MagicMock()
    response.embeddings = [("float", [[0.1, 0.2]])]
    response.meta = None
    return response


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


@pytest.mark.parametrize("embedding_dimension", [None, 512])
def test_get_response_sends_output_dimension_only_when_set(embedding_dimension):
    client = MagicMock()
    client.embed.return_value = _mock_response()

    get_response(
        client,
        ["hello"],
        "embed-v5.0-pro",
        "search_query",
        "END",
        embedding_dimension=embedding_dimension,
    )

    kwargs = client.embed.call_args.kwargs
    if embedding_dimension is None:
        assert "output_dimension" not in kwargs
    else:
        assert kwargs["output_dimension"] == embedding_dimension


@pytest.mark.asyncio
@pytest.mark.parametrize("embedding_dimension", [None, 512])
async def test_get_async_response_sends_output_dimension_only_when_set(embedding_dimension):
    client = MagicMock()
    client.embed = AsyncMock(return_value=_mock_response())

    await get_async_response(
        cohere_async_client=client,
        texts=["hello"],
        model_name="embed-v5.0-pro",
        input_type="search_query",
        truncate="END",
        embedding_dimension=embedding_dimension,
    )

    kwargs = client.embed.call_args.kwargs
    if embedding_dimension is None:
        assert "output_dimension" not in kwargs
    else:
        assert kwargs["output_dimension"] == embedding_dimension
