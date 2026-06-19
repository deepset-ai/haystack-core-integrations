# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import Mock

import pytest
import torch
from haystack.utils.device import ComponentDevice
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from haystack_integrations.common.transformers.utils import (
    _resolve_hf_device_map,
    _StopWordsCriteria,
)


def test_resolve_hf_device_map_only_device():
    model_kwargs = _resolve_hf_device_map(device=None, model_kwargs={})
    assert model_kwargs["device_map"] == ComponentDevice.resolve_device(None).to_hf()


def test_resolve_hf_device_map_only_device_map():
    model_kwargs = _resolve_hf_device_map(device=None, model_kwargs={"device_map": "cpu"})
    assert model_kwargs["device_map"] == "cpu"


def test_resolve_hf_device_map_device_and_device_map(caplog):
    with caplog.at_level(logging.WARNING):
        model_kwargs = _resolve_hf_device_map(
            device=ComponentDevice.from_str("cpu"), model_kwargs={"device_map": "cuda:0"}
        )
        assert "The parameters `device` and `device_map` from `model_kwargs` are both provided." in caplog.text
    assert model_kwargs["device_map"] == "cuda:0"


def test_stop_words_criteria_with_a_mocked_tokenizer():
    """
    Test that _StopWordsCriteria will caught stop word tokens in a continuous and sequential order in the input_ids
    """
    stop_words_id = torch.LongTensor([[73, 24621, 11937]])  # "unambiguously"
    # "This is ambiguously, but is unrelated."
    input_ids_one = torch.LongTensor([[100, 19, 24621, 11937, 6, 68, 19, 73, 3897, 5]])
    input_ids_two = torch.LongTensor([[100, 19, 73, 24621, 11937]])  # "This is unambiguously"

    mock_tokenizer = Mock(spec=PreTrainedTokenizerFast)
    mock_tokenizer.pad_token = "<pad>"
    stop_words_criteria = _StopWordsCriteria(tokenizer=mock_tokenizer, stop_words=["mock data"])
    stop_words_criteria.stop_ids = stop_words_id

    assert not stop_words_criteria(input_ids_one, scores=None)
    assert stop_words_criteria(input_ids_two, scores=None)


@pytest.mark.integration
def test_stop_words_criteria_using_hf_tokenizer():
    """
    Test that _StopWordsCriteria catches stop word tokens in a continuous and sequential order in the input_ids
    using a real Huggingface tokenizer.
    """

    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    criteria = _StopWordsCriteria(tokenizer=tokenizer, stop_words=["unambiguously"])

    text_one = "This is ambiguously, but is unrelated."
    generated_text_ids = tokenizer.encode(text_one, add_special_tokens=False, return_tensors="pt")
    assert criteria(generated_text_ids, scores=None) is False

    text_two = "This is unambiguously"
    generated_text_ids = tokenizer.encode(text_two, add_special_tokens=False, return_tensors="pt")
    assert criteria(generated_text_ids, scores=None) is True
