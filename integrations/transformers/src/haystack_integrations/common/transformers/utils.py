# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
from typing import Any

import torch
from haystack import logging
from haystack.dataclasses import AsyncStreamingCallbackT, ComponentInfo, StreamingChunk, SyncStreamingCallbackT
from haystack.utils.auth import Secret
from haystack.utils.device import ComponentDevice
from huggingface_hub import model_info

from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    TextStreamer,
)

logger = logging.getLogger(__name__)


def _resolve_hf_device_map(device: ComponentDevice | None, model_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """
    Update `model_kwargs` to include the keyword argument `device_map`.

    This method is useful you want to force loading a transformers model when using `AutoModel.from_pretrained` to
    use `device_map`.

    We handle the edge case where `device` and `device_map` is specified by ignoring the `device` parameter and printing
    a warning.

    :param device: The device on which the model is loaded. If `None`, the default device is automatically
        selected.
    :param model_kwargs: Additional HF keyword arguments passed to `AutoModel.from_pretrained`.
        For details on what kwargs you can pass, see the model's documentation.
    """
    model_kwargs = copy.copy(model_kwargs) or {}
    if model_kwargs.get("device_map"):
        if device is not None:
            logger.warning(
                "The parameters `device` and `device_map` from `model_kwargs` are both provided. "
                "Ignoring `device` and using `device_map`."
            )
        # Resolve device if device_map is provided in model_kwargs
        device_map = model_kwargs["device_map"]
    else:
        device_map = ComponentDevice.resolve_device(device).to_hf()

    # Set up device_map which allows quantized loading and multi device inference
    # requires accelerate which is always installed when using `pip install transformers[torch]`
    model_kwargs["device_map"] = device_map

    return model_kwargs


def _resolve_hf_pipeline_kwargs(
    huggingface_pipeline_kwargs: dict[str, Any],
    model: str,
    task: str | None,
    supported_tasks: list[str],
    device: ComponentDevice | None,
    token: Secret | None,
) -> dict[str, Any]:
    """
    Resolve the HuggingFace pipeline keyword arguments based on explicit user inputs.

    :param huggingface_pipeline_kwargs: Dictionary containing keyword arguments used to initialize a
        Hugging Face pipeline.
    :param model: The name or path of a Hugging Face model for on the HuggingFace Hub.
    :param task: The task for the Hugging Face pipeline.
    :param supported_tasks: The list of supported tasks to check the task of the model against. If the task of the model
        is not present within this list then a ValueError is thrown.
    :param device: The device on which the model is loaded. If `None`, the default device is automatically
        selected. If a device/device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
    :param token: The token to use as HTTP bearer authorization for remote files.
        If the token is also specified in the `huggingface_pipeline_kwargs`, this parameter will be ignored.
    """
    resolved_token = token.resolve_value() if token else None
    # check if the huggingface_pipeline_kwargs contain the essential parameters
    # otherwise, populate them with values from other init parameters
    huggingface_pipeline_kwargs.setdefault("model", model)
    huggingface_pipeline_kwargs.setdefault("token", resolved_token)

    resolved_device = ComponentDevice.resolve_device(device)
    resolved_device.update_hf_kwargs(huggingface_pipeline_kwargs, overwrite=False)

    # task identification and validation
    task = task or huggingface_pipeline_kwargs.get("task")
    if task is None and isinstance(huggingface_pipeline_kwargs["model"], str):
        task = model_info(huggingface_pipeline_kwargs["model"], token=huggingface_pipeline_kwargs["token"]).pipeline_tag

    if task not in supported_tasks:
        msg = f"Task '{task}' is not supported. The supported tasks are: {', '.join(supported_tasks)}."
        raise ValueError(msg)
    huggingface_pipeline_kwargs["task"] = task
    return huggingface_pipeline_kwargs


class _StopWordsCriteria(StoppingCriteria):
    """
    Stops text generation in HuggingFace generators if any one of the stop words is generated.

    Note: When a stop word is encountered, the generation of new text is stopped.
    However, if the stop word is in the prompt itself, it can stop generating new text
    prematurely after the first token. This is particularly important for LLMs designed
    for dialogue generation. For these models, like for example mosaicml/mpt-7b-chat,
    the output includes both the new text and the original prompt. Therefore, it's important
    to make sure your prompt has no stop words.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        stop_words: list[str],
        device: str | torch.device = "cpu",
    ) -> None:
        """Creates an instance of _StopWordsCriteria."""
        super().__init__()
        # check if tokenizer is a valid tokenizer
        if not isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            msg = (
                f"Invalid tokenizer provided for _StopWordsCriteria - {tokenizer}. "
                f"Please provide a valid tokenizer from the HuggingFace Transformers library."
            )
            raise TypeError(msg)
        if not tokenizer.pad_token:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        encoded_stop_words = tokenizer(stop_words, add_special_tokens=False, padding=True, return_tensors="pt")
        self.stop_ids = encoded_stop_words.input_ids.to(device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:  # noqa: ARG002
        """Check if any of the stop words are generated in the current text generation step."""
        for stop_id in self.stop_ids:
            found_stop_word = self.is_stop_word_found(input_ids, stop_id)
            if found_stop_word:
                return True
        return False

    @staticmethod
    def is_stop_word_found(generated_text_ids: torch.Tensor, stop_id: torch.Tensor) -> bool:
        """
        Performs phrase matching.

        Checks if a sequence of stop tokens appears in a continuous or sequential order within the generated text.
        """
        generated_text_ids = generated_text_ids[-1]
        len_generated_text_ids = generated_text_ids.size(0)
        len_stop_id = stop_id.size(0)
        return all(generated_text_ids[len_generated_text_ids - len_stop_id :].eq(stop_id))


class _HFTokenStreamingHandler(TextStreamer):
    """
    Streaming handler for TransformersChatGenerator.

    Note: This is a helper class for TransformersChatGenerator enabling streaming
    of generated text via Haystack SyncStreamingCallbackT callbacks.

    Do not use this class directly.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        stream_handler: SyncStreamingCallbackT,
        stop_words: list[str] | None = None,
        component_info: ComponentInfo | None = None,
    ) -> None:
        """Creates an instance of _HFTokenStreamingHandler."""
        super().__init__(tokenizer=tokenizer, skip_prompt=True)
        self.token_handler = stream_handler
        self.stop_words = stop_words or []
        self.component_info = component_info
        self._call_counter = 0

    def on_finalized_text(self, word: str, stream_end: bool = False) -> None:
        """Callback function for handling the generated text."""
        self._call_counter += 1
        word_to_send = word + "\n" if stream_end else word
        if word_to_send.strip() not in self.stop_words:
            self.token_handler(
                StreamingChunk(
                    content=word_to_send, index=0, start=self._call_counter == 1, component_info=self.component_info
                )
            )


class _AsyncHFTokenStreamingHandler(TextStreamer):
    """
    Async streaming handler for TransformersChatGenerator.

    Note: This is a helper class for TransformersChatGenerator enabling
    async streaming of generated text via Haystack Callable[StreamingChunk, Awaitable[None]] callbacks.

    Do not use this class directly.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        stream_handler: AsyncStreamingCallbackT,
        stop_words: list[str] | None = None,
        component_info: ComponentInfo | None = None,
    ) -> None:
        """Creates an instance of _AsyncHFTokenStreamingHandler."""
        super().__init__(tokenizer=tokenizer, skip_prompt=True)
        self.token_handler = stream_handler
        self.stop_words = stop_words or []
        self.component_info = component_info
        self._queue: asyncio.Queue[StreamingChunk] = asyncio.Queue()

    def on_finalized_text(self, word: str, stream_end: bool = False) -> None:
        """Synchronous callback that puts chunks in a queue."""
        word_to_send = word + "\n" if stream_end else word
        if word_to_send.strip() not in self.stop_words:
            self._queue.put_nowait(StreamingChunk(content=word_to_send, component_info=self.component_info))

    async def process_queue(self) -> None:
        """Process the queue of streaming chunks."""
        while True:
            try:
                chunk = await self._queue.get()
                await self.token_handler(chunk)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
