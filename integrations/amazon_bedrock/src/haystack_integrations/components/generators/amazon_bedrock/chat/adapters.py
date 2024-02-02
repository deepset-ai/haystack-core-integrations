import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk

from haystack_integrations.components.generators.amazon_bedrock.handlers import DefaultPromptHandler

logger = logging.getLogger(__name__)


class BedrockModelChatAdapter(ABC):
    """
    Base class for Amazon Bedrock chat model adapters.
    """

    def __init__(self, generation_kwargs: Dict[str, Any]) -> None:
        self.generation_kwargs = generation_kwargs

    @abstractmethod
    def prepare_body(self, messages: List[ChatMessage], **inference_kwargs) -> Dict[str, Any]:
        """Prepares the body for the Amazon Bedrock request."""

    def get_responses(self, response_body: Dict[str, Any]) -> List[ChatMessage]:
        """Extracts the responses from the Amazon Bedrock response."""
        return self._extract_messages_from_response(response_body)

    def get_stream_responses(self, stream, stream_handler: Callable[[StreamingChunk], None]) -> List[str]:
        tokens: List[str] = []
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                decoded_chunk = json.loads(chunk["bytes"].decode("utf-8"))
                token = self._extract_token_from_stream(decoded_chunk)
                # take all the rest key/value pairs from the chunk, add them to the metadata
                stream_metadata = {k: v for (k, v) in decoded_chunk.items() if v != token}
                stream_chunk = StreamingChunk(content=token, meta=stream_metadata)
                # callback the stream handler with StreamingChunk
                stream_handler(stream_chunk)
                tokens.append(token)
        responses = ["".join(tokens).lstrip()]
        return responses

    def _update_params(self, target_dict: Dict[str, Any], updates_dict: Dict[str, Any]) -> None:
        """
        Updates target_dict with values from updates_dict. Merges lists instead of overriding them.

        :param target_dict: The dictionary to update.
        :param updates_dict: The dictionary with updates.
        """
        for key, value in updates_dict.items():
            if key in target_dict and isinstance(target_dict[key], list) and isinstance(value, list):
                # Merge lists and remove duplicates
                target_dict[key] = sorted(set(target_dict[key] + value))
            else:
                # Override the value in target_dict
                target_dict[key] = value

    def _get_params(self, inference_kwargs: Dict[str, Any], default_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merges params from inference_kwargs with the default params and self.generation_kwargs.
        Uses a helper function to merge lists or override values as necessary.
        """
        # Start with a copy of default_params
        kwargs = default_params.copy()

        # Update the default params with self.generation_kwargs and finally inference_kwargs
        self._update_params(kwargs, self.generation_kwargs)
        self._update_params(kwargs, inference_kwargs)

        return kwargs

    @abstractmethod
    def _extract_messages_from_response(self, response_body: Dict[str, Any]) -> List[ChatMessage]:
        """Extracts the responses from the Amazon Bedrock response."""

    @abstractmethod
    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        """Extracts the token from a streaming chunk."""


class AnthropicClaudeChatAdapter(BedrockModelChatAdapter):
    """
    Model adapter for the Anthropic Claude model.
    """

    ANTHROPIC_USER_TOKEN = "\n\nHuman:"
    ANTHROPIC_ASSISTANT_TOKEN = "\n\nAssistant:"

    def __init__(self, generation_kwargs: Dict[str, Any]) -> None:
        super().__init__(generation_kwargs)

        # We pop the model_max_length as it is not sent to the model
        # but used to truncate the prompt if needed
        # Anthropic Claude has a limit of at least 100000 tokens
        # https://docs.anthropic.com/claude/reference/input-and-output-sizes
        model_max_length = self.generation_kwargs.get("model_max_length", 100000)

        # Truncate prompt if prompt tokens > model_max_length-max_length
        # (max_length is the length of the generated text)
        # TODO use Anthropic tokenizer to get the precise prompt length
        # See https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#token-counting
        self.prompt_handler = DefaultPromptHandler(
            model="gpt2",
            model_max_length=model_max_length,
            max_length=self.generation_kwargs.get("max_tokens_to_sample") or 512,
        )

    def prepare_body(self, messages: List[ChatMessage], **inference_kwargs) -> Dict[str, Any]:
        default_params = {
            "max_tokens_to_sample": self.generation_kwargs.get("max_tokens_to_sample") or 512,
            "stop_sequences": ["\n\nHuman:"],
        }

        # combine stop words with default stop sequences, remove stop_words as Anthropic does not support it
        stop_sequences = inference_kwargs.get("stop_sequences", []) + inference_kwargs.pop("stop_words", [])
        if stop_sequences:
            inference_kwargs["stop_sequences"] = stop_sequences
        params = self._get_params(inference_kwargs, default_params)
        body = {"prompt": self.prepare_chat_messages(messages=messages), **params}
        return body

    def prepare_chat_messages(self, messages: List[ChatMessage]) -> str:
        conversation = []
        for index, message in enumerate(messages):
            if message.is_from(ChatRole.USER):
                conversation.append(f"{AnthropicClaudeChatAdapter.ANTHROPIC_USER_TOKEN} {message.content.strip()}")
            elif message.is_from(ChatRole.ASSISTANT):
                conversation.append(f"{AnthropicClaudeChatAdapter.ANTHROPIC_ASSISTANT_TOKEN} {message.content.strip()}")
            elif message.is_from(ChatRole.FUNCTION):
                error_message = "Anthropic does not support function calls."
                raise ValueError(error_message)
            elif message.is_from(ChatRole.SYSTEM) and index == 0:
                # Until we transition to the new chat message format system messages will be ignored
                # see https://docs.anthropic.com/claude/reference/messages_post for more details
                logger.warning(
                    "System messages are not fully supported by the current version of Claude and will be ignored."
                )
            else:
                invalid_role = f"Invalid role {message.role} for message {message.content}"
                raise ValueError(invalid_role)

        return "".join(conversation) + AnthropicClaudeChatAdapter.ANTHROPIC_ASSISTANT_TOKEN + " "

    def _extract_messages_from_response(self, response_body: Dict[str, Any]) -> List[ChatMessage]:
        metadata = {k: v for (k, v) in response_body.items() if k != "completion"}
        return [ChatMessage.from_assistant(response_body["completion"], meta=metadata)]

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        return chunk.get("completion", "")


class MetaLlama2ChatAdapter(BedrockModelChatAdapter):
    """
    Model adapter for the Meta Llama model(s).
    """

    def __init__(self, generation_kwargs: Dict[str, Any]) -> None:
        super().__init__(generation_kwargs)
        # We pop the model_max_length as it is not sent to the model
        # but used to truncate the prompt if needed
        # Llama 2 has context window size of 4096 tokens
        model_max_length = self.generation_kwargs.get("model_max_length", 4096)
        # Truncate prompt if prompt tokens > model_max_length-max_length
        self.prompt_handler = DefaultPromptHandler(
            model="meta-llama/Llama-2-7b-chat-hf",
            model_max_length=model_max_length,
            max_length=self.generation_kwargs.get("max_gen_len") or 512,
        )

    def prepare_body(self, messages: List[ChatMessage], **inference_kwargs) -> Dict[str, Any]:
        default_params = {"max_gen_len": self.generation_kwargs.get("max_gen_len") or 512}

        # combine stop words with default stop sequences, remove stop_words as MetaLlama2 does not support it
        stop_sequences = inference_kwargs.get("stop_sequences", []) + inference_kwargs.pop("stop_words", [])
        if stop_sequences:
            inference_kwargs["stop_sequences"] = stop_sequences
        params = self._get_params(inference_kwargs, default_params)
        body = {"prompt": self.prepare_chat_messages(messages=messages), **params}
        return body

    def prepare_chat_messages(self, messages: List[ChatMessage]) -> str:
        prepared_prompt: str = self.prompt_handler.tokenizer.apply_chat_template(conversation=messages, tokenize=False)
        return prepared_prompt

    def _extract_messages_from_response(self, response_body: Dict[str, Any]) -> List[ChatMessage]:
        metadata = {k: v for (k, v) in response_body.items() if k != "generation"}
        return [ChatMessage.from_assistant(response_body["generation"], meta=metadata)]

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        return chunk.get("generation", "")
