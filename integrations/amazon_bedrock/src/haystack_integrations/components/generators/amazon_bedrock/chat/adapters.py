import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

from botocore.eventstream import EventStream
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from transformers import AutoTokenizer, PreTrainedTokenizer

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
        return self._extract_messages_from_response(self.response_body_message_key(), response_body)

    def get_stream_responses(self, stream: EventStream, stream_handler: Callable[[StreamingChunk], None]) -> List[str]:
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

    @staticmethod
    def _update_params(target_dict: Dict[str, Any], updates_dict: Dict[str, Any]) -> None:
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

        :param inference_kwargs: The inference kwargs to merge.
        :param default_params: The default params to start with.
        :return: The merged params.
        """
        # Start with a copy of default_params
        kwargs = default_params.copy()

        # Update the default params with self.generation_kwargs and finally inference_kwargs
        self._update_params(kwargs, self.generation_kwargs)
        self._update_params(kwargs, inference_kwargs)

        return kwargs

    def _ensure_token_limit(self, prompt: str) -> str:
        resize_info = self.check_prompt(prompt)
        if resize_info["prompt_length"] != resize_info["new_prompt_length"]:
            logger.warning(
                "The prompt was truncated from %s tokens to %s tokens so that the prompt length and "
                "the answer length (%s tokens) fit within the model's max token limit (%s tokens). "
                "Shorten the prompt or it will be cut off.",
                resize_info["prompt_length"],
                max(0, resize_info["model_max_length"] - resize_info["max_length"]),  # type: ignore
                resize_info["max_length"],
                resize_info["model_max_length"],
            )
        return str(resize_info["resized_prompt"])

    @abstractmethod
    def check_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Checks the prompt length and resizes it if necessary.

        :param prompt: The prompt to check.
        :return: A dictionary containing the resized prompt and additional information.
        """

    def _extract_messages_from_response(self, message_tag: str, response_body: Dict[str, Any]) -> List[ChatMessage]:
        metadata = {k: v for (k, v) in response_body.items() if k != message_tag}
        return [ChatMessage.from_assistant(response_body[message_tag], meta=metadata)]

    @abstractmethod
    def response_body_message_key(self) -> str:
        """Returns the key for the message in the response body."""

    @abstractmethod
    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        """Extracts the token from a streaming chunk."""


class AnthropicClaudeChatAdapter(BedrockModelChatAdapter):
    """
    Model adapter for the Anthropic Claude model.
    """

    ANTHROPIC_USER_TOKEN = "\n\nHuman:"
    ANTHROPIC_ASSISTANT_TOKEN = "\n\nAssistant:"

    def __init__(self, generation_kwargs: Dict[str, Any]):
        super().__init__(generation_kwargs)

        # We pop the model_max_length as it is not sent to the model
        # but used to truncate the prompt if needed
        # Anthropic Claude has a limit of at least 100000 tokens
        # https://docs.anthropic.com/claude/reference/input-and-output-sizes
        model_max_length = self.generation_kwargs.pop("model_max_length", 100000)

        # Truncate prompt if prompt tokens > model_max_length-max_length
        # (max_length is the length of the generated text)
        # TODO use Anthropic tokenizer to get the precise prompt length
        # See https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#token-counting
        self.prompt_handler = DefaultPromptHandler(
            tokenizer="gpt2",
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

        prepared_prompt = "".join(conversation) + AnthropicClaudeChatAdapter.ANTHROPIC_ASSISTANT_TOKEN + " "
        return self._ensure_token_limit(prepared_prompt)

    def check_prompt(self, prompt: str) -> Dict[str, Any]:
        return self.prompt_handler(prompt)

    def response_body_message_key(self) -> str:
        return "completion"

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        return chunk.get("completion", "")


class MetaLlama2ChatAdapter(BedrockModelChatAdapter):
    """
    Model adapter for the Meta Llama 2 models.
    """

    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{% set loop_messages = messages[1:] %}"
        "{% set system_message = messages[0]['content'] %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% set system_message = false %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
        "{% endif %}"
        "{% if loop.index0 == 0 and system_message != false %}"
        "{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}"
        "{% else %}"
        "{% set content = message['content'] %}"
        "{% endif %}"
        "{% if message['role'] == 'user' %}"
        "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ ' '  + content.strip() + ' ' + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
    )

    def __init__(self, generation_kwargs: Dict[str, Any]) -> None:
        super().__init__(generation_kwargs)
        # We pop the model_max_length as it is not sent to the model
        # but used to truncate the prompt if needed
        # Llama 2 has context window size of 4096 tokens
        # with some exceptions when the context window has been extended
        model_max_length = self.generation_kwargs.pop("model_max_length", 4096)

        # Use `google/flan-t5-base` as it's also BPE sentencepiece tokenizer just like llama 2
        # a) we should get good estimates for the prompt length (empirically close to llama 2)
        # b) we can use apply_chat_template with the template above to delineate ChatMessages
        tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.unk_token = "<unk>"
        self.prompt_handler = DefaultPromptHandler(
            tokenizer=tokenizer,
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
        prepared_prompt: str = self.prompt_handler.tokenizer.apply_chat_template(
            conversation=messages, tokenize=False, chat_template=self.chat_template
        )
        return self._ensure_token_limit(prepared_prompt)

    def check_prompt(self, prompt: str) -> Dict[str, Any]:
        return self.prompt_handler(prompt)

    def response_body_message_key(self) -> str:
        return "generation"

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        return chunk.get("generation", "")
