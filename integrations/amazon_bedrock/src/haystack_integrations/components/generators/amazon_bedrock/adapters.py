import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .handlers import TokenStreamingHandler


class BedrockModelAdapter(ABC):
    """
    Base class for Amazon Bedrock model adapters.
    """

    def __init__(self, model_kwargs: Dict[str, Any], max_length: Optional[int]) -> None:
        self.model_kwargs = model_kwargs
        self.max_length = max_length

    @abstractmethod
    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        """Prepares the body for the Amazon Bedrock request."""

    def get_responses(self, response_body: Dict[str, Any]) -> List[str]:
        """Extracts the responses from the Amazon Bedrock response."""
        completions = self._extract_completions_from_response(response_body)
        responses = [completion.lstrip() for completion in completions]
        return responses

    def get_stream_responses(self, stream, stream_handler: TokenStreamingHandler) -> List[str]:
        tokens: List[str] = []
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                decoded_chunk = json.loads(chunk["bytes"].decode("utf-8"))
                token = self._extract_token_from_stream(decoded_chunk)
                tokens.append(stream_handler(token, event_data=decoded_chunk))
        responses = ["".join(tokens).lstrip()]
        return responses

    def _get_params(self, inference_kwargs: Dict[str, Any], default_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merges the default params with the inference kwargs and model kwargs.

        Includes param if it's in kwargs or its default is not None (i.e. it is actually defined).
        """
        kwargs = self.model_kwargs.copy()
        kwargs.update(inference_kwargs)
        return {
            param: kwargs.get(param, default)
            for param, default in default_params.items()
            if param in kwargs or default is not None
        }

    @abstractmethod
    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        """Extracts the responses from the Amazon Bedrock response."""

    @abstractmethod
    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        """Extracts the token from a streaming chunk."""


class AnthropicClaudeAdapter(BedrockModelAdapter):
    """
    Model adapter for the Anthropic's Claude model.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        default_params = {
            "max_tokens_to_sample": self.max_length,
            "stop_sequences": ["\n\nHuman:"],
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }
        params = self._get_params(inference_kwargs, default_params)

        body = {"prompt": f"\n\nHuman: {prompt}\n\nAssistant:", **params}
        return body

    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        return [response_body["completion"]]

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        return chunk.get("completion", "")


class CohereCommandAdapter(BedrockModelAdapter):
    """
    Model adapter for the Cohere's Command model.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        default_params = {
            "max_tokens": self.max_length,
            "stop_sequences": None,
            "temperature": None,
            "p": None,
            "k": None,
            "return_likelihoods": None,
            "stream": None,
            "logit_bias": None,
            "num_generations": None,
            "truncate": None,
        }
        params = self._get_params(inference_kwargs, default_params)

        body = {"prompt": prompt, **params}
        return body

    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        responses = [generation["text"] for generation in response_body["generations"]]
        return responses

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        return chunk.get("text", "")


class AI21LabsJurassic2Adapter(BedrockModelAdapter):
    """
    Model adapter for AI21 Labs' Jurassic 2 models.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        default_params = {
            "maxTokens": self.max_length,
            "stopSequences": None,
            "temperature": None,
            "topP": None,
            "countPenalty": None,
            "presencePenalty": None,
            "frequencyPenalty": None,
            "numResults": None,
        }
        params = self._get_params(inference_kwargs, default_params)

        body = {"prompt": prompt, **params}
        return body

    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        responses = [completion["data"]["text"] for completion in response_body["completions"]]
        return responses

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        msg = "Streaming is not supported for AI21 Jurassic 2 models."
        raise NotImplementedError(msg)


class AmazonTitanAdapter(BedrockModelAdapter):
    """
    Model adapter for Amazon's Titan models.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        default_params = {
            "maxTokenCount": self.max_length,
            "stopSequences": None,
            "temperature": None,
            "topP": None,
        }
        params = self._get_params(inference_kwargs, default_params)

        body = {"inputText": prompt, "textGenerationConfig": params}
        return body

    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        responses = [result["outputText"] for result in response_body["results"]]
        return responses

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        return chunk.get("outputText", "")


class MetaLlama2ChatAdapter(BedrockModelAdapter):
    """
    Model adapter for Meta's Llama 2 Chat models.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        default_params = {
            "max_gen_len": self.max_length,
            "temperature": None,
            "top_p": None,
        }
        params = self._get_params(inference_kwargs, default_params)

        body = {"prompt": prompt, **params}
        return body

    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        return [response_body["generation"]]

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        return chunk.get("generation", "")
