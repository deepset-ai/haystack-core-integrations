import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .handlers import TokenStreamingHandler


class BedrockModelAdapter(ABC):
    """
    Base class for Amazon Bedrock model adapters.

    Each subclass of this class is designed to address the unique specificities of a particular LLM it adapts,
    focusing on preparing the requests and extracting the responses from the Amazon Bedrock hosted LLMs.
    """

    def __init__(self, model_kwargs: Dict[str, Any], max_length: Optional[int]) -> None:
        self.model_kwargs = model_kwargs
        self.max_length = max_length

    @abstractmethod
    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        """
        Prepares the body for the Amazon Bedrock request.
        Each subclass should implement this method to prepare the request body for the specific model.

        :param prompt: The prompt to be sent to the model.
        :param inference_kwargs: Additional keyword arguments passed to the handler.
        :returns: A dictionary containing the body for the request.
        """

    def get_responses(self, response_body: Dict[str, Any]) -> List[str]:
        """
        Extracts the responses from the Amazon Bedrock response.

        :param response_body: The response body from the Amazon Bedrock request.
        :returns: A list of responses.
        """
        completions = self._extract_completions_from_response(response_body)
        responses = [completion.lstrip() for completion in completions]
        return responses

    def get_stream_responses(self, stream, stream_handler: TokenStreamingHandler) -> List[str]:
        """
        Extracts the responses from the Amazon Bedrock streaming response.

        :param stream: The streaming response from the Amazon Bedrock request.
        :param stream_handler: The handler for the streaming response.
        :returns: A list of string responses.
        """
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
        :param inference_kwargs: The inference kwargs.
        :param default_params: The default params.
        :returns: A dictionary containing the merged params.
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
        """
        Extracts the responses from the Amazon Bedrock response.

        :param response_body: The response body from the Amazon Bedrock request.
        :returns: A list of string responses.
        """

    @abstractmethod
    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        """
        Extracts the token from a streaming chunk.

        :param chunk: The streaming chunk.
        :returns: A string token.
        """


class AnthropicClaudeAdapter(BedrockModelAdapter):
    """
    Adapter for the Anthropic Claude models.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        """
        Prepares the body for the Claude model

        :param prompt: The prompt to be sent to the model.
        :param inference_kwargs: Additional keyword arguments passed to the handler.
        :returns: A dictionary with the following keys:
            - `prompt`: The prompt to be sent to the model.
            - specified inference parameters.
        """
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
        """
        Extracts the responses from the Amazon Bedrock response.

        :param response_body: The response body from the Amazon Bedrock request.
        :returns: A list of string responses.
        """
        return [response_body["completion"]]

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        """
        Extracts the token from a streaming chunk.

        :param chunk: The streaming chunk.
        :returns: A string token.
        """
        return chunk.get("completion", "")


class CohereCommandAdapter(BedrockModelAdapter):
    """
    Adapter for the Cohere Command model.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        """
        Prepares the body for the Command model

        :param prompt: The prompt to be sent to the model.
        :param inference_kwargs: Additional keyword arguments passed to the handler.
        :returns: A dictionary with the following keys:
            - `prompt`: The prompt to be sent to the model.
            - specified inference parameters.
        """
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
        """
        Extracts the responses from the Cohere Command model response.

        :param response_body: The response body from the Amazon Bedrock request.
        :returns: A list of string responses.
        """
        responses = [generation["text"] for generation in response_body["generations"]]
        return responses

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        """
        Extracts the token from a streaming chunk.

        :param chunk: The streaming chunk.
        :returns: A string token.
        """
        return chunk.get("text", "")


class AI21LabsJurassic2Adapter(BedrockModelAdapter):
    """
    Model adapter for AI21 Labs' Jurassic 2 models.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        """Prepares the body for the Jurassic 2 model.

        :param prompt: The prompt to be sent to the model.
        :param inference_kwargs: Additional keyword arguments passed to the handler.
        :returns: A dictionary with the following keys:
            -  `prompt`: The prompt to be sent to the model.
            - specified inference parameters.
        """
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
    Adapter for Amazon's Titan models.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        """
        Prepares the body for the Titan model

        :param prompt: The prompt to be sent to the model.
        :param inference_kwargs: Additional keyword arguments passed to the handler.
        :returns: A dictionary with the following keys
            - `inputText`: The prompt to be sent to the model.
            - specified inference parameters.
        """
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
        """
        Extracts the responses from the Titan model response.

        :param response_body: The response body for Titan model response.
        :returns: A list of string responses.
        """
        responses = [result["outputText"] for result in response_body["results"]]
        return responses

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        """
        Extracts the token from a streaming chunk.

        :param chunk: The streaming chunk.
        :returns: A string token.
        """
        return chunk.get("outputText", "")


class MetaLlama2ChatAdapter(BedrockModelAdapter):
    """
    Adapter for Meta's Llama2 models.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        """
        Prepares the body for the Llama2 model

        :param prompt: The prompt to be sent to the model.
        :param inference_kwargs: Additional keyword arguments passed to the handler.
        :returns: A dictionary with the following keys:
            - `prompt`: The prompt to be sent to the model.
            - specified inference parameters.
        """
        default_params = {
            "max_gen_len": self.max_length,
            "temperature": None,
            "top_p": None,
        }
        params = self._get_params(inference_kwargs, default_params)

        body = {"prompt": prompt, **params}
        return body

    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        """
        Extracts the responses from the Llama2 model response.

        :param response_body: The response body from the Llama2 model request.
        :returns: A list of string responses.
        """
        return [response_body["generation"]]

    def _extract_token_from_stream(self, chunk: Dict[str, Any]) -> str:
        """
        Extracts the token from a streaming chunk.

        :param chunk: The streaming chunk.
        :returns: A string token.
        """
        return chunk.get("generation", "")
