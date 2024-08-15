import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from haystack.dataclasses import StreamingChunk


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

    def get_stream_responses(self, stream, streaming_callback: Callable[[StreamingChunk], None]) -> List[str]:
        """
        Extracts the responses from the Amazon Bedrock streaming response.

        :param stream: The streaming response from the Amazon Bedrock request.
        :param streaming_callback: The handler for the streaming response.
        :returns: A list of string responses.
        """
        streaming_chunks: List[StreamingChunk] = []
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                decoded_chunk = json.loads(chunk["bytes"].decode("utf-8"))
                streaming_chunk: StreamingChunk = self._build_streaming_chunk(decoded_chunk)
                streaming_chunks.append(streaming_chunk)
                streaming_callback(streaming_chunk)

        responses = ["".join(streaming_chunk.content for streaming_chunk in streaming_chunks).lstrip()]
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
    def _build_streaming_chunk(self, chunk: Dict[str, Any]) -> StreamingChunk:
        """
        Extracts the content and meta from a streaming chunk.

        :param chunk: The streaming chunk as dict.
        :returns: A StreamingChunk object.
        """


class AnthropicClaudeAdapter(BedrockModelAdapter):
    """
    Adapter for the Anthropic Claude models.
    """

    def __init__(self, model_kwargs: Dict[str, Any], max_length: Optional[int]) -> None:
        self.use_messages_api = model_kwargs.get("use_messages_api", True)
        super().__init__(model_kwargs, max_length)

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        """
        Prepares the body for the Claude model

        :param prompt: The prompt to be sent to the model.
        :param inference_kwargs: Additional keyword arguments passed to the handler.
        :returns: A dictionary with the following keys:
            - `prompt`: The prompt to be sent to the model.
            - specified inference parameters.
        """
        if self.use_messages_api:
            default_params: Dict[str, Any] = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_length,
                "system": None,
                "stop_sequences": None,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            }
            params = self._get_params(inference_kwargs, default_params)

            body = {"messages": [{"role": "user", "content": prompt}], **params}
        else:
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
        if self.use_messages_api:
            return [content["text"] for content in response_body["content"]]

        return [response_body["completion"]]

    def _build_streaming_chunk(self, chunk: Dict[str, Any]) -> StreamingChunk:
        """
        Extracts the content and meta from a streaming chunk.

        :param chunk: The streaming chunk as dict.
        :returns: A StreamingChunk object.
        """
        if self.use_messages_api:
            return StreamingChunk(content=chunk.get("delta", {}).get("text", ""), meta=chunk)

        return StreamingChunk(content=chunk.get("completion", ""), meta=chunk)


class MistralAdapter(BedrockModelAdapter):
    """
    Adapter for the Mistral models.
    """

    def prepare_body(self, prompt: str, **inference_kwargs) -> Dict[str, Any]:
        """
        Prepares the body for the Mistral model

        :param prompt: The prompt to be sent to the model.
        :param inference_kwargs: Additional keyword arguments passed to the handler.
        :returns: A dictionary with the following keys:
            - `prompt`: The prompt to be sent to the model.
            - specified inference parameters.
        """
        default_params: Dict[str, Any] = {
            "max_tokens": self.max_length,
            "stop": [],
            "temperature": None,
            "top_p": None,
            "top_k": None,
        }
        params = self._get_params(inference_kwargs, default_params)
        # Add the instruction tag to the prompt if it's not already there
        formatted_prompt = f"<s>[INST] {prompt} [/INST]" if "INST" not in prompt else prompt
        return {"prompt": formatted_prompt, **params}

    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        """
        Extracts the responses from the Amazon Bedrock response.

        :param response_body: The response body from the Amazon Bedrock request.
        :returns: A list of string responses.
        """
        return [output.get("text", "") for output in response_body.get("outputs", [])]

    def _build_streaming_chunk(self, chunk: Dict[str, Any]) -> StreamingChunk:
        """
        Extracts the content and meta from a streaming chunk.

        :param chunk: The streaming chunk as dict.
        :returns: A StreamingChunk object.
        """
        content = ""
        chunk_list = chunk.get("outputs", [])
        if chunk_list:
            content = chunk_list[0].get("text", "")
        return StreamingChunk(content=content, meta=chunk)


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

    def _build_streaming_chunk(self, chunk: Dict[str, Any]) -> StreamingChunk:
        """
        Extracts the content and meta from a streaming chunk.

        :param chunk: The streaming chunk as dict.
        :returns: A StreamingChunk object.
        """
        return StreamingChunk(content=chunk.get("text", ""), meta=chunk)


class CohereCommandRAdapter(BedrockModelAdapter):
    """
    Adapter for the Cohere Command R models.
    """

    def prepare_body(self, prompt: str, **inference_kwargs: Any) -> Dict[str, Any]:
        """
        Prepares the body for the Command model

        :param prompt: The prompt to be sent to the model.
        :param inference_kwargs: Additional keyword arguments passed to the handler.
        :returns: A dictionary with the following keys:
            - `prompt`: The prompt to be sent to the model.
            - specified inference parameters.
        """
        default_params = {
            "chat_history": None,
            "documents": None,
            "search_query_only": None,
            "preamble": None,
            "max_tokens": self.max_length,
            "temperature": None,
            "p": None,
            "k": None,
            "prompt_truncation": None,
            "frequency_penalty": None,
            "presence_penalty": None,
            "seed": None,
            "return_prompt": None,
            "tools": None,
            "tool_results": None,
            "stop_sequences": None,
            "raw_prompting": None,
        }
        params = self._get_params(inference_kwargs, default_params)

        body = {"message": prompt, **params}
        return body

    def _extract_completions_from_response(self, response_body: Dict[str, Any]) -> List[str]:
        """
        Extracts the responses from the Cohere Command model response.

        :param response_body: The response body from the Amazon Bedrock request.
        :returns: A list of string responses.
        """
        responses = [response_body["text"]]
        return responses

    def _build_streaming_chunk(self, chunk: Dict[str, Any]) -> StreamingChunk:
        """
        Extracts the content and meta from a streaming chunk.

        :param chunk: The streaming chunk as dict.
        :returns: A StreamingChunk object.
        """
        token: str = chunk.get("text", "")
        return StreamingChunk(content=token, meta=chunk)


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

    def _build_streaming_chunk(self, chunk: Dict[str, Any]) -> StreamingChunk:
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

    def _build_streaming_chunk(self, chunk: Dict[str, Any]) -> StreamingChunk:
        """
        Extracts the content and meta from a streaming chunk.

        :param chunk: The streaming chunk as dict.
        :returns: A StreamingChunk object.
        """
        return StreamingChunk(content=chunk.get("outputText", ""), meta=chunk)


class MetaLlamaAdapter(BedrockModelAdapter):
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

    def _build_streaming_chunk(self, chunk: Dict[str, Any]) -> StreamingChunk:
        """
        Extracts the content and meta from a streaming chunk.

        :param chunk: The streaming chunk as dict.
        :returns: A StreamingChunk object.
        """
        return StreamingChunk(content=chunk.get("generation", ""), meta=chunk)
