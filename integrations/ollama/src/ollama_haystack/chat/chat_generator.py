from typing import Any, Callable, Dict, List, Optional

import requests
from haystack import component
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from requests import Response


@component
class OllamaChatGenerator:
    """
    Chat Generator based on Ollama. Ollama is a library for easily running LLMs locally.
    This component provides an interface to generate text using a LLM running in Ollama.
    """

    def __init__(
        self,
        model: str = "orca-mini",
        url: str = "http://localhost:11434/api/chat",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        template: Optional[str] = None,
        timeout: int = 30,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        self.streaming_callback = streaming_callback
        self.timeout = timeout
        self.template = template
        self.generation_kwargs = generation_kwargs or {}
        self.url = url
        self.model = model

    def _get_telemetry_data(self) -> Dict[str, str]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def _message_to_dict(self, message: ChatMessage) -> Dict[str, str]:
        role = "user"
        if message.role == ChatRole.ASSISTANT:
            role = "assistant"
        return {"role": role, "content": message.content}

    def _chat_history_to_dict(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        return [self._message_to_dict(message) for message in messages]

    def _create_json_payload(self, messages: List[ChatMessage], generation_kwargs=None) -> Dict[str, Any]:
        """
        Returns A dictionary of JSON arguments for a POST request to an Ollama service
        :param messages: A history of chat messages
        :param generation_kwargs:
        :return: A dictionary of arguments for a POST request to an Ollama service
        """
        generation_kwargs = generation_kwargs or {}
        return {
            "messages": self._chat_history_to_dict(messages),
            "model": self.model,
            "stream": bool(self.streaming_callback),
            "template": self.template,
            "options": generation_kwargs,
        }

    def _build_message(self, ollama_response: Response):
        """
        Converts the non-streaming response from the Ollama API to a ChatMessage.
        :param ollama_response: The completion returned by the Ollama API.
        :return: The ChatMessage.
        """
        json_content = ollama_response.json()
        message = ChatMessage.from_assistant(content=json_content["message"]["content"])
        message.metadata.update({key: value for key, value in json_content.items() if key != "message"})
        return message

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Run an Ollama Model on a given chat history.
        :param messages: A list of ChatMessage instances representing the input messages.
        :param generation_kwargs: Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, etc. See the
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :return: A dictionary of the replies containing their metadata
        """
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        json_payload = self._create_json_payload(messages, generation_kwargs)

        response = requests.post(url=self.url, json=json_payload, timeout=self.timeout)

        # throw error on unsuccessful response
        response.raise_for_status()

        # Todo: Implement streaming
        if self.streaming_callback:
            raise NotImplementedError

        return {"replies": [self._build_message(response)]}
