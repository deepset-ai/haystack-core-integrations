from typing import Any, Dict, List, Optional

import requests
from haystack import component
from haystack.dataclasses import ChatMessage
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
        timeout: int = 120,
    ):
        """
        :param model: The name of the model to use. The model should be available in the running Ollama instance.
            Default is "orca-mini".
        :param url: The URL of the chat endpoint of a running Ollama instance.
            Default is "http://localhost:11434/api/chat".
        :param generation_kwargs: Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, and others. See the available arguments in
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param template: The full prompt template (overrides what is defined in the Ollama Modelfile).
        :param timeout: The number of seconds before throwing a timeout error from the Ollama API.
            Default is 120 seconds.
        """

        self.timeout = timeout
        self.template = template
        self.generation_kwargs = generation_kwargs or {}
        self.url = url
        self.model = model

    def _message_to_dict(self, message: ChatMessage) -> Dict[str, str]:
        return {"role": message.role.value, "content": message.content}

    def _create_json_payload(self, messages: List[ChatMessage], generation_kwargs=None) -> Dict[str, Any]:
        """
        Returns A dictionary of JSON arguments for a POST request to an Ollama service
        :param messages: A history/list of chat messages
        :param generation_kwargs:
        :return: A dictionary of arguments for a POST request to an Ollama service
        """
        generation_kwargs = generation_kwargs or {}
        return {
            "messages": [self._message_to_dict(message) for message in messages],
            "model": self.model,
            "stream": False,
            "template": self.template,
            "options": generation_kwargs,
        }

    def _build_message_from_ollama_response(self, ollama_response: Response) -> ChatMessage:
        """
        Converts the non-streaming response from the Ollama API to a ChatMessage.
        :param ollama_response: The completion returned by the Ollama API.
        :return: The ChatMessage.
        """
        json_content = ollama_response.json()
        message = ChatMessage.from_assistant(content=json_content["message"]["content"])
        message.meta.update({key: value for key, value in json_content.items() if key != "message"})
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

        return {"replies": [self._build_message_from_ollama_response(response)]}
