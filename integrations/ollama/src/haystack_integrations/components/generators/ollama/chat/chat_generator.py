import json
from typing import Any, Callable, Dict, List, Optional

import requests
from haystack import component
from haystack.dataclasses import ChatMessage, StreamingChunk
from requests import Response


@component
class OllamaChatGenerator:
    """
    Supports models running on Ollama, such as llama2 and mixtral.  Find the full list of supported models
    [here](https://ollama.ai/library).

    Usage example:
    ```python
    from haystack_integrations.components.generators.ollama import OllamaChatGenerator
    from haystack.dataclasses import ChatMessage

    generator = OllamaChatGenerator(model="zephyr",
                                url = "http://localhost:11434/api/chat",
                                generation_kwargs={
                                "num_predict": 100,
                                "temperature": 0.9,
                                })

    messages = [ChatMessage.from_system("\nYou are a helpful, respectful and honest assistant"),
    ChatMessage.from_user("What's Natural Language Processing?")]

    print(generator.run(messages=messages))
    ```
    """

    def __init__(
        self,
        model: str = "orca-mini",
        url: str = "http://localhost:11434/api/chat",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        template: Optional[str] = None,
        timeout: int = 120,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        :param model:
            The name of the model to use. The model should be available in the running Ollama instance.
        :param url:
            The URL of the chat endpoint of a running Ollama instance.
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, and others. See the available arguments in
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param template:
            The full prompt template (overrides what is defined in the Ollama Modelfile).
        :param timeout:
            The number of seconds before throwing a timeout error from the Ollama API.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        """

        self.timeout = timeout
        self.template = template
        self.generation_kwargs = generation_kwargs or {}
        self.url = url
        self.model = model
        self.streaming_callback = streaming_callback

    def _message_to_dict(self, message: ChatMessage) -> Dict[str, str]:
        return {"role": message.role.value, "content": message.content}

    def _create_json_payload(self, messages: List[ChatMessage], stream=False, generation_kwargs=None) -> Dict[str, Any]:
        """
        Returns A dictionary of JSON arguments for a POST request to an Ollama service
        """
        generation_kwargs = generation_kwargs or {}
        return {
            "messages": [self._message_to_dict(message) for message in messages],
            "model": self.model,
            "stream": stream,
            "template": self.template,
            "options": generation_kwargs,
        }

    def _build_message_from_ollama_response(self, ollama_response: Response) -> ChatMessage:
        """
        Converts the non-streaming response from the Ollama API to a ChatMessage.
        """
        json_content = ollama_response.json()
        message = ChatMessage.from_assistant(content=json_content["message"]["content"])
        message.meta.update({key: value for key, value in json_content.items() if key != "message"})
        return message

    def _convert_to_streaming_response(self, chunks: List[StreamingChunk]) -> Dict[str, List[Any]]:
        """
        Converts a list of chunks response required Haystack format.
        """

        replies = [ChatMessage.from_assistant("".join([c.content for c in chunks]))]
        meta = {key: value for key, value in chunks[0].meta.items() if key != "message"}

        return {"replies": replies, "meta": [meta]}

    def _build_chunk(self, chunk_response: Any) -> StreamingChunk:
        """
        Converts the response from the Ollama API to a StreamingChunk.
        """
        decoded_chunk = json.loads(chunk_response.decode("utf-8"))

        content = decoded_chunk["message"]["content"]
        meta = {key: value for key, value in decoded_chunk.items() if key != "message"}
        meta["role"] = decoded_chunk["message"]["role"]

        chunk_message = StreamingChunk(content, meta)
        return chunk_message

    def _handle_streaming_response(self, response) -> List[StreamingChunk]:
        """
        Handles Streaming response cases
        """
        chunks: List[StreamingChunk] = []
        for chunk in response.iter_lines():
            chunk_delta: StreamingChunk = self._build_chunk(chunk)
            chunks.append(chunk_delta)
            if self.streaming_callback is not None:
                self.streaming_callback(chunk_delta)
        return chunks

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Runs an Ollama Model on a given chat history.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, etc. See the
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param streaming_callback:
            A callback function that will be called with each response chunk in streaming mode.
        :returns: A dictionary with the following keys:
            - `replies`: The responses from the model
        """
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        stream = self.streaming_callback is not None

        json_payload = self._create_json_payload(messages, stream, generation_kwargs)

        response = requests.post(url=self.url, json=json_payload, timeout=self.timeout, stream=stream)

        # throw error on unsuccessful response
        response.raise_for_status()

        if stream:
            chunks: List[StreamingChunk] = self._handle_streaming_response(response)
            return self._convert_to_streaming_response(chunks)

        return {"replies": [self._build_message_from_ollama_response(response)]}
