from typing import Any, Callable, Dict, List, Optional

import requests
from haystack import component
from haystack.dataclasses import StreamingChunk
from requests import Response
import json


@component
class OllamaGenerator:
    """
    Generator based on Ollama. Ollama is a library for easily running LLMs locally.
    This component provides an interface to generate text using a LLM running in Ollama.
    """

    def __init__(
        self,
        model: str = "orca-mini",
        url: str = "http://localhost:11434/api/generate",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        template: Optional[str] = None,
        raw: bool = False,
        timeout: int = 120,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        :param model: The name of the model to use. The model should be available in the running Ollama instance.
            Default is "orca-mini".
        :param url: The URL of the generation endpoint of a running Ollama instance.
            Default is "http://localhost:11434/api/generate".
        :param generation_kwargs: Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, and others. See the available arguments in
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param system_prompt: Optional system message (overrides what is defined in the Ollama Modelfile).
        :param template: The full prompt template (overrides what is defined in the Ollama Modelfile).
        :param raw: If True, no formatting will be applied to the prompt. You may choose to use the raw parameter
            if you are specifying a full templated prompt in your API request.
        :param timeout: The number of seconds before throwing a timeout error from the Ollama API.
            Default is 120 seconds.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        """
        self.timeout = timeout
        self.raw = raw
        self.template = template
        self.system_prompt = system_prompt
        self.model = model
        self.url = url
        self.generation_kwargs = generation_kwargs or {}
        self.streaming_callback = streaming_callback

    def _create_json_payload(self, prompt: str, stream: bool, generation_kwargs=None) -> Dict[str, Any]:
        """
        Returns a dictionary of JSON arguments for a POST request to an Ollama service.
        :param prompt: The prompt to generate a response for.
        :param generation_kwargs: Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, and others. See the available arguments in
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :return: A dictionary of arguments for a POST request to an Ollama service.
        """
        generation_kwargs = generation_kwargs or {}
        return {
            "prompt": prompt,
            "model": self.model,
            "stream": stream,
            "raw": self.raw,
            "template": self.template,
            "system": self.system_prompt,
            "options": generation_kwargs,
        }

    def _convert_to_haystack_response(self, ollama_response: Response) -> Dict[str, List[Any]]:
        """
        Convert a response from the Ollama API to the required Haystack format.
        :param ollama_response: A response (requests library) from the Ollama API.
        :return: A dictionary of the returned responses and metadata.
        """
        
        resp_dict = ollama_response.json()

        replies = [resp_dict["response"]]
        meta = {key: value for key, value in resp_dict.items() if key != "response"}

        return {"replies": replies, "meta": [meta]}
    
    def _convert_to_response(self, chunks: List[StreamingChunk]) -> Dict[str, List[Any]]:
        """
        Convert a list of chunks response required Haystack format.
        :param chunks: List of StreamingChunks
        :return: A dictionary of the returned responses and metadata.
        """
        
        replies = ["".join([c.content for c in chunks])]
        meta = {key: value for key, value in chunks[0].meta.items() if key != "response"}

        return {"replies": replies, "meta": [meta]}

    def _handle_streaming_response(self, response) -> List[StreamingChunk]:
        """ Handles Streaming response case

        :param response: streaming response from ollama api.
        :return: The List[StreamingChunk].
        """
        chunks: List[StreamingChunk] = []
        for chunk in response.iter_lines():
            chunk_delta: StreamingChunk = self._build_chunk(chunk)
            chunks.append(chunk_delta)
            self.streaming_callback(chunk_delta)
        return chunks

    def _build_chunk(self, chunk_response: Any) -> StreamingChunk:
        """
        Converts the response from the Ollama API to a StreamingChunk.
        :param chunk: The chunk returned by the Ollama API.
        :return: The StreamingChunk.
        """
        decoded_chunk = json.loads(chunk_response.decode("utf-8"))

        content = decoded_chunk['response']
        meta = {key: value for key, value in decoded_chunk.items() if key != "response"}

        chunk_message = StreamingChunk(content, meta)
        return chunk_message

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(
        self,
        prompt: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Run an Ollama Model on the given prompt.
        :param prompt: The prompt to generate a response for.
        :param generation_kwargs: Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, and others. See the available arguments in
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :return: A dictionary of the response and returned metadata
        """
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        stream = self.streaming_callback is not None

        json_payload = self._create_json_payload(prompt, stream, generation_kwargs)

        response = requests.post(url=self.url, json=json_payload, timeout=self.timeout, stream=stream)

        # throw error on unsuccessful response
        response.raise_for_status()

        if stream:
            chunks: List[StreamingChunk] = self._handle_streaming_response(response)
            return self._convert_to_response(chunks)

        return self._convert_to_haystack_response(response)
