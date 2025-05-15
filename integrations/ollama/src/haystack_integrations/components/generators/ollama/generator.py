from typing import Any, Callable, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import StreamingChunk
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

from ollama import Client, GenerateResponse


def _convert_ollama_meta_to_openai_format(intput_response_dict: Dict) -> Dict:
    """
    Converts Ollama metadata format to OpenAI-metadata format for standardization.

    The conversion follows these mapping rules:
    - 'done_reason' -> 'finish_reason'
    - 'created_at' -> 'completion_start_time'
    - 'eval_count' and 'prompt_eval_count' -> 'usage' dict with token counts
    - All other fields are preserved as-is except 'response'.

    Input Ollama-metadata format example:
    {
        'model': 'llama3.2:3b',
        'created_at': '2025-05-15T14:11:57.8920338Z',
        'done': True,
        'done_reason': 'stop',
        'total_duration': 871519900,
        'load_duration': 44376100,
        'prompt_eval_count': 29,
        'prompt_eval_duration': 380700800,
        'eval_count': 8,
        'eval_duration': 445131300,
        'context': [28006,
         9125,
         ...
         220,
         18]
    }

    Returned OpenAI-metadata format example:
    {
        'model': 'llama3.2:3b',
        'finish_reason': 'stop',
        'usage': {
            'completion_tokens': 8,
            'prompt_tokens': 29,
            'total_tokens': 37,
        }
        'created_at': '2025-05-15T14:11:57.8920338Z',
        'done': True,
        'total_duration': 871519900,
        'load_duration': 44376100,
        'prompt_eval_duration': 380700800,
        'eval_duration': 445131300,
        'context': [28006,
         9125,
         ...
         220,
         18]
    }
    """

    meta = {key: value for key, value in intput_response_dict.items() if key != "response"}

    if "done_reason" in meta:
        meta["finish_reason"] = meta.pop("done_reason")
    if "eval_count" in meta and "prompt_eval_count" in meta:
        eval_count = meta.pop("eval_count")
        prompt_eval_count = meta.pop("prompt_eval_count")
        meta["usage"] = {
            "completion_tokens": eval_count,
            "prompt_tokens": prompt_eval_count,
            "total_tokens": eval_count + prompt_eval_count,
        }
    return meta


@component
class OllamaGenerator:
    """
    Provides an interface to generate text using an LLM running on Ollama.

    Usage example:
    ```python
    from haystack_integrations.components.generators.ollama import OllamaGenerator

    generator = OllamaGenerator(model="zephyr",
                                url = "http://localhost:11434",
                                generation_kwargs={
                                "num_predict": 100,
                                "temperature": 0.9,
                                })

    print(generator.run("Who is the best American actor?"))
    ```
    """

    def __init__(
        self,
        model: str = "orca-mini",
        url: str = "http://localhost:11434",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        template: Optional[str] = None,
        raw: bool = False,
        timeout: int = 120,
        keep_alive: Optional[Union[float, str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        :param model:
            The name of the model to use. The model should be available in the running Ollama instance.
        :param url:
            The URL of a running Ollama instance.
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, and others. See the available arguments in
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param system_prompt:
            Optional system message (overrides what is defined in the Ollama Modelfile).
        :param template:
            The full prompt template (overrides what is defined in the Ollama Modelfile).
        :param raw:
            If True, no formatting will be applied to the prompt. You may choose to use the raw parameter
            if you are specifying a full templated prompt in your API request.
        :param timeout:
            The number of seconds before throwing a timeout error from the Ollama API.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param keep_alive:
            The option that controls how long the model will stay loaded into memory following the request.
            If not set, it will use the default value from the Ollama (5 minutes).
            The value can be set to:
            - a duration string (such as "10m" or "24h")
            - a number in seconds (such as 3600)
            - any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
            - '0' which will unload the model immediately after generating a response.
        """
        self.timeout = timeout
        self.raw = raw
        self.template = template
        self.system_prompt = system_prompt
        self.model = model
        self.url = url
        self.keep_alive = keep_alive
        self.generation_kwargs = generation_kwargs or {}
        self.streaming_callback = streaming_callback

        self._client = Client(host=self.url, timeout=self.timeout)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
              Dictionary with serialized data.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            timeout=self.timeout,
            raw=self.raw,
            template=self.template,
            system_prompt=self.system_prompt,
            model=self.model,
            url=self.url,
            keep_alive=self.keep_alive,
            generation_kwargs=self.generation_kwargs,
            streaming_callback=callback_name,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OllamaGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    def _convert_to_response(self, ollama_response: GenerateResponse) -> Dict[str, List[Any]]:
        """
        Converts a response from the Ollama API to the required Haystack format.
        """
        response_dict = ollama_response.model_dump()
        reply = response_dict["response"]
        meta = _convert_ollama_meta_to_openai_format(response_dict)

        return {"replies": [reply], "meta": [meta]}

    def _convert_to_streaming_response(self, chunks: List[StreamingChunk]) -> Dict[str, List[Any]]:
        """
        Converts a list of chunks response required Haystack format.
        """

        replies = ["".join([c.content for c in chunks])]

        # Convert the metadata from the last chunk
        meta = _convert_ollama_meta_to_openai_format(chunks[-1].meta)

        return {"replies": replies, "meta": [meta]}

    def _handle_streaming_response(
        self, response: Any, streaming_callback: Optional[Callable[[StreamingChunk], None]]
    ) -> List[StreamingChunk]:
        """
        Handles Streaming response cases
        """
        chunks: List[StreamingChunk] = []
        for chunk in response:
            chunk_delta: StreamingChunk = self._build_chunk(chunk)
            chunks.append(chunk_delta)
            if streaming_callback is not None:
                streaming_callback(chunk_delta)
        return chunks

    def _build_chunk(self, chunk_response: Any) -> StreamingChunk:
        """
        Converts the response from the Ollama API to a StreamingChunk.
        """
        chunk_response_dict = chunk_response.model_dump()
        content = chunk_response_dict["response"]
        meta = {key: value for key, value in chunk_response_dict.items() if key != "response"}

        chunk_message = StreamingChunk(content, meta)
        return chunk_message

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(
        self,
        prompt: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        *,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Runs an Ollama Model on the given prompt.

        :param prompt:
            The prompt to generate a response for.
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, and others. See the available arguments in
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :returns: A dictionary with the following keys:
            - `replies`: The responses from the model
            - `meta`: The metadata collected during the run
        """
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        resolved_streaming_callback = streaming_callback or self.streaming_callback
        stream = resolved_streaming_callback is not None

        response = self._client.generate(
            model=self.model,
            prompt=prompt,
            stream=stream,
            keep_alive=self.keep_alive,
            options=generation_kwargs,
        )

        if stream:
            chunks: List[StreamingChunk] = self._handle_streaming_response(response, resolved_streaming_callback)
            return self._convert_to_streaming_response(chunks)

        return self._convert_to_response(response)
