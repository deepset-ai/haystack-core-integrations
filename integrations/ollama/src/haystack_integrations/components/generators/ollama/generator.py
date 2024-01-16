from typing import Any, Dict, List, Optional

import requests
from haystack import component
from requests import Response


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
        """
        self.timeout = timeout
        self.raw = raw
        self.template = template
        self.system_prompt = system_prompt
        self.model = model
        self.url = url
        self.generation_kwargs = generation_kwargs or {}

    def _create_json_payload(self, prompt: str, generation_kwargs=None) -> Dict[str, Any]:
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
            "stream": False,
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

        json_payload = self._create_json_payload(prompt, generation_kwargs)

        response = requests.post(url=self.url, json=json_payload, timeout=self.timeout)

        # throw error on unsuccessful response
        response.raise_for_status()

        return self._convert_to_haystack_response(response)
