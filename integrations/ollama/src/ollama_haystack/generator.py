from typing import Any, Dict, List, Optional

import requests
from haystack import component
from requests import Response


def convert_to_haystack_response(ollama_response: Response) -> Dict[str, List[Any]]:
    """
    Convert a response from the Ollama API to the required Haystack format
    :param ollama_response: A response (requests library) from the Ollama API
    :return: A dictionary of the returned responses and metadata
    """
    json = ollama_response.json()

    replies = json["response"]
    meta = {key: value for key, value in json.items() if key != "response"}

    return {"replies": [replies], "meta": [meta]}


@component
class OllamaGenerator:
    def __init__(
        self,
        model_name: str = "orca-mini",
        url: str = "http://localhost:11434/api/generate",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        template: Optional[str] = None,
        raw: bool = False,
        timeout: int = 30,
    ):
        """

        :param model_name: The name of the LLM to use (from Ollama). Default is orca-mini
        :param url: The URL to a running Ollama instance. Default is http://localhost:11434/api/generate
        :param generation_kwargs: Optional arguments to pass to the Ollama generate function
        :param system_prompt: Optional system message to (overrides what is defined in the Modelfile)
        :param template: The full prompt or prompt template (overrides what is defined in the Modelfile)
        :param raw: If True, no formatting will be applied to the prompt. You may choose to use the raw parameter
            if you are specifying a full templated prompt in your request to the API.\
        :param timeout: The number of seconds before throwing a timeout error from the Ollama API. Default 3-
        """
        self.timeout = timeout
        self.raw = raw
        self.template = template
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.url = url
        self.generation_kwargs = generation_kwargs or {}

    def _get_telemetry_data(self) -> Dict[str, str]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def _json_payload(self, prompt: str, generation_kwargs=None) -> Dict[str, Any]:
        """
        Returns A dictionary of JSON arguments for a POST request to an Ollama service
        :param prompt: the prompt to generate a response for
        :param generation_kwargs:
        :return: A dictionary of arguments for a POST request to an Ollama service
        """
        if generation_kwargs is None:
            generation_kwargs = {}
        return {
            "prompt": prompt,
            "model": self.model_name,
            "stream": False,
            "raw": self.raw,
            "template": self.template,
            "system": self.system_prompt,
            "options": generation_kwargs,
        }

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(
        self,
        prompt: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Run an Ollama Model against a given prompt
        :param prompt: The prompt to generate a response for
        :param generation_kwargs: Additional model parameters listed in the documentation for the Ollama
            Modelfile such as temperature
        :return: A dictionary of the response and returned metadata
        """
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        json_payload = self._json_payload(prompt, generation_kwargs)

        response = requests.post(url=self.url, json=json_payload, timeout=self.timeout)

        # Throw error on unsuccessful response
        response.raise_for_status()

        return convert_to_haystack_response(response)
