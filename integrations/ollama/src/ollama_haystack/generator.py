from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from haystack import component


@dataclass
class OllamaResponse:
    model: str
    created_at: datetime
    response: str
    context: List[int]
    done: bool
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int

    def __post_init__(self):
        self.meta = {key: value for key, value in self.__dict__.items() if key != "response"}

    def as_haystack_generator_response(self) -> Dict[str, List]:
        """Returns replies and metadata in the format required by haystack"""
        return {"replies": [self.response], "meta": [self.meta]}


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

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def _post_args(self, prompt: str, generation_kwargs=None) -> Dict[str, Any]:
        """
        Returns A dictionary of arguments for a POST request to an Ollama service
        :param prompt: the prompt to generate a response for
        :param generation_kwargs:
        :return: A dictionary of arguments for a POST request to an Ollama service
        """
        if generation_kwargs is None:
            generation_kwargs = {}
        return {
            "url": self.url,
            "json": {
                "prompt": prompt,
                "model": self.model_name,
                "stream": False,
                "raw": self.raw,
                "template": self.template,
                "system": self.system_prompt,
                "options": generation_kwargs,
            },
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

        post_arguments = self._post_args(prompt, generation_kwargs)

        response = requests.post(url=post_arguments["url"], json=post_arguments["json"], timeout=self.timeout)

        # Throw error on unsuccessful response
        response.raise_for_status()

        ollama_response = OllamaResponse(**response.json())

        return ollama_response.as_haystack_generator_response()
