from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any, Dict, List
from urllib.parse import urljoin

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
        url: str = "http://localhost:11434/",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        template: Optional[str] = None,
        raw: bool = False,
    ):
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

    def _post_args(self, prompt: str, generation_kwargs=None):
        if generation_kwargs is None:
            generation_kwargs = {}
        return {
            "url": urljoin(self.url, "/api/generate"),
            "json": {
                "prompt": prompt,
                "model": self.model_name,
                "stream": False,
                "raw": self.raw,
                "options": generation_kwargs,
            },
        }

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        response = requests.post(**self._post_args(prompt, generation_kwargs))

        response.raise_for_status()

        ollama_response = OllamaResponse(**response.json())

        return ollama_response.as_haystack_generator_response()
