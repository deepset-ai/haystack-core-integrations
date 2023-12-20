import dataclasses
import logging
import os
from typing import Optional, List, Callable, Dict, Any

from haystack import component, default_from_dict, default_to_dict
from haystack.components.generators.utils import serialize_callback_handler, deserialize_callback_handler
from haystack.dataclasses import StreamingChunk, ChatMessage

logger = logging.getLogger(__name__)

import requests

@component
class OllamaGenerator:
    """
    Enables text generation using Ollama (Run large language models (LLMs) locally.
    See [Ollama AI](https://ollama.ai) for more details.

    """

    def __init__(
        self,
        model_name: str = "orca-mini:3b",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        generation_endpoint: str = "http://localhost:11434/api/generate",
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of Ollama.

        """

        self.model_name = model_name
        self.generation_kwargs = generation_kwargs or {}
        self.streaming_callback = streaming_callback
        self.generation_endpoint = generation_endpoint

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        """

        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        json_response = requests.post(
            url=self.generation_endpoint,
            json={
                "prompt": prompt,
                "model": self.model_name,
                "stream": False,
                "options": generation_kwargs,
            },
        ).json()

        return {
            "replies": json_response["response"]
        }