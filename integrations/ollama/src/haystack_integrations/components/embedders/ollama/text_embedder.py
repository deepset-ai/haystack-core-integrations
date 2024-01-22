from typing import Any, Dict, List, Optional

import requests
from haystack import component


@component
class OllamaTextEmbedder:
    def __init__(
        self,
        model: str = "orca-mini",
        url: str = "http://localhost:11434/api/embeddings",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
    ):
        """
        :param model: The name of the model to use. The model should be available in the running Ollama instance.
            Default is "orca-mini".
        :param url: The URL of the chat endpoint of a running Ollama instance.
            Default is "http://localhost:11434/api/embeddings".
        :param generation_kwargs: Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, and others. See the available arguments in
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param timeout: The number of seconds before throwing a timeout error from the Ollama API.
            Default is 120 seconds.
        """
        self.timeout = timeout
        self.generation_kwargs = generation_kwargs or {}
        self.url = url
        self.model = model

    def _create_json_payload(self, text: str, generation_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Returns A dictionary of JSON arguments for a POST request to an Ollama service
          :param text: Text that is to be converted to an embedding
          :param generation_kwargs:
          :return: A dictionary of arguments for a POST request to an Ollama service
        """
        return {"model": self.model, "prompt": text, "options": {**self.generation_kwargs, **(generation_kwargs or {})}}

    @component.output_types(embedding=List[float])
    def run(self, text: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Run an Ollama Model on a given chat history.
        :param text: Text to be converted to an embedding.
        :param generation_kwargs: Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, etc. See the
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :return: A dictionary with the key "embedding" and a list of floats as the value
        """

        payload = self._create_json_payload(text, generation_kwargs)

        response = requests.post(url=self.url, json=payload, timeout=self.timeout)

        response.raise_for_status()

        return response.json()