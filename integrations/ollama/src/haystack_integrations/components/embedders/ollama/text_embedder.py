from typing import Any, Dict, List, Optional, Union

from haystack import component

from ollama import AsyncClient, Client


@component
class OllamaTextEmbedder:
    """
    Computes the embeddings of a list of Documents and stores the obtained vectors in the embedding field of
    each Document. It uses embedding models compatible with the Ollama Library.

    Usage example:
    ```python
    from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder

    embedder = OllamaTextEmbedder()
    result = embedder.run(text="What do llamas say once you have thanked them? No probllama!")
    print(result['embedding'])
    ```
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        url: str = "http://localhost:11434",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        keep_alive: Optional[Union[float, str]] = None,
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
        :param timeout:
            The number of seconds before throwing a timeout error from the Ollama API.
        :param keep_alive:
            The option that controls how long the model will stay loaded into memory following the request.
            If not set, it will use the default value from the Ollama (5 minutes).
            The value can be set to:
            - a duration string (such as "10m" or "24h")
            - a number in seconds (such as 3600)
            - any negative number which will keep the model loaded in memory (e.g. -1 or "-1m")
            - '0' which will unload the model immediately after generating a response.
        """
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.generation_kwargs = generation_kwargs or {}
        self.url = url
        self.model = model

        self._client = Client(host=self.url, timeout=self.timeout)
        self._async_client = AsyncClient(host=self.url, timeout=self.timeout)

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(
        self, text: str, generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Union[List[float], Dict[str, Any]]]:
        """
        Runs an Ollama Model to compute embeddings of the provided text.

        :param text:
            Text to be converted to an embedding.
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, etc. See the
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :returns: A dictionary with the following keys:
            - `embedding`: The computed embeddings
            - `meta`: The metadata collected during the embedding process
        """
        result = self._client.embeddings(
            model=self.model,
            prompt=text,
            options=generation_kwargs,
            keep_alive=self.keep_alive,
        ).model_dump()
        result["meta"] = {"model": self.model}

        return result

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    async def run_async(
        self, text: str, generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Union[List[float], Dict[str, Any]]]:
        """
        Asynchronously run an Ollama Model to compute embeddings of the provided text.

        :param text:
            Text to be converted to an embedding.
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, etc. See the
            [Ollama docs](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :returns: A dictionary with the following keys:
            - `embedding`: The computed embeddings
            - `meta`: The metadata collected during the embedding process
        """
        response = await self._async_client.embeddings(
            model=self.model,
            prompt=text,
            options=generation_kwargs,
            keep_alive=self.keep_alive,
        )
        result = response.model_dump()
        result["meta"] = {"model": self.model}

        return result
