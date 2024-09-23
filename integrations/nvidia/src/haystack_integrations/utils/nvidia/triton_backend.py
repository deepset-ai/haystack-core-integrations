from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from haystack import Document
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret

from haystack_integrations.utils.nvidia.utils import REQUEST_TIMEOUT, Model

with LazyImport("Run 'pip install tritonclient[http]'") as tritonclient_http:
    import tritonclient.http

with LazyImport("Run 'pip install tritonclient[grpc]'") as tritonclient_grpc:
    import tritonclient.grpc


class TritonBackend:
    def __init__(
        self,
        model: str,
        api_url: str,
        api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
        model_kwargs: Optional[Dict[str, Any]] = None,
        protocol: Literal["http", "grpc"] = "http",
        timeout: Optional[float] = None,
    ):
        self.headers = {}

        if api_key:
            self.headers["authorization"] = f"Bearer {api_key.resolve_value()}"

        if protocol == "grpc":
            tritonclient_grpc.check()
            self.triton = tritonclient.grpc
        else:
            tritonclient_http.check()
            self.triton = tritonclient.http

        self.client = self.triton.InferenceServerClient(url=api_url)

        self.model = model
        self.api_url = api_url
        self.model_kwargs = model_kwargs or {}

        if timeout is None:
            timeout = REQUEST_TIMEOUT
        self.timeout = int(timeout)

    def embed(self, texts: List[str]) -> Tuple[List[List[float]], Dict[str, Any]]:
        inputs = []
        text_input = self.triton.InferInput("text", [len(texts)], "BYTES")
        text_input.set_data_from_numpy(np.array(texts, dtype=object))
        inputs.append(text_input)

        results = self.client.infer(
            model_name=self.model,
            inputs=inputs,
            headers=self.headers,
            timeout=self.timeout,
        )

        embeddings = results.as_numpy("embeddings").tolist()

        return embeddings, {}

    def generate(self, prompt: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        raise NotImplementedError()

    def models(self) -> List[Model]:
        data = self.client.get_model_repository_index(
            headers=self.headers,
        )

        models = [Model(result["name"]) for result in data]
        if not models:
            msg = f"No hosted model were found at URL '{self.api_url}'."
            raise ValueError(msg)
        return models

    def rank(
        self,
        query: str,
        documents: List[Document],
        endpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError()
