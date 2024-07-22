from typing import Any, Dict, List, Optional, Tuple

import requests
from haystack.utils import Secret
from typing import Literal, Optional

from .backend import EmbedderBackend

REQUEST_TIMEOUT = 60

class Model:
    """
    Model information.

    id: unique identifier for the model, passed as model parameter for requests
    model_type: API type (chat, vlm, embedding, ranking, completion)
    client: client name, e.g. ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
    endpoint: custom endpoint for the model
    aliases: list of aliases for the model
    supports_tools: whether the model supports tool calling

    All aliases are deprecated and will trigger a warning when used.
    """
    id: str
    model_type: Optional[
        Literal["embedding"]
    ] = None
    aliases: Optional[list] = None
    base_model: Optional[str] = None

class NimBackend(EmbedderBackend):
    def __init__(
        self,
        model: str,
        api_url: str,
        api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        if api_key:
            headers["authorization"] = f"Bearer {api_key.resolve_value()}"

        self.session = requests.Session()
        self.session.headers.update(headers)

        self.model = model
        self.api_url = api_url
        self.model_kwargs = model_kwargs or {}

    def embed(self, texts: List[str]) -> Tuple[List[List[float]], Dict[str, Any]]:
        url = f"{self.api_url}/embeddings"

        res = self.session.post(
            url,
            json={
                "model": self.model,
                "input": texts,
                **self.model_kwargs,
            },
            timeout=REQUEST_TIMEOUT,
        )
        res.raise_for_status()

        data = res.json()
        # Sort the embeddings by index, we don't know whether they're out of order or not
        embeddings = [e["embedding"] for e in sorted(data["data"], key=lambda e: e["index"])]

        return embeddings, {"usage": data["usage"]}
    
    def models(self) ->List[Model]:
        url = f"{self.api_url}/models"

        res = self.session.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )
        res.raise_for_status()

        data = res.json()["data"]
        models = []
        for element in data:
            assert "id" in element, f"No id found in {element}"
            models.append(Model(id=element["id"]))

        return models
