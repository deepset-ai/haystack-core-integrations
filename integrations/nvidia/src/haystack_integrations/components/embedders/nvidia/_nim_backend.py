from typing import Any, Dict, List, Optional, Tuple

import requests
from haystack.utils.auth import Secret

from .backend import EmbedderBackend


class NimBackend(EmbedderBackend):
    def __init__(
        self,
        model: str,
        api_url: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        api_key: Optional[Secret] = None,
    ):
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key.resolve_value()}"
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
        )
        res.raise_for_status()

        data = res.json()
        # Sort the embeddings by index, we don't know whether they're out of order or not
        embeddings = [e["embedding"] for e in sorted(data["data"], key=lambda e: e["index"])]

        return embeddings, {"usage": data["usage"]}
