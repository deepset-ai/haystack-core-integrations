from __future__ import annotations
from typing import Any, Dict, List, Optional
import requests
from haystack import component
from haystack.dataclasses import Document
from haystack.utils.auth import Secret


class _IsaacusClient:
    def __init__(self, api_key: str, base_url: str = "https://api.isaacus.com/v1", timeout: int = 30):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def embeddings_create(
        self,
        *,
        model: str,
        texts: List[str],
        task: Optional[str] = None,
        dimensions: Optional[int] = None,
        overflow_strategy: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        payload: Dict[str, Any] = {"model": model, "texts": texts}
        if task:
            payload["task"] = task
        if dimensions is not None:
            payload["dimensions"] = int(dimensions)
        if overflow_strategy:
            payload["overflow_strategy"] = overflow_strategy

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # Expected shape: {"embeddings": [{"embedding": [...]}, ...]}
        items = data.get("embeddings", [])
        return [it["embedding"] for it in items]


@component
class Kanon2TextEmbedder:
    """Embeds a text string into a vector using Isaacus Kanon 2.

    Returns a single vector under the key `embedding`.
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("ISAACUS_API_KEY"),
        base_url: str = "https://api.isaacus.com/v1",
        model: str = "kanon-2-embedder",
        task: str = "retrieval/query",
        dimensions: Optional[int] = None,
        overflow_strategy: Optional[str] = "drop_end",
        timeout: int = 30,
    ):
        self._client = _IsaacusClient(api_key.resolve_value(), base_url, timeout)
        self.model = model
        self.task = task
        self.dimensions = dimensions
        self.overflow_strategy = overflow_strategy

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        if not text or not text.strip():
            return {"embedding": []}
        vectors = self._client.embeddings_create(
            model=self.model,
            texts=[text],
            task=self.task,
            dimensions=self.dimensions,
            overflow_strategy=self.overflow_strategy,
        )
        return {"embedding": vectors[0]}


@component
class Kanon2DocumentEmbedder:
    """Embeds a list of Haystack `Document`s using Isaacus Kanon 2.

    Writes vectors to `document.embedding` and returns the list under `documents`.
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("ISAACUS_API_KEY"),
        base_url: str = "https://api.isaacus.com/v1",
        model: str = "kanon-2-embedder",
        task: str = "retrieval/document",
        dimensions: Optional[int] = None,
        overflow_strategy: Optional[str] = "drop_end",
        batch_size: int = 128,
        timeout: int = 30,
    ):
        self._client = _IsaacusClient(api_key.resolve_value(), base_url, timeout)
        self.model = model
        self.task = task
        self.dimensions = dimensions
        self.overflow_strategy = overflow_strategy
        self.batch_size = max(1, min(128, batch_size))

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        if not documents:
            return {"documents": []}
        docs = [d for d in documents if (d.content or "").strip()]
        for i in range(0, len(docs), self.batch_size):
            batch = docs[i : i + self.batch_size]
            vectors = self._client.embeddings_create(
                model=self.model,
                texts=[d.content for d in batch],
                task=self.task,
                dimensions=self.dimensions,
                overflow_strategy=self.overflow_strategy,
            )
            for d, v in zip(batch, vectors):
                d.embedding = v
        return {"documents": documents}
