from __future__ import annotations
from typing import Any, Dict, List, Optional
import requests


class IsaacusClient:
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
        items = data.get("embeddings", [])
        return [it["embedding"] for it in items]
