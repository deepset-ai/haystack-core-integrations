from __future__ import annotations
from typing import List, Optional
from haystack import component
from haystack.utils import Secret
from .utils import IsaacusClient


@component
class IsaacusTextEmbedder:
    """
    Embeds a text string into a vector using Isaacus (configurable model).
    Returns a single vector under the key `embedding`.

    Parameters
    ----------
    api_key : Secret
        Isaacus API key (default reads ISAACUS_API_KEY env var).
    base_url : str
        Isaacus API base URL.
    model : str
        Embedding model name (e.g., "kanon-2-embedder").
    task : str
        Embedding task name ("retrieval/query" by default for queries).
    dimensions : Optional[int]
        Optional output dimensionality (e.g., 1792, 1024, 768...).
    overflow_strategy : Optional[str]
        Truncation strategy for long inputs (e.g., "drop_end").
    timeout : int
        HTTP timeout in seconds.
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
        self._client = IsaacusClient(api_key.resolve_value(), base_url, timeout)
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
