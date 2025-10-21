from __future__ import annotations
from typing import List, Optional
from haystack import component
from haystack.dataclasses import Document
from haystack.utils import Secret
from .utils import IsaacusClient


@component
class IsaacusDocumentEmbedder:
    """
    Embeds a list of Haystack `Document`s using Isaacus (configurable model).
    Writes vectors to `document.embedding` and returns the list under `documents`.

    Parameters mirror IsaacusTextEmbedder, with an additional `batch_size`.
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
        self._client = IsaacusClient(api_key.resolve_value(), base_url, timeout)
        self.model = model
        self.task = task
        self.dimensions = dimensions
        self.overflow_strategy = overflow_strategy
        self.batch_size = max(1, min(128, batch_size))

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        if not documents:
            return {"documents": []}

        # Only embed non-empty docs
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
