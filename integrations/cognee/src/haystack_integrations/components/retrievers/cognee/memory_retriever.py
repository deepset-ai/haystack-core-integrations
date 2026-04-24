# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging

from haystack_integrations.memory_stores.cognee import CogneeMemoryStore

logger = logging.getLogger(__name__)


@component
class CogneeRetriever:
    """
    Retrieves memories from a `CogneeMemoryStore` and returns them as Haystack `Document`s.

    Follows the same pattern as other Haystack retrievers (e.g. `OpenSearchBM25Retriever`):
    the store is the source of truth for configuration (`search_type`, `dataset_name`, `top_k`)
    and the retriever delegates the actual search call to `CogneeMemoryStore.search_memories`.

    Usage:
    ```python
    from haystack_integrations.memory_stores.cognee import CogneeMemoryStore
    from haystack_integrations.components.retrievers.cognee import CogneeRetriever

    store = CogneeMemoryStore(search_type="GRAPH_COMPLETION", top_k=5, dataset_name="my_data")
    retriever = CogneeRetriever(memory_store=store)
    results = retriever.run(query="What is Cognee?")
    for doc in results["documents"]:
        print(doc.content)
    ```
    """

    def __init__(
        self,
        *,
        memory_store: CogneeMemoryStore,
        top_k: int | None = None,
    ):
        """
        Initialize the CogneeRetriever.

        :param memory_store: An instance of `CogneeMemoryStore` to retrieve memories from.
        :param top_k: Default maximum number of results to return. When `None`, the
            store's own `top_k` is used. Can be overridden at runtime via `run(top_k=...)`.
        :raises ValueError: If `memory_store` is not an instance of `CogneeMemoryStore`.
        """
        if not isinstance(memory_store, CogneeMemoryStore):
            msg = "memory_store must be an instance of CogneeMemoryStore"
            raise ValueError(msg)

        self._memory_store = memory_store
        self._top_k = top_k

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        top_k: int | None = None,
        user_id: str | None = None,
    ) -> dict[str, list[Document]]:
        """
        Search the attached `CogneeMemoryStore` and return matching memories as Documents.

        The arguments mirror `CogneeMemoryStore.search_memories` so that pipeline builders
        can scope a retrieval call at runtime (e.g. per end-user).

        :param query: The search query.
        :param top_k: Maximum number of results to return. Overrides the retriever's
            init-time default (if set), which in turn overrides the store's default.
        :param user_id: Optional cognee user UUID to scope the search to a specific user.
            When `None`, cognee's default user is used.
        :returns: Dictionary with key `documents` containing the search results as
            Haystack `Document` objects.
        """
        effective_top_k = top_k if top_k is not None else self._top_k
        search_kwargs: dict[str, Any] = {"query": query, "user_id": user_id}
        if effective_top_k is not None:
            search_kwargs["top_k"] = effective_top_k

        memories = self._memory_store.search_memories(**search_kwargs)

        documents = [Document(content=m.text, meta={"source": "cognee"}) for m in memories if m.text]

        logger.info(
            "Cognee retriever returned {count} documents for query '{query}'",
            count=len(documents),
            query=query[:80],
        )
        return {"documents": documents}

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            memory_store=self._memory_store.to_dict(),
            top_k=self._top_k,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CogneeRetriever":
        """Deserialize a component from a dictionary."""
        data["init_parameters"]["memory_store"] = CogneeMemoryStore.from_dict(data["init_parameters"]["memory_store"])
        return default_from_dict(cls, data)
