# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict, logging

import cognee  # type: ignore[import-untyped]
from cognee.api.v1.search import SearchType  # type: ignore[import-untyped]
from haystack_integrations.components.connectors.cognee._utils import CogneeSearchType, extract_text, run_sync

logger = logging.getLogger(__name__)


@component
class CogneeRetriever:
    """
    Retrieves documents from Cognee's memory.

    Wraps `cognee.search()` and converts results into Haystack `Document` objects.

    Usage:
    ```python
    from haystack_integrations.components.retrievers.cognee import CogneeRetriever

    retriever = CogneeRetriever(search_type="GRAPH_COMPLETION", top_k=5)
    results = retriever.run(query="What is Cognee?")
    for doc in results["documents"]:
        print(doc.content)
    ```
    """

    def __init__(
        self, search_type: CogneeSearchType = "GRAPH_COMPLETION", top_k: int = 10, dataset_name: str | None = None
    ):
        """
        :param search_type: Cognee search type. One of: GRAPH_COMPLETION, CHUNKS,
            SUMMARIES, INSIGHTS, etc.
        :param top_k: Maximum number of results to return.
        :param dataset_name: Optional dataset name to restrict search scope.
        """
        self.search_type = search_type
        self.top_k = top_k
        self.dataset_name = dataset_name

    @component.output_types(documents=list[Document])
    def run(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """
        Search Cognee's memory and return matching documents.

        :param query: The search query.
        :param top_k: Override the default maximum number of results.
        :returns: Dictionary with key `documents` containing the search results
            as Haystack Document objects.
        """
        effective_top_k = top_k if top_k is not None else self.top_k
        search_type_enum = SearchType[self.search_type]

        search_kwargs: dict[str, Any] = {
            "query_text": query,
            "query_type": search_type_enum,
        }
        if self.dataset_name:
            search_kwargs["datasets"] = [self.dataset_name]

        raw_results = run_sync(cognee.search(**search_kwargs))

        documents = _convert_results(raw_results, effective_top_k)

        logger.info(
            "Cognee search returned {count} documents for query '{query}'",
            count=len(documents),
            query=query[:80],
        )
        return {"documents": documents}

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            search_type=self.search_type,
            top_k=self.top_k,
            dataset_name=self.dataset_name,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CogneeRetriever":
        return default_from_dict(cls, data)


def _convert_results(raw_results: list[Any], top_k: int) -> list[Document]:
    """Convert Cognee search results to Haystack Documents."""
    documents: list[Document] = []
    if not raw_results:
        return documents

    for item in raw_results[:top_k]:
        text = extract_text(item)
        if text:
            documents.append(
                Document(
                    content=text,
                    meta={"source": "cognee", "search_result_type": type(item).__name__},
                )
            )
    return documents
