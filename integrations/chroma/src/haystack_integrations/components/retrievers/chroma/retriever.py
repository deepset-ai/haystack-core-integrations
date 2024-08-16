# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.chroma import ChromaDocumentStore


@component
class ChromaQueryTextRetriever:
    """
    A component for retrieving documents from a [Chroma database](https://docs.trychroma.com/) using the `query` API.

    Example usage:
    ```python
    from haystack import Pipeline
    from haystack.components.converters import TextFileToDocument
    from haystack.components.writers import DocumentWriter

    from haystack_integrations.document_stores.chroma import ChromaDocumentStore
    from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever

    file_paths = ...

    # Chroma is used in-memory so we use the same instances in the two pipelines below
    document_store = ChromaDocumentStore()

    indexing = Pipeline()
    indexing.add_component("converter", TextFileToDocument())
    indexing.add_component("writer", DocumentWriter(document_store))
    indexing.connect("converter", "writer")
    indexing.run({"converter": {"sources": file_paths}})

    querying = Pipeline()
    querying.add_component("retriever", ChromaQueryTextRetriever(document_store))
    results = querying.run({"retriever": {"query": "Variable declarations", "top_k": 3}})

    for d in results["retriever"]["documents"]:
        print(d.meta, d.score)
    ```
    """

    def __init__(
        self,
        document_store: ChromaDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        """
        :param document_store: an instance of `ChromaDocumentStore`.
        :param filters: filters to narrow down the search space.
        :param top_k: the maximum number of documents to retrieve.
        :param filter_policy: Policy to determine how filters are applied.
        """
        self.filters = filters or {}
        self.top_k = top_k
        self.document_store = document_store
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ):
        """
        Run the retriever on the given input data.

        :param query: The input data for the retriever. In this case, a plain-text query.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k: The maximum number of documents to retrieve.
            If not specified, the default value from the constructor is used.
        :returns: A dictionary with the following keys:
            - `documents`: List of documents returned by the search engine.

        :raises ValueError: If the specified document store is not found or is not a MemoryDocumentStore instance.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = top_k or self.top_k
        return {"documents": self.document_store.search([query], top_k, filters)[0]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChromaQueryTextRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        document_store = ChromaDocumentStore.from_dict(data["init_parameters"]["document_store"])
        data["init_parameters"]["document_store"] = document_store
        # Pipelines serialized with old versions of the component might not
        # have the filter_policy field.
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)

        return default_from_dict(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )


@component
class ChromaEmbeddingRetriever(ChromaQueryTextRetriever):
    """
    A component for retrieving documents from a [Chroma database](https://docs.trychroma.com/) using embeddings.
    """

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ):
        """
        Run the retriever on the given input data.

        :param query_embedding: the query embeddings.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k: the maximum number of documents to retrieve.
            If not specified, the default value from the constructor is used.

        :returns: a dictionary with the following keys:
            - `documents`: List of documents returned by the search engine.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)

        top_k = top_k or self.top_k

        query_embeddings = [query_embedding]
        return {"documents": self.document_store.search_embeddings(query_embeddings, top_k, filters)[0]}
