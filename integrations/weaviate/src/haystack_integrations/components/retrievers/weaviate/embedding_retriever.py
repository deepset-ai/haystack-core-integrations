# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore


@component
class WeaviateEmbeddingRetriever:
    """
    A retriever that uses Weaviate's vector search to find similar documents based on the embeddings of the query.
    """

    def __init__(
        self,
        *,
        document_store: WeaviateDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        distance: Optional[float] = None,
        certainty: Optional[float] = None,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        """
        Creates a new instance of WeaviateEmbeddingRetriever.

        :param document_store:
            Instance of WeaviateDocumentStore that will be used from this retriever.
        :param filters:
            Custom filters applied when running the retriever.
        :param top_k:
            Maximum number of documents to return.
        :param distance:
            The maximum allowed distance between Documents' embeddings.
        :param certainty:
            Normalized distance between the result item and the search vector.
        :param filter_policy:
            Policy to determine how filters are applied.
        :raises ValueError:
            If both `distance` and `certainty` are provided.
            See https://weaviate.io/developers/weaviate/api/graphql/search-operators#variables to learn more about
            `distance` and `certainty` parameters.
        """
        if distance is not None and certainty is not None:
            msg = "Can't use 'distance' and 'certainty' parameters together"
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k
        self._distance = distance
        self._certainty = certainty
        self._filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self._filters,
            top_k=self._top_k,
            distance=self._distance,
            certainty=self._certainty,
            filter_policy=self._filter_policy.value,
            document_store=self._document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeaviateEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        data["init_parameters"]["document_store"] = WeaviateDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )

        # Pipelines serialized with old versions of the component might not
        # have the filter_policy field.
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)

        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        distance: Optional[float] = None,
        certainty: Optional[float] = None,
    ):
        """
        Retrieves documents from Weaviate using the vector search.

        :param query_embedding:
            Embedding of the query.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k:
            The maximum number of documents to return.
        :param distance:
            The maximum allowed distance between Documents' embeddings.
        :param certainty:
            Normalized distance between the result item and the search vector.
        :raises ValueError:
            If both `distance` and `certainty` are provided.
            See https://weaviate.io/developers/weaviate/api/graphql/search-operators#variables to learn more about
            `distance` and `certainty` parameters.
        """
        filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        top_k = top_k or self._top_k

        distance = distance or self._distance
        certainty = certainty or self._certainty
        if distance is not None and certainty is not None:
            msg = "Can't use 'distance' and 'certainty' parameters together"
            raise ValueError(msg)

        documents = self._document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            distance=distance,
            certainty=certainty,
        )
        return {"documents": documents}
