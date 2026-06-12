# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.oracle import OracleDocumentStore

_VALID_SEARCH_MODES = {"keyword", "hybrid", "semantic"}


@component
class OracleHybridRetriever:
    """
    Retrieves documents with DBMS_HYBRID_VECTOR.SEARCH.
    """

    def __init__(
        self,
        *,
        document_store: OracleDocumentStore,
        index_name: str,
        search_mode: Literal["keyword", "hybrid", "semantic"] = "hybrid",
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        params: dict[str, Any] | None = None,
        return_scores: bool = False,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        if not isinstance(document_store, OracleDocumentStore):
            msg = "document_store must be an instance of OracleDocumentStore"
            raise TypeError(msg)
        if search_mode not in _VALID_SEARCH_MODES:
            msg = f"search_mode must be one of {_VALID_SEARCH_MODES}, got {search_mode!r}"
            raise ValueError(msg)

        self.document_store = document_store
        self.index_name = index_name
        self.search_mode = search_mode
        self.filters = filters or {}
        self.top_k = top_k
        self.params = OracleDocumentStore._validate_hybrid_params(params or {})
        self.return_scores = return_scores
        self.filter_policy = FilterPolicy.from_str(filter_policy) if isinstance(filter_policy, str) else filter_policy

    def _merged_params(self, params: dict[str, Any] | None) -> dict[str, Any]:
        merged_params = dict(self.params)
        merged_params.update(OracleDocumentStore._validate_hybrid_params(params or {}))
        return merged_params

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents for a text query.
        """
        merged_filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        documents = self.document_store._hybrid_retrieval(
            query,
            index_name=self.index_name,
            search_mode=self.search_mode,
            filters=merged_filters,
            top_k=top_k if top_k is not None else self.top_k,
            params=self._merged_params(params),
            return_scores=self.return_scores,
        )
        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously retrieve documents for a text query.
        """
        merged_filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        documents = await self.document_store._hybrid_retrieval_async(
            query,
            index_name=self.index_name,
            search_mode=self.search_mode,
            filters=merged_filters,
            top_k=top_k if top_k is not None else self.top_k,
            params=self._merged_params(params),
            return_scores=self.return_scores,
        )
        return {"documents": documents}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.
        """
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            index_name=self.index_name,
            search_mode=self.search_mode,
            filters=self.filters,
            top_k=self.top_k,
            params=self.params,
            return_scores=self.return_scores,
            filter_policy=self.filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleHybridRetriever":
        """
        Deserializes the component from a dictionary.
        """
        params = data.get("init_parameters", {})
        if "document_store" in params:
            params["document_store"] = OracleDocumentStore.from_dict(params["document_store"])
        if filter_policy := params.get("filter_policy"):
            params["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)
