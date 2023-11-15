# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack.preview import component, default_from_dict, default_to_dict
from haystack.preview.dataclasses import Document

from elasticsearch_haystack.document_store import ElasticsearchDocumentStore


@component
class ElasticsearchEmbeddingRetriever:
    def __init__(
        self,
        *,
        document_store: ElasticsearchDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        num_candidates: Optional[int] = None,
    ):
        if not isinstance(document_store, ElasticsearchDocumentStore):
            msg = "document_store must be an instance of ElasticsearchDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k
        self._num_candidates = num_candidates

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            filters=self._filters,
            top_k=self._top_k,
            num_candidates=self._num_candidates,
            document_store=self._document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElasticsearchEmbeddingRetriever":
        data["init_parameters"]["document_store"] = ElasticsearchDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float]):
        docs = self._document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=self._filters,
            top_k=self._top_k,
            num_candidates=self._num_candidates,
        )
        return {"documents": docs}
