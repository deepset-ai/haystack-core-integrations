# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document

from elasticsearch_haystack.document_store import ElasticsearchDocumentStore


@component
class ElasticsearchBM25Retriever:
    def __init__(
        self,
        *,
        document_store: ElasticsearchDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        fuzziness: str = "AUTO",
        top_k: int = 10,
        scale_score: bool = False,
    ):
        if not isinstance(document_store, ElasticsearchDocumentStore):
            msg = "document_store must be an instance of ElasticsearchDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters or {}
        self._fuzziness = fuzziness
        self._top_k = top_k
        self._scale_score = scale_score

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            filters=self._filters,
            fuzziness=self._fuzziness,
            top_k=self._top_k,
            scale_score=self._scale_score,
            document_store=self._document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElasticsearchBM25Retriever":
        data["init_parameters"]["document_store"] = ElasticsearchDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query: str, top_k: Optional[int] = None):
        """
        Retrieve documents using the BM25 keyword-based algorithm.

        :param query: String to search in Documents' text.
        :param top_k: Maximum number of Documents to return.
        :return: List of Documents that match the query.
        """
        docs = self._document_store._bm25_retrieval(
            query=query,
            filters=self._filters,
            fuzziness=self._fuzziness,
            top_k=top_k or self._top_k,
            scale_score=self._scale_score,
        )
        return {"documents": docs}
