# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document

from opensearch_haystack.document_store import OpenSearchDocumentStore


@component
class OpenSearchBM25Retriever:
    def __init__(
        self,
        *,
        document_store: OpenSearchDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        fuzziness: str = "AUTO",
        top_k: int = 10,
        scale_score: bool = False,
        all_terms_must_match: bool = False,
    ):
        """
        Create the OpenSearchBM25Retriever component.

        :param document_store: An instance of OpenSearchDocumentStore.
        :param filters: Filters applied to the retrieved Documents. Defaults to None.
        :param fuzziness: Fuzziness parameter for full-text queries. Defaults to "AUTO".
        :param top_k: Maximum number of Documents to return, defaults to 10
        :param scale_score: Whether to scale the score of retrieved documents between 0 and 1.
            This is useful when comparing documents across different indexes. Defaults to False.
        :param all_terms_must_match: If True, all terms in the query string must be present in the retrieved documents.
            This is useful when searching for short text where even one term can make a difference. Defaults to False.
        :raises ValueError: If `document_store` is not an instance of OpenSearchDocumentStore.

        """
        if not isinstance(document_store, OpenSearchDocumentStore):
            msg = "document_store must be an instance of OpenSearchDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters or {}
        self._fuzziness = fuzziness
        self._top_k = top_k
        self._scale_score = scale_score
        self._all_terms_must_match = all_terms_must_match

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
    def from_dict(cls, data: Dict[str, Any]) -> "OpenSearchBM25Retriever":
        data["init_parameters"]["document_store"] = OpenSearchDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        all_terms_must_match: Optional[bool] = None,
        top_k: Optional[int] = None,
        fuzziness: Optional[str] = None,
        scale_score: Optional[bool] = None,
    ):
        """
        Retrieve documents using BM25 retrieval.

        :param query: The query string
        :param filters: Optional filters to narrow down the search space.
        :param all_terms_must_match: If True, all terms in the query string must be present in the retrieved documents.
        :param top_k: Maximum number of Documents to return.
        :param fuzziness: Fuzziness parameter for full-text queries.
        :param scale_score: Whether to scale the score of retrieved documents between 0 and 1.
            This is useful when comparing documents across different indexes.
        :return: A dictionary containing the retrieved documents.
        """
        if filters is None:
            filters = self._filters
        if all_terms_must_match is None:
            all_terms_must_match = self._all_terms_must_match
        if top_k is None:
            top_k = self._top_k
        if fuzziness is None:
            fuzziness = self._fuzziness
        if scale_score is None:
            scale_score = self._scale_score

        docs = self._document_store._bm25_retrieval(
            query=query,
            filters=filters,
            fuzziness=fuzziness,
            top_k=top_k,
            scale_score=scale_score,
            all_terms_must_match=all_terms_must_match,
        )
        return {"documents": docs}
