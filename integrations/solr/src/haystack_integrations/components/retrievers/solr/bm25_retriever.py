# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.solr import SolrDocumentStore

logger = logging.getLogger(__name__)


@component
class SolrBM25Retriever:
    """
    Fetches documents from a `SolrDocumentStore` using Solr's BM25 similarity.

    Usage example:
    ```python
    from haystack_integrations.document_stores.solr import SolrDocumentStore
    from haystack_integrations.components.retrievers.solr import SolrBM25Retriever

    document_store = SolrDocumentStore(core="haystack")
    retriever = SolrBM25Retriever(document_store=document_store)
    result = retriever.run(query="Apache Solr")
    ```
    """

    def __init__(
        self,
        *,
        document_store: SolrDocumentStore,
        filters: dict[str, Any] | None = None,
        fuzziness: int = 0,
        top_k: int = 10,
        scale_score: bool = False,
        all_terms_must_match: bool = False,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
        raise_on_failure: bool = True,
    ) -> None:
        """
        Create a `SolrBM25Retriever`.

        :param document_store: the document store to search.
        :param filters: filters applied to the search. Combined with the filters passed to `run`
            according to `filter_policy`.
        :param fuzziness: per-term edit distance. `0`, the default, disables fuzzy matching.
        :param top_k: maximum number of documents to return.
        :param scale_score: whether to scale scores into the `(0, 1)` range.
        :param all_terms_must_match: whether every query term must match.
        :param filter_policy: how runtime filters combine with the filters given here.
        :param raise_on_failure: whether a failing search raises, or logs and returns no documents.
        :raises ValueError: if `document_store` is not a `SolrDocumentStore`, or `top_k` is not positive.
        """
        if not isinstance(document_store, SolrDocumentStore):
            msg = "document_store must be an instance of SolrDocumentStore"
            raise ValueError(msg)
        self._validate_top_k(top_k)

        self._document_store = document_store
        self._filters = filters or {}
        self._fuzziness = fuzziness
        self._top_k = top_k
        self._scale_score = scale_score
        self._all_terms_must_match = all_terms_must_match
        self._filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )
        self._raise_on_failure = raise_on_failure

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self._document_store.to_dict(),
            filters=self._filters,
            fuzziness=self._fuzziness,
            top_k=self._top_k,
            scale_score=self._scale_score,
            all_terms_must_match=self._all_terms_must_match,
            filter_policy=self._filter_policy.value,
            raise_on_failure=self._raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SolrBM25Retriever":
        """
        Deserializes the component from a dictionary.

        :param data: dictionary to deserialize from.
        :returns: deserialized component.
        """
        init_parameters = data["init_parameters"]
        init_parameters["document_store"] = SolrDocumentStore.from_dict(init_parameters["document_store"])
        # Pipelines serialized before `filter_policy` existed omit the key entirely.
        if filter_policy := init_parameters.get("filter_policy"):
            init_parameters["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)

    @staticmethod
    def _validate_top_k(top_k: int | None) -> None:
        if top_k is not None and top_k <= 0:
            msg = f"top_k must be > 0, but got {top_k}"
            raise ValueError(msg)

    def _search_kwargs(
        self,
        filters: dict[str, Any] | None,
        top_k: int | None,
        fuzziness: int | None,
        scale_score: bool | None,
        all_terms_must_match: bool | None,
    ) -> dict[str, Any]:
        return {
            "filters": apply_filter_policy(self._filter_policy, self._filters, filters),
            "top_k": top_k if top_k is not None else self._top_k,
            "fuzziness": fuzziness if fuzziness is not None else self._fuzziness,
            "scale_score": scale_score if scale_score is not None else self._scale_score,
            "all_terms_must_match": (
                all_terms_must_match if all_terms_must_match is not None else self._all_terms_must_match
            ),
        }

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        fuzziness: int | None = None,
        scale_score: bool | None = None,
        all_terms_must_match: bool | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents matching `query`.

        :param query: the query string.
        :param filters: filters applied to the search.
        :param top_k: maximum number of documents to return.
        :param fuzziness: per-term edit distance.
        :param scale_score: whether to scale scores into the `(0, 1)` range.
        :param all_terms_must_match: whether every query term must match.
        :returns: a dictionary with a `documents` key holding the retrieved documents.
        :raises ValueError: if `top_k` is not positive.
        """
        self._validate_top_k(top_k)
        kwargs = self._search_kwargs(filters, top_k, fuzziness, scale_score, all_terms_must_match)
        try:
            documents = self._document_store._bm25_retrieval(query, **kwargs)
        except Exception as error:
            if self._raise_on_failure:
                raise
            logger.warning("An error occurred during BM25 retrieval and will be ignored: {error}", error=str(error))
            documents = []
        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
        fuzziness: int | None = None,
        scale_score: bool | None = None,
        all_terms_must_match: bool | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents matching `query`, asynchronously.

        :param query: the query string.
        :param filters: filters applied to the search.
        :param top_k: maximum number of documents to return.
        :param fuzziness: per-term edit distance.
        :param scale_score: whether to scale scores into the `(0, 1)` range.
        :param all_terms_must_match: whether every query term must match.
        :returns: a dictionary with a `documents` key holding the retrieved documents.
        :raises ValueError: if `top_k` is not positive.
        """
        self._validate_top_k(top_k)
        kwargs = self._search_kwargs(filters, top_k, fuzziness, scale_score, all_terms_must_match)
        try:
            documents = await self._document_store._bm25_retrieval_async(query, **kwargs)
        except Exception as error:
            if self._raise_on_failure:
                raise
            logger.warning("An error occurred during BM25 retrieval and will be ignored: {error}", error=str(error))
            documents = []
        return {"documents": documents}

    def close(self) -> None:
        """Close the underlying document store connection."""
        self._document_store.close()

    async def close_async(self) -> None:
        """Close the underlying document store async connection."""
        await self._document_store.close_async()
