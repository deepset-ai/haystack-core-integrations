# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Literal

from haystack import component, default_from_dict, default_to_dict, logging

from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

logger = logging.getLogger(__name__)


@component
class OpenSearchMetadataRetriever:
    """
    Retrieves and ranks metadata from documents stored in an OpenSearchDocumentStore.

    It searches specified metadata fields for matches to a given query, ranks the results based on relevance using
    Jaccard similarity, and returns the top-k results containing only the specified metadata fields. Additionally, it
    adds a boost to the score of exact matches.

    The search is designed for metadata fields whose values are **text** (strings). It uses prefix, wildcard and fuzzy
    matching to find candidate documents; these query types operate only on text/keyword fields in OpenSearch.

    Metadata fields with **non-string types** (integers, floats, booleans, lists of non-strings) are indexed by
    OpenSearch as numeric, boolean, or array types. Those field types do not support prefix, wildcard, or full-text
    match queries, so documents are typically not found when you search only by such fields.

    **Mixed types** in the same metadata field (e.g. a list containing both strings and numbers) are not supported.

    Must be connected to the OpenSearchDocumentStore to run.

    Example:
        ```python
        from haystack import Document
        from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
        from haystack_integrations.components.retrievers.opensearch import OpenSearchMetadataRetriever

        # Create documents with metadata
        docs = [
            Document(
                content="Python programming guide",
                meta={"category": "Python", "status": "active", "priority": 1, "author": "John Doe"}
            ),
            Document(
                content="Java tutorial",
                meta={"category": "Java", "status": "active", "priority": 2, "author": "Jane Smith"}
            ),
            Document(
                content="Python advanced topics",
                meta={"category": "Python", "status": "inactive", "priority": 3, "author": "John Doe"}
            ),
        ]
        document_store.write_documents(docs, refresh=True)

        # Create retriever specifying which metadata fields to search and return
        retriever = OpenSearchMetadataRetriever(
            document_store=document_store,
            metadata_fields=["category", "status", "priority"],
            top_k=10,
        )

        # Search for metadata
        result = retriever.run(query="Python")

        # Result structure:
        # {
        #     "metadata": [
        #         {"category": "Python", "status": "active", "priority": 1},
        #         {"category": "Python", "status": "inactive", "priority": 3},
        #     ]
        # }
        #
        # Note: Only the specified metadata_fields are returned in the results.
        # Other metadata fields (like "author") and document content are excluded.
        ```
    """

    def __init__(
        self,
        *,
        document_store: OpenSearchDocumentStore,
        metadata_fields: list[str],
        top_k: int = 20,
        exact_match_weight: float = 0.6,
        mode: Literal["strict", "fuzzy"] = "fuzzy",
        fuzziness: int | Literal["AUTO"] = 2,
        prefix_length: int = 0,
        max_expansions: int = 200,
        tie_breaker: float = 0.7,
        jaccard_n: int = 3,
        raise_on_failure: bool = True,
    ):
        """
        Create the OpenSearchMetadataRetriever component.

        :param document_store: An instance of OpenSearchDocumentStore to use with the Retriever.
        :param metadata_fields: List of metadata field names to search within each document's metadata.
        :param top_k: Maximum number of top results to return based on relevance. Default is 20.
        :param exact_match_weight: Weight to boost the score of exact matches in metadata fields.
            Default is 0.6. It's used on both "strict" and "fuzzy" modes and applied after the search executes.
        :param mode: Search mode. "strict" uses prefix and wildcard matching,
            "fuzzy" uses fuzzy matching with dis_max queries. Default is "fuzzy".
            In both modes, results are scored using Jaccard similarity (n-gram based)
            computed server-side via a Painless script; n is controlled by jaccard_n.
        :param fuzziness: Maximum allowed Damerau-Levenshtein distance (edit distance) for fuzzy matching.
            Accepts an integer (e.g., 0, 1, 2) or "AUTO" which chooses based on term length.
            Default is 2. Only applies when mode is "fuzzy".
        :param prefix_length: Number of leading characters that must match exactly before fuzzy matching applies.
            Default is 0 (no prefix requirement). Only applies when mode is "fuzzy".
        :param max_expansions: Maximum number of term variations the fuzzy query can generate.
            Default is 200. Only applies when mode is "fuzzy".
        :param tie_breaker: Weight (0..1) for other matching clauses in the dis_max query.
            Boosts documents that match multiple clauses. Default is 0.7. Only applies when mode is "fuzzy".
        :param jaccard_n: N-gram size for Jaccard similarity scoring. Default 3; larger n favors longer token matches.
        :param raise_on_failure:
            If `True`, raises an exception if the API call fails.
            If `False`, logs a warning and returns an empty list.

        :raises ValueError: If `document_store` is not an instance of OpenSearchDocumentStore.
        """
        if not isinstance(document_store, OpenSearchDocumentStore):
            msg = "document_store must be an instance of OpenSearchDocumentStore"
            raise ValueError(msg)

        if not metadata_fields:
            msg = "fields must be a non-empty list of metadata field names"
            raise ValueError(msg)

        self._document_store = document_store
        self._metadata_fields = metadata_fields
        self._top_k = top_k
        self._exact_match_weight = exact_match_weight
        self._mode = mode
        self._fuzziness = fuzziness
        self._prefix_length = prefix_length
        self._max_expansions = max_expansions
        self._tie_breaker = tie_breaker
        self._jaccard_n = jaccard_n
        self._raise_on_failure = raise_on_failure

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self._document_store.to_dict(),
            metadata_fields=self._metadata_fields,
            top_k=self._top_k,
            exact_match_weight=self._exact_match_weight,
            mode=self._mode,
            fuzziness=self._fuzziness,
            prefix_length=self._prefix_length,
            max_expansions=self._max_expansions,
            tie_breaker=self._tie_breaker,
            jaccard_n=self._jaccard_n,
            raise_on_failure=self._raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenSearchMetadataRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.

        :returns:
            Deserialized component.
        """
        data["init_parameters"]["document_store"] = OpenSearchDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(metadata=list[dict[str, Any]])
    def run(
        self,
        query: str,
        *,
        document_store: OpenSearchDocumentStore | None = None,
        metadata_fields: list[str] | None = None,
        top_k: int | None = None,
        exact_match_weight: float | None = None,
        mode: Literal["strict", "fuzzy"] | None = None,
        fuzziness: int | Literal["AUTO"] | None = None,
        prefix_length: int | None = None,
        max_expansions: int | None = None,
        tie_breaker: float | None = None,
        jaccard_n: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Execute a search query against the metadata fields of documents stored in the Document Store.

        :param query: The search query string, which can contain multiple comma-separated parts.
            Each part will be searched across all specified fields.
        :param document_store: The Document Store to run the query against.
            If not provided, the one provided in `__init__` is used.
        :param metadata_fields: List of metadata field names to search within.
            If not provided, the fields provided in `__init__` are used.
        :param top_k: Maximum number of top results to return based on relevance.
            The search retrieves up to 1000 hits from OpenSearch, then applies boosting and filters
            the results to the top_k most relevant matches.
            If not provided, the top_k provided in `__init__` is used.
        :param exact_match_weight: Weight to boost the score of exact matches in metadata fields.
            If not provided, the exact_match_weight provided in `__init__` is used.
        :param mode: Search mode. "strict" uses prefix and wildcard matching,
            "fuzzy" uses fuzzy matching with dis_max queries.
            In both modes, results are scored using Jaccard similarity (n-gram based) via a Painless script.
            If not provided, the mode provided in `__init__` is used.
        :param fuzziness: Maximum allowed Damerau-Levenshtein distance (edit distance) for fuzzy matching.
            Accepts an integer (e.g., 0, 1, 2) or "AUTO" which chooses based on term length.
            Only applies when mode is "fuzzy". If not provided, the fuzziness provided in `__init__` is used.
        :param prefix_length: Number of leading characters that must match exactly before fuzzy matching applies.
            Only applies when mode is "fuzzy". If not provided, the prefix_length provided in `__init__` is used.
        :param max_expansions: Maximum number of term variations the fuzzy query can generate.
            Only applies when mode is "fuzzy". If not provided, the max_expansions provided in `__init__` is used.
        :param tie_breaker: Weight (0..1) for other matching clauses; boosts docs matching multiple
            clauses. Only applies when mode is "fuzzy". If not provided, the tie_breaker provided in `__init__` is used.
        :param jaccard_n: N-gram size for Jaccard similarity scoring. If not provided, the jaccard_n from `__init__`
            is used.
        :param filters: Additional filters to apply to the search query.

        :returns:
            A dictionary containing the top-k retrieved metadata results.


        Example:
            ```python
            from haystack import Document

            # First, add a document with matching metadata to the store
            store.write_documents([
                Document(
                    content="Python programming guide",
                    meta={"category": "Python", "status": "active", "priority": 1}
                )
            ])

            retriever = OpenSearchMetadataRetriever(
                document_store=store,
                metadata_fields=["category", "status", "priority"]
            )
            result = retriever.run(query="Python, active")
            # Returns: {"metadata": [{"category": "Python", "status": "active", "priority": 1}]}
            ```
        """
        doc_store = document_store or self._document_store
        if not isinstance(doc_store, OpenSearchDocumentStore):
            msg = "document_store must be an instance of OpenSearchDocumentStore"
            raise ValueError(msg)

        fields_to_use = metadata_fields if metadata_fields is not None else self._metadata_fields
        top_k_to_use = top_k if top_k is not None else self._top_k
        exact_match_weight_to_use = exact_match_weight if exact_match_weight is not None else self._exact_match_weight
        mode_to_use = mode if mode is not None else self._mode
        fuzziness_to_use = fuzziness if fuzziness is not None else self._fuzziness
        prefix_length_to_use = prefix_length if prefix_length is not None else self._prefix_length
        max_expansions_to_use = max_expansions if max_expansions is not None else self._max_expansions
        tie_breaker_to_use = tie_breaker if tie_breaker is not None else self._tie_breaker
        jaccard_n_to_use = jaccard_n if jaccard_n is not None else self._jaccard_n

        if mode_to_use not in ["strict", "fuzzy"]:
            msg = "mode must be either 'strict' or 'fuzzy'"
            raise ValueError(msg)

        try:
            result = doc_store._metadata_search(
                query=query,
                fields=fields_to_use,
                mode=mode_to_use,
                top_k=top_k_to_use,
                exact_match_weight=exact_match_weight_to_use,
                fuzziness=fuzziness_to_use,
                prefix_length=prefix_length_to_use,
                max_expansions=max_expansions_to_use,
                tie_breaker=tie_breaker_to_use,
                jaccard_n=jaccard_n_to_use,
                filters=filters,
            )
            return {"metadata": result}
        except Exception as e:
            if self._raise_on_failure:
                raise
            logger.warning(
                "Metadata search failed and will be ignored by returning empty results: {error}",
                error=str(e),
                exc_info=True,
            )
            return {"metadata": []}

    @component.output_types(metadata=list[dict[str, Any]])
    async def run_async(
        self,
        query: str,
        *,
        document_store: OpenSearchDocumentStore | None = None,
        metadata_fields: list[str] | None = None,
        top_k: int | None = None,
        exact_match_weight: float | None = None,
        mode: Literal["strict", "fuzzy"] | None = None,
        fuzziness: int | Literal["AUTO"] | None = None,
        prefix_length: int | None = None,
        max_expansions: int | None = None,
        tie_breaker: float | None = None,
        jaccard_n: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Asynchronously execute a search query against the metadata fields of documents stored in the Document Store.

        :param query: The search query string, which can contain multiple comma-separated parts.
            Each part will be searched across all specified fields.
        :param document_store: The Document Store to run the query against.
            If not provided, the one provided in `__init__` is used.
        :param metadata_fields: List of metadata field names to search within.
            If not provided, the fields provided in `__init__` are used.
        :param top_k: Maximum number of top results to return based on relevance.
            The search retrieves up to 1000 hits from OpenSearch, then applies boosting and filters
            the results to the top_k most relevant matches.
            If not provided, the top_k provided in `__init__` is used.
        :param exact_match_weight: Weight to boost the score of exact matches in metadata fields.
            If not provided, the exact_match_weight provided in `__init__` is used.
        :param mode: Search mode. "strict" uses prefix and wildcard matching,
            "fuzzy" uses fuzzy matching with dis_max queries.
            In both modes, results are scored using Jaccard similarity (n-gram based) via a Painless script.
            If not provided, the mode provided in `__init__` is used.
        :param fuzziness: Maximum allowed Damerau-Levenshtein distance (edit distance) for fuzzy matching.
            Accepts an integer (e.g., 0, 1, 2) or "AUTO" which chooses based on term length.
            Only applies when mode is "fuzzy". If not provided, the fuzziness provided in `__init__` is used.
        :param prefix_length: Number of leading characters that must match exactly before fuzzy matching applies.
            Only applies when mode is "fuzzy". If not provided, the prefix_length provided in `__init__` is used.
        :param max_expansions: Maximum number of term variations the fuzzy query can generate.
            Only applies when mode is "fuzzy". If not provided, the max_expansions provided in `__init__` is used.
        :param tie_breaker: Weight (0..1) for other matching clauses; boosts docs matching multiple clauses.
            Only applies when mode is "fuzzy". If not provided, the tie_breaker provided in `__init__` is used.
        :param jaccard_n: N-gram size for Jaccard similarity scoring. If not provided, the jaccard_n from `__init__`
            is used.
        :param filters: Additional filters to apply to the search query.
        :returns: A dictionary containing the top-k retrieved metadata results.

        Example:
            ```python
            from haystack import Document

            # First, add a document with matching metadata to the store
            await store.write_documents_async([
                Document(
                    content="Python programming guide",
                    meta={"category": "Python", "status": "active", "priority": 1}
                )
            ])

            retriever = OpenSearchMetadataRetriever(
                document_store=store,
                metadata_fields=["category", "status", "priority"]
            )
            result = await retriever.run_async(query="Python, active")
            # Returns: {"metadata": [{"category": "Python", "status": "active", "priority": 1}]}
            ```
        """
        doc_store = document_store or self._document_store
        if not isinstance(doc_store, OpenSearchDocumentStore):
            msg = "document_store must be an instance of OpenSearchDocumentStore"
            raise ValueError(msg)

        fields_to_use = metadata_fields if metadata_fields is not None else self._metadata_fields
        top_k_to_use = top_k if top_k is not None else self._top_k
        exact_match_weight_to_use = exact_match_weight if exact_match_weight is not None else self._exact_match_weight
        mode_to_use = mode if mode is not None else self._mode
        fuzziness_to_use = fuzziness if fuzziness is not None else self._fuzziness
        prefix_length_to_use = prefix_length if prefix_length is not None else self._prefix_length
        max_expansions_to_use = max_expansions if max_expansions is not None else self._max_expansions
        tie_breaker_to_use = tie_breaker if tie_breaker is not None else self._tie_breaker
        jaccard_n_to_use = jaccard_n if jaccard_n is not None else self._jaccard_n

        if mode_to_use not in ["strict", "fuzzy"]:
            msg = "mode must be either 'strict' or 'fuzzy'"
            raise ValueError(msg)

        try:
            result = await doc_store._metadata_search_async(
                query=query,
                fields=fields_to_use,
                mode=mode_to_use,
                top_k=top_k_to_use,
                exact_match_weight=exact_match_weight_to_use,
                fuzziness=fuzziness_to_use,
                prefix_length=prefix_length_to_use,
                max_expansions=max_expansions_to_use,
                tie_breaker=tie_breaker_to_use,
                jaccard_n=jaccard_n_to_use,
                filters=filters,
            )
            return {"metadata": result}
        except Exception as e:
            if self._raise_on_failure:
                raise
            logger.warning(
                "Metadata search failed and will be ignored by returning empty results: {error}",
                error=str(e),
                exc_info=True,
            )
            return {"metadata": []}
