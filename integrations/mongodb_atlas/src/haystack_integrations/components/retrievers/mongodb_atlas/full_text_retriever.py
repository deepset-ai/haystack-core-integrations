# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Literal, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


@component
class MongoDBAtlasFullTextRetriever:
    """
    Retrieves documents from the MongoDBAtlasDocumentStore by full-text search.

    The full-text search is dependent on the full_text_search_index used in the MongoDBAtlasDocumentStore.
    See MongoDBAtlasDocumentStore for more information.

    Usage example:
    ```python
    from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
    from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasFullTextRetriever

    store = MongoDBAtlasDocumentStore(database_name="your_existing_db",
                                      collection_name="your_existing_collection",
                                      vector_search_index="your_existing_index",
                                      full_text_search_index="your_existing_index")
    retriever = MongoDBAtlasFullTextRetriever(document_store=store)

    results = retriever.run(query="Lorem ipsum")
    print(results["documents"])
    ```

    The example above retrieves the 10 most similar documents to the query "Lorem ipsum" from the
    MongoDBAtlasDocumentStore.
    """

    def __init__(
        self,
        *,
        document_store: MongoDBAtlasDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        """
        :param document_store: An instance of MongoDBAtlasDocumentStore.
        :param filters: Filters applied to the retrieved Documents. Make sure that the fields used in the filters are
            included in the configuration of the `full_text_search_index`. The configuration must be done manually
            in the Web UI of MongoDB Atlas.
        :param top_k: Maximum number of Documents to return.
        :param filter_policy: Policy to determine how filters are applied.

        :raises ValueError: If `document_store` is not an instance of MongoDBAtlasDocumentStore.
        """

        if not isinstance(document_store, MongoDBAtlasDocumentStore):
            msg = "document_store must be an instance of MongoDBAtlasDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = (
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
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MongoDBAtlasFullTextRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
              Deserialized component.
        """
        data["init_parameters"]["document_store"] = MongoDBAtlasDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )

        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: Union[str, List[str]],
        fuzzy: Optional[Dict[str, int]] = None,
        match_criteria: Optional[Literal["any", "all"]] = None,
        score: Optional[Dict[str, Dict]] = None,
        synonyms: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents from the MongoDBAtlasDocumentStore by full-text search.

        :param query: The query string or a list of query strings to search for.
            If the query contains multiple terms, Atlas Search evaluates each term separately for matches.
        :param fuzzy: Enables finding strings similar to the search term(s).
            Note, `fuzzy` cannot be used with `synonyms`. Configurable options include `maxEdits`, `prefixLength`,
            and `maxExpansions`. For more details refer to MongoDB Atlas
            [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
        :param match_criteria: Defines how terms in the query are matched. Supported options are `"any"` and `"all"`.
            For more details refer to MongoDB Atlas
            [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
        :param score: Specifies the scoring method for matching results. Supported options include `boost`, `constant`,
            and `function`. For more details refer to MongoDB Atlas
            [documentation](https://www.mongodb.com/docs/atlas/atlas-search/text/#fields).
        :param synonyms: The name of the synonym mapping definition in the index. This value cannot be an empty string.
            Note, `synonyms` can not be used with `fuzzy`.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k: Maximum number of Documents to return. Overrides the value specified at initialization.
        :returns: A dictionary with the following keys:
            - `documents`: List of Documents most similar to the given `query`
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = top_k or self.top_k

        docs = self.document_store._fulltext_retrieval(
            query=query,
            fuzzy=fuzzy,
            match_criteria=match_criteria,
            score=score,
            synonyms=synonyms,
            filters=filters,
            top_k=top_k,
        )

        return {"documents": docs}
