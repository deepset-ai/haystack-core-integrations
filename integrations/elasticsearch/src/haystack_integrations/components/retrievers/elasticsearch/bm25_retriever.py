# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy
from haystack_integrations.document_stores.elasticsearch.document_store import ElasticsearchDocumentStore


@component
class ElasticsearchBM25Retriever:
    """
    ElasticsearchBM25Retriever retrieves documents from the ElasticsearchDocumentStore using BM25 algorithm to find the
    most similar documents to a user's query.

    This retriever is only compatible with ElasticsearchDocumentStore.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
    from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever

    document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")
    retriever = ElasticsearchBM25Retriever(document_store=document_store)

    # Add documents to DocumentStore
    documents = [
        Document(text="My name is Carla and I live in Berlin"),
        Document(text="My name is Paul and I live in New York"),
        Document(text="My name is Silvano and I live in Matera"),
        Document(text="My name is Usagi Tsukino and I live in Tokyo"),
    ]
    document_store.write_documents(documents)

    result = retriever.run(query="Who lives in Berlin?")
    for doc in result["documents"]:
        print(doc.content)
    ```
    """

    def __init__(
        self,
        *,
        document_store: ElasticsearchDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        fuzziness: str = "AUTO",
        top_k: int = 10,
        scale_score: bool = False,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        """
        Initialize ElasticsearchBM25Retriever with an instance ElasticsearchDocumentStore.

        :param document_store: An instance of ElasticsearchDocumentStore.
        :param filters: Filters applied to the retrieved Documents, for more info
                        see `ElasticsearchDocumentStore.filter_documents`.
        :param fuzziness: Fuzziness parameter passed to Elasticsearch. See the official
            [documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/common-options.html#fuzziness)
            for more details.
        :param top_k: Maximum number of Documents to return.
        :param scale_score: If `True` scales the Document`s scores between 0 and 1.
        :param filter_policy: Policy to determine how filters are applied.
        :raises ValueError: If `document_store` is not an instance of `ElasticsearchDocumentStore`.
        """

        if not isinstance(document_store, ElasticsearchDocumentStore):
            msg = "document_store must be an instance of ElasticsearchDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters or {}
        self._fuzziness = fuzziness
        self._top_k = top_k
        self._scale_score = scale_score
        self._filter_policy = FilterPolicy.from_str(filter_policy) if isinstance(filter_policy, str) else filter_policy

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self._filters,
            fuzziness=self._fuzziness,
            top_k=self._top_k,
            scale_score=self._scale_score,
            filter_policy=self._filter_policy.value,
            document_store=self._document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElasticsearchBM25Retriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        data["init_parameters"]["document_store"] = ElasticsearchDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        # Pipelines serialized with old versions of the component might not
        # have the filter_policy field.
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None):
        """
        Retrieve documents using the BM25 keyword-based algorithm.

        :param query: String to search in `Document`s' text.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k: Maximum number of `Document` to return.
        :returns: A dictionary with the following keys:
            - `documents`: List of `Document`s that match the query.
        """
        filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        docs = self._document_store._bm25_retrieval(
            query=query,
            filters=filters,
            fuzziness=self._fuzziness,
            top_k=top_k or self._top_k,
            scale_score=self._scale_score,
        )
        return {"documents": docs}
