# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from haystack import Document, Pipeline, default_from_dict, default_to_dict, logging, super_component
from haystack.components.embedders.types import TextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.joiners.document_joiner import JoinMode
from haystack.core.serialization import component_to_dict
from haystack.document_stores.types import FilterPolicy
from haystack.utils import deserialize_chatgenerator_inplace

from haystack_integrations.components.retrievers.solr import SolrBM25Retriever, SolrEmbeddingRetriever
from haystack_integrations.document_stores.solr import SolrDocumentStore

logger = logging.getLogger(__name__)

#: Extra per-branch init arguments callers may pass through.
_EXTRA_ARG_KEYS = ("bm25_retriever", "embedding_retriever")


@super_component
class SolrHybridRetriever:
    """
    Hybrid retrieval over a `SolrDocumentStore`, combining BM25 and dense vector search.

    Wraps a pipeline that embeds the query, runs a BM25 and an embedding retriever over the same core,
    and fuses the two result lists with a `DocumentJoiner`.

    Usage example:
    ```python
    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from haystack_integrations.document_stores.solr import SolrDocumentStore
    from haystack_integrations.components.retrievers.solr import SolrHybridRetriever

    document_store = SolrDocumentStore(core="haystack", embedding_dim=384)
    retriever = SolrHybridRetriever(
        document_store=document_store,
        embedder=SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
    )
    retriever.warm_up()
    result = retriever.run(query="Apache Solr")
    ```
    """

    def __init__(
        self,
        document_store: SolrDocumentStore,
        *,
        embedder: TextEmbedder,
        filters_bm25: dict[str, Any] | None = None,
        fuzziness: int = 0,
        top_k_bm25: int = 10,
        scale_score: bool = False,
        all_terms_must_match: bool = False,
        filter_policy_bm25: str | FilterPolicy = FilterPolicy.REPLACE,
        filters_embedding: dict[str, Any] | None = None,
        top_k_embedding: int = 10,
        filter_policy_embedding: str | FilterPolicy = FilterPolicy.REPLACE,
        join_mode: str | JoinMode = JoinMode.RECIPROCAL_RANK_FUSION,
        weights: list[float] | None = None,
        top_k: int | None = None,
        sort_by_score: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Create a `SolrHybridRetriever`.

        :param document_store: the document store both retrievers search.
        :param embedder: the text embedder turning the query into a vector.
        :param filters_bm25: filters for the BM25 branch.
        :param fuzziness: per-term edit distance for the BM25 branch.
        :param top_k_bm25: maximum number of documents from the BM25 branch.
        :param scale_score: whether to scale BM25 scores into the `(0, 1)` range.
        :param all_terms_must_match: whether every query term must match in the BM25 branch.
        :param filter_policy_bm25: filter policy for the BM25 branch.
        :param filters_embedding: filters for the embedding branch.
        :param top_k_embedding: maximum number of documents from the embedding branch.
        :param filter_policy_embedding: filter policy for the embedding branch.
        :param join_mode: how the two result lists are fused.
        :param weights: per-branch weights used by the joiner.
        :param top_k: maximum number of documents returned after fusion.
        :param sort_by_score: whether the fused documents are sorted by score.
        :param kwargs: extra init arguments for the underlying retrievers, given as
            `bm25_retriever={...}` and/or `embedding_retriever={...}`.
        :raises ValueError: if `kwargs` contains a key other than those two.
        """
        self.document_store = document_store
        self.embedder = embedder

        self.filters_bm25 = filters_bm25
        self.fuzziness = fuzziness
        self.top_k_bm25 = top_k_bm25
        self.scale_score = scale_score
        self.all_terms_must_match = all_terms_must_match
        self.filter_policy_bm25 = filter_policy_bm25

        self.filters_embedding = filters_embedding
        self.top_k_embedding = top_k_embedding
        self.filter_policy_embedding = filter_policy_embedding

        self.join_mode = join_mode
        self.weights = weights
        self.top_k = top_k
        self.sort_by_score = sort_by_score

        init_args: dict[str, Any] = {
            "bm25_retriever": {
                "document_store": self.document_store,
                "filters": self.filters_bm25,
                "fuzziness": self.fuzziness,
                "top_k": self.top_k_bm25,
                "scale_score": self.scale_score,
                "all_terms_must_match": self.all_terms_must_match,
                "filter_policy": self.filter_policy_bm25,
            },
            "embedding_retriever": {
                "document_store": self.document_store,
                "filters": self.filters_embedding,
                "top_k": self.top_k_embedding,
                "filter_policy": self.filter_policy_embedding,
            },
            "document_joiner": {
                "join_mode": self.join_mode,
                "weights": self.weights,
                "top_k": self.top_k,
                "sort_by_score": self.sort_by_score,
            },
        }

        for key in kwargs:
            if key not in _EXTRA_ARG_KEYS:
                msg = f"valid extra args are only: {' and '.join(repr(k) for k in _EXTRA_ARG_KEYS)}. Found: {key}"
                raise ValueError(msg)
        self.extra_args = kwargs

        for key in _EXTRA_ARG_KEYS:
            if key in kwargs:
                init_args[key].update(kwargs[key])
                # The document store is shared by both branches and must not be overridden.
                init_args[key]["document_store"] = self.document_store

        self.pipeline = self._create_pipeline(init_args)

    if TYPE_CHECKING:

        def warm_up(self) -> None:
            """Warm up the underlying pipeline components."""
            ...

        def run(
            self,
            query: str,
            filters_bm25: dict[str, Any] | None = None,
            filters_embedding: dict[str, Any] | None = None,
            top_k_bm25: int | None = None,
            top_k_embedding: int | None = None,
        ) -> dict[str, list[Document]]:
            """Run the hybrid retrieval pipeline and return the retrieved documents."""
            ...

    def _create_pipeline(self, data: dict[str, Any]) -> Pipeline:
        """Build the wrapped pipeline and declare how its inputs and outputs are exposed."""
        hybrid_retrieval = Pipeline()
        hybrid_retrieval.add_component("text_embedder", self.embedder)
        hybrid_retrieval.add_component("embedding_retriever", SolrEmbeddingRetriever(**data["embedding_retriever"]))
        hybrid_retrieval.add_component("bm25_retriever", SolrBM25Retriever(**data["bm25_retriever"]))
        hybrid_retrieval.add_component("document_joiner", DocumentJoiner(**data["document_joiner"]))

        hybrid_retrieval.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
        hybrid_retrieval.connect("bm25_retriever", "document_joiner")
        hybrid_retrieval.connect("embedding_retriever", "document_joiner")

        self.input_mapping = {
            # The single "query" input feeds both the embedder and the BM25 retriever.
            "query": ["text_embedder.text", "bm25_retriever.query"],
            "filters_bm25": ["bm25_retriever.filters"],
            "filters_embedding": ["embedding_retriever.filters"],
            "top_k_bm25": ["bm25_retriever.top_k"],
            "top_k_embedding": ["embedding_retriever.top_k"],
        }
        self.output_mapping = {"document_joiner.documents": "documents"}

        return hybrid_retrieval

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            embedder=component_to_dict(obj=self.embedder, name="embedder"),
            filters_bm25=self.filters_bm25,
            fuzziness=self.fuzziness,
            top_k_bm25=self.top_k_bm25,
            scale_score=self.scale_score,
            all_terms_must_match=self.all_terms_must_match,
            filter_policy_bm25=(
                self.filter_policy_bm25.value
                if isinstance(self.filter_policy_bm25, FilterPolicy)
                else self.filter_policy_bm25
            ),
            filters_embedding=self.filters_embedding,
            top_k_embedding=self.top_k_embedding,
            filter_policy_embedding=(
                self.filter_policy_embedding.value
                if isinstance(self.filter_policy_embedding, FilterPolicy)
                else self.filter_policy_embedding
            ),
            join_mode=(self.join_mode.value if isinstance(self.join_mode, JoinMode) else self.join_mode),
            weights=self.weights,
            top_k=self.top_k,
            sort_by_score=self.sort_by_score,
            **self.extra_args,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SolrHybridRetriever":
        """
        Deserializes the component from a dictionary.

        :param data: dictionary to deserialize from.
        :returns: deserialized component.
        """
        init_parameters = data["init_parameters"]
        init_parameters["document_store"] = SolrDocumentStore.from_dict(init_parameters["document_store"])
        # ToDo: replace with the upcoming generic `deserialize_component_inplace` helper.
        deserialize_chatgenerator_inplace(init_parameters, key="embedder")

        for key in ("filter_policy_bm25", "filter_policy_embedding"):
            if key in init_parameters:
                init_parameters[key] = FilterPolicy.from_str(init_parameters[key])
        if "join_mode" in init_parameters:
            init_parameters["join_mode"] = JoinMode.from_str(init_parameters["join_mode"])

        return default_from_dict(cls, data)

    def close(self) -> None:
        """Close the underlying document store connection."""
        self.document_store.close()

    async def close_async(self) -> None:
        """Close the underlying document store async connection."""
        await self.document_store.close_async()
