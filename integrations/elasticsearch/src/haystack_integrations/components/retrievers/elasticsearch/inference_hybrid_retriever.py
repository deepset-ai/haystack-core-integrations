# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from haystack import Document, Pipeline, default_from_dict, default_to_dict, super_component
from haystack.components.joiners import DocumentJoiner
from haystack.components.joiners.document_joiner import JoinMode
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

from .bm25_retriever import ElasticsearchBM25Retriever
from .inference_sparse_retriever import ElasticsearchInferenceSparseRetriever


@super_component
class ElasticsearchInferenceHybridRetriever:
    """
    A hybrid retriever that combines BM25 keyword search and Elasticsearch inference-based sparse vector search.

    This retriever sends the query to both an Elasticsearch BM25 retriever and an inference-based sparse
    vector retriever (e.g. ELSER), then merges the results through a DocumentJoiner.
    Because sparse vector inference runs server-side inside Elasticsearch, no local embedding model is required.

    Example usage:

    Make sure Elasticsearch is running with a deployed ELSER model:

        docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" \\
          docker.elastic.co/elasticsearch/elasticsearch:8.0.0

    ```python
    from haystack import Document
    from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchInferenceHybridRetriever
    from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

    doc_store = ElasticsearchDocumentStore(
        hosts="http://localhost:9200",
        index="document_store",
        sparse_vector_field="sparse_vec",
    )

    docs = [
        Document(content="Machine learning is a subset of artificial intelligence."),
        Document(content="Deep learning is a subset of machine learning."),
    ]
    doc_store.write_documents(docs)

    retriever = ElasticsearchInferenceHybridRetriever(
        document_store=doc_store,
        inference_id=".elser_model_2",
        top_k_bm25=5,
        top_k_sparse=5,
    )

    results = retriever.run(query="What is reinforcement learning?")
    >> results["documents"]
    [Document(id=..., content="Machine learning is a subset of artificial intelligence.", score=1.0), ...]
    ```
    """

    def __init__(
        self,
        document_store: ElasticsearchDocumentStore,
        *,
        inference_id: str,
        # ElasticsearchBM25Retriever
        filters_bm25: dict[str, Any] | None = None,
        fuzziness: str = "AUTO",
        top_k_bm25: int = 10,
        scale_score: bool = False,
        filter_policy_bm25: str | FilterPolicy = FilterPolicy.REPLACE,
        # ElasticsearchInferenceSparseRetriever
        filters_sparse: dict[str, Any] | None = None,
        top_k_sparse: int = 10,
        filter_policy_sparse: str | FilterPolicy = FilterPolicy.REPLACE,
        # DocumentJoiner
        join_mode: str | JoinMode = JoinMode.RECIPROCAL_RANK_FUSION,
        weights: list[float] | None = None,
        top_k: int | None = None,
        sort_by_score: bool = True,
        # extra kwargs
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ElasticsearchInferenceHybridRetriever.

        This is a super component combining BM25 and Elasticsearch inference-based sparse retrieval.
        Inference runs entirely on the Elasticsearch side — no local embedding model is required.

        To pass extra parameters to individual sub-retrievers, use kwargs with the component name as key:
            - "bm25_retriever" -> ElasticsearchBM25Retriever
            - "sparse_retriever" -> ElasticsearchInferenceSparseRetriever

        :param document_store: The ElasticsearchDocumentStore to retrieve from.
        :param inference_id: Elasticsearch inference model ID used for sparse vector search (e.g. ".elser_model_2").
        :param filters_bm25: Filters for the BM25 retriever.
        :param fuzziness: Fuzziness for BM25 matching. See Elasticsearch docs for valid values.
        :param top_k_bm25: Maximum number of documents to return from the BM25 retriever.
        :param scale_score: Whether to scale BM25 scores to [0, 1] using a sigmoid.
        :param filter_policy_bm25: Filter policy for the BM25 retriever.
        :param filters_sparse: Filters for the inference sparse retriever.
        :param top_k_sparse: Maximum number of documents to return from the sparse retriever.
        :param filter_policy_sparse: Filter policy for the inference sparse retriever.
        :param join_mode: How to merge results from both retrievers.
        :param weights: Weights for the document joiner (relevant for LINEAR join mode).
        :param top_k: Maximum number of documents to return after joining.
        :param sort_by_score: Whether to sort the final results by score.
        :param **kwargs: Extra parameters for sub-retrievers. Keys: "bm25_retriever", "sparse_retriever".
        :raises ValueError: If `document_store` is not an ElasticsearchDocumentStore, `inference_id` is empty,
            or unknown kwargs are passed.
        """
        if not isinstance(document_store, ElasticsearchDocumentStore):
            msg = "document_store must be an instance of ElasticsearchDocumentStore"
            raise ValueError(msg)

        if not inference_id:
            msg = "inference_id must be provided"
            raise ValueError(msg)

        self.document_store = document_store
        self.inference_id = inference_id

        # BM25 retriever params
        self.filters_bm25 = filters_bm25
        self.fuzziness = fuzziness
        self.top_k_bm25 = top_k_bm25
        self.scale_score = scale_score
        self.filter_policy_bm25 = filter_policy_bm25

        # Inference sparse retriever params
        self.filters_sparse = filters_sparse
        self.top_k_sparse = top_k_sparse
        self.filter_policy_sparse = filter_policy_sparse

        # DocumentJoiner params
        self.join_mode = join_mode
        self.weights = weights
        self.top_k = top_k
        self.sort_by_score = sort_by_score

        for k in kwargs:
            if k not in ["bm25_retriever", "sparse_retriever"]:
                msg = f"valid extra args are only: 'bm25_retriever' and 'sparse_retriever'. Found: {k}"
                raise ValueError(msg)

        self.extra_args = kwargs

        init_args: dict[str, Any] = {
            "bm25_retriever": {
                "document_store": self.document_store,
                "filters": self.filters_bm25,
                "fuzziness": self.fuzziness,
                "top_k": self.top_k_bm25,
                "scale_score": self.scale_score,
                "filter_policy": self.filter_policy_bm25,
            },
            "sparse_retriever": {
                "document_store": self.document_store,
                "inference_id": self.inference_id,
                "filters": self.filters_sparse,
                "top_k": self.top_k_sparse,
                "filter_policy": self.filter_policy_sparse,
            },
            "document_joiner": {
                "join_mode": self.join_mode,
                "weights": self.weights,
                "top_k": self.top_k,
                "sort_by_score": self.sort_by_score,
            },
        }

        if "bm25_retriever" in kwargs:
            init_args["bm25_retriever"].update(kwargs["bm25_retriever"])
            init_args["bm25_retriever"]["document_store"] = self.document_store
        if "sparse_retriever" in kwargs:
            init_args["sparse_retriever"].update(kwargs["sparse_retriever"])
            init_args["sparse_retriever"]["document_store"] = self.document_store
            init_args["sparse_retriever"]["inference_id"] = self.inference_id

        self.pipeline = self._create_pipeline(init_args)

    if TYPE_CHECKING:

        def warm_up(self) -> None:
            """Warm up the underlying pipeline components."""
            ...

        def run(
            self,
            query: str,
            filters_bm25: dict[str, Any] | None = None,
            filters_sparse: dict[str, Any] | None = None,
            top_k_bm25: int | None = None,
            top_k_sparse: int | None = None,
        ) -> dict[str, list[Document]]:
            """Run the hybrid retrieval pipeline and return retrieved documents."""
            ...

    def _create_pipeline(self, data: dict[str, Any]) -> Pipeline:
        bm25_retriever = ElasticsearchBM25Retriever(**data["bm25_retriever"])
        sparse_retriever = ElasticsearchInferenceSparseRetriever(**data["sparse_retriever"])
        document_joiner = DocumentJoiner(**data["document_joiner"])

        pipeline = Pipeline()
        pipeline.add_component("bm25_retriever", bm25_retriever)
        pipeline.add_component("sparse_retriever", sparse_retriever)
        pipeline.add_component("document_joiner", document_joiner)

        pipeline.connect("bm25_retriever", "document_joiner")
        pipeline.connect("sparse_retriever", "document_joiner")

        self.input_mapping = {
            "query": ["bm25_retriever.query", "sparse_retriever.query"],
            "filters_bm25": ["bm25_retriever.filters"],
            "filters_sparse": ["sparse_retriever.filters"],
            "top_k_bm25": ["bm25_retriever.top_k"],
            "top_k_sparse": ["sparse_retriever.top_k"],
        }
        self.output_mapping = {"document_joiner.documents": "documents"}

        return pipeline

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            inference_id=self.inference_id,
            # BM25
            filters_bm25=self.filters_bm25,
            fuzziness=self.fuzziness,
            top_k_bm25=self.top_k_bm25,
            scale_score=self.scale_score,
            filter_policy_bm25=(
                self.filter_policy_bm25.value
                if isinstance(self.filter_policy_bm25, FilterPolicy)
                else self.filter_policy_bm25
            ),
            # Sparse inference
            filters_sparse=self.filters_sparse,
            top_k_sparse=self.top_k_sparse,
            filter_policy_sparse=(
                self.filter_policy_sparse.value
                if isinstance(self.filter_policy_sparse, FilterPolicy)
                else self.filter_policy_sparse
            ),
            # DocumentJoiner
            join_mode=(self.join_mode.value if isinstance(self.join_mode, JoinMode) else self.join_mode),
            weights=self.weights,
            top_k=self.top_k,
            sort_by_score=self.sort_by_score,
            # extra kwargs
            **self.extra_args,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ElasticsearchInferenceHybridRetriever":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized ElasticsearchInferenceHybridRetriever instance.
        """
        doc_store = ElasticsearchDocumentStore.from_dict(data["init_parameters"]["document_store"])
        data["init_parameters"]["document_store"] = doc_store

        if "filter_policy_bm25" in data["init_parameters"]:
            data["init_parameters"]["filter_policy_bm25"] = FilterPolicy.from_str(
                data["init_parameters"]["filter_policy_bm25"]
            )
        if "filter_policy_sparse" in data["init_parameters"]:
            data["init_parameters"]["filter_policy_sparse"] = FilterPolicy.from_str(
                data["init_parameters"]["filter_policy_sparse"]
            )
        if "join_mode" in data["init_parameters"]:
            data["init_parameters"]["join_mode"] = JoinMode.from_str(data["init_parameters"]["join_mode"])

        return default_from_dict(cls, data)
