# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
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

from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

from .bm25_retriever import ElasticsearchBM25Retriever
from .embedding_retriever import ElasticsearchEmbeddingRetriever

logger = logging.getLogger(__name__)


@super_component
class ElasticsearchHybridRetriever:
    """
    A hybrid retriever that combines embedding-based and keyword-based retrieval from Elasticsearch.

    This retriever uses both BM25 keyword search and vector similarity search, combining the results
    through a document joiner for improved relevance ranking.

    Example usage:

    Make sure you have "sentence-transformers>=3.0.0":

        pip install haystack-ai datasets "sentence-transformers>=3.0.0"

    And Elasticsearch running. You can run Elasticsearch with Docker:

        docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" \
          -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.0.0

    ```python
    from haystack import Document
    from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
    from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchHybridRetriever
    from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

    # Initialize the document store
    doc_store = ElasticsearchDocumentStore(
        hosts="http://localhost:9200",
        index="document_store",
        embedding_dim=384,
    )

    # Create some sample documents
    docs = [
        Document(content="Machine learning is a subset of artificial intelligence."),
        Document(content="Deep learning is a subset of machine learning."),
        Document(content="Natural language processing is a field of AI."),
        Document(content="Reinforcement learning is a type of machine learning."),
        Document(content="Supervised learning is a type of machine learning."),
    ]

    # Embed the documents and add them to the document store
    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    doc_embedder.warm_up()
    docs = doc_embedder.run(docs)
    doc_store.write_documents(docs['documents'])

    # Initialize a text embedder
    embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize the hybrid retriever
    retriever = ElasticsearchHybridRetriever(
        document_store=doc_store,
        embedder=embedder,
        top_k_bm25=3,
        top_k_embedding=3,
        join_mode="reciprocal_rank_fusion"
    )

    # Run the retriever
    results = retriever.run(query="What is reinforcement learning?", filters_bm25=None, filters_embedding=None)

    >> results['documents']
    [Document(id=..., content='Reinforcement learning is a type of machine learning.', score=1.0),
     Document(id=..., content='Supervised learning is a type of machine learning.', score=0.9760624679979518),
     Document(id=..., content='Deep learning is a subset of machine learning.', score=0.4919354838709677),
     Document(id=..., content='Machine learning is a subset of artificial intelligence.', score=0.4841269841269841)]
    ```
    """

    def __init__(
        self,
        document_store: ElasticsearchDocumentStore,
        *,
        embedder: TextEmbedder,
        # ElasticsearchBM25Retriever
        filters_bm25: dict[str, Any] | None = None,
        fuzziness: str = "AUTO",
        top_k_bm25: int = 10,
        scale_score: bool = False,
        filter_policy_bm25: str | FilterPolicy = FilterPolicy.REPLACE,
        # ElasticsearchEmbeddingRetriever
        filters_embedding: dict[str, Any] | None = None,
        top_k_embedding: int = 10,
        num_candidates: int | None = None,
        filter_policy_embedding: str | FilterPolicy = FilterPolicy.REPLACE,
        # DocumentJoiner
        join_mode: str | JoinMode = JoinMode.RECIPROCAL_RANK_FUSION,
        weights: list[float] | None = None,
        top_k: int | None = None,
        sort_by_score: bool = True,
        # extra kwargs
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ElasticsearchHybridRetriever using both embedding-based and keyword-based retrieval methods.

        This is a super component to retrieve documents from Elasticsearch using both retrieval methods.

        We don't explicitly define all the init parameters of the components in the constructor, for each
        of the components, to keep the constructor clean and easy to read. If you need to pass extra parameters to
        the components, you can do so by passing them as kwargs. It expects a dictionary with the component name as
        the key and the parameters as the value. The component name should be:

            - "bm25_retriever" -> ElasticsearchBM25Retriever
            - "embedding_retriever" -> ElasticsearchEmbeddingRetriever

        :param document_store:
            The ElasticsearchDocumentStore to use for retrieval.
        :param embedder:
            A TextEmbedder to use for embedding the query.
            See `haystack.components.embedders.types.protocol.TextEmbedder` for more information.
        :param filters_bm25:
            Filters for the BM25 retriever.
        :param fuzziness:
            The fuzziness for the BM25 retriever. See Elasticsearch documentation for valid values.
        :param top_k_bm25:
            The number of results to return from the BM25 retriever.
        :param scale_score:
            Whether to scale the score for the BM25 retriever.
        :param filter_policy_bm25:
            The filter policy for the BM25 retriever.
        :param filters_embedding:
            Filters for the embedding retriever.
        :param top_k_embedding:
            The number of results to return from the embedding retriever.
        :param num_candidates:
            Number of approximate nearest neighbor candidates to consider per shard in the embedding retriever.
            Defaults to top_k_embedding * 10 if not specified. Higher values improve accuracy at the cost of latency.
        :param filter_policy_embedding:
            The filter policy for the embedding retriever.
        :param join_mode:
            The mode to use for joining the results from the BM25 and embedding retrievers.
        :param weights:
            The weights for the joiner.
        :param top_k:
            The number of results to return from the joiner.
        :param sort_by_score:
            Whether to sort the results by score.
        :param **kwargs:
            Additional keyword arguments. Use the following keys to pass extra parameters to the retrievers:
            - "bm25_retriever" -> ElasticsearchBM25Retriever
            - "embedding_retriever" -> ElasticsearchEmbeddingRetriever

        :raises ValueError: If invalid extra kwargs are provided.
        """
        self.document_store = document_store
        self.embedder = embedder

        # ElasticsearchBM25Retriever
        self.filters_bm25 = filters_bm25
        self.fuzziness = fuzziness
        self.top_k_bm25 = top_k_bm25
        self.scale_score = scale_score
        self.filter_policy_bm25 = filter_policy_bm25

        # ElasticsearchEmbeddingRetriever
        self.filters_embedding = filters_embedding
        self.top_k_embedding = top_k_embedding
        self.num_candidates = num_candidates
        self.filter_policy_embedding = filter_policy_embedding

        # DocumentJoiner
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
                "filter_policy": self.filter_policy_bm25,
            },
            "embedding_retriever": {
                "document_store": self.document_store,
                "filters": self.filters_embedding,
                "top_k": self.top_k_embedding,
                "num_candidates": self.num_candidates,
                "filter_policy": self.filter_policy_embedding,
            },
            "document_joiner": {
                "join_mode": self.join_mode,
                "weights": self.weights,
                "top_k": self.top_k,
                "sort_by_score": self.sort_by_score,
            },
        }

        for k in kwargs:
            if k not in ["bm25_retriever", "embedding_retriever"]:
                msg = f"valid extra args are only: 'bm25_retriever' and 'embedding_retriever'. Found: {k}"
                raise ValueError(msg)

        self.extra_args = kwargs

        # handle extra kwargs for the bm25 and embedding retrievers and the doc store as init param
        if "bm25_retriever" in kwargs:
            init_args["bm25_retriever"].update(kwargs["bm25_retriever"])
            init_args["bm25_retriever"]["document_store"] = self.document_store
        if "embedding_retriever" in kwargs:
            init_args["embedding_retriever"].update(kwargs["embedding_retriever"])
            init_args["embedding_retriever"]["document_store"] = self.document_store

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
            """Run the hybrid retrieval pipeline and return retrieved documents."""
            ...

    def _create_pipeline(self, data: dict[str, Any]) -> Pipeline:
        """
        Create the pipeline for the ElasticsearchHybridRetriever.

        :param data: Dictionary containing initialization parameters for the retrievers and joiner.
        :returns: A Pipeline configured with BM25 and embedding retrievers joined via DocumentJoiner.
        """
        embedding_retriever = ElasticsearchEmbeddingRetriever(**data["embedding_retriever"])
        bm25_retriever = ElasticsearchBM25Retriever(**data["bm25_retriever"])
        document_joiner = DocumentJoiner(**data["document_joiner"])

        hybrid_retrieval = Pipeline()
        hybrid_retrieval.add_component("text_embedder", self.embedder)
        hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
        hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
        hybrid_retrieval.add_component("document_joiner", document_joiner)

        hybrid_retrieval.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
        hybrid_retrieval.connect("bm25_retriever", "document_joiner")
        hybrid_retrieval.connect("embedding_retriever", "document_joiner")

        # Define how pipeline inputs/outputs map to subcomponent inputs/outputs
        self.input_mapping = {
            # The pipeline input "query" feeds into each of the retrievers
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
        Serialize ElasticsearchHybridRetriever to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            # DocumentStore
            document_store=self.document_store.to_dict(),
            embedder=component_to_dict(obj=self.embedder, name="embedder"),
            # BM25Retriever
            filters_bm25=self.filters_bm25,
            fuzziness=self.fuzziness,
            top_k_bm25=self.top_k_bm25,
            scale_score=self.scale_score,
            filter_policy_bm25=(
                self.filter_policy_bm25.value
                if isinstance(self.filter_policy_bm25, FilterPolicy)
                else self.filter_policy_bm25
            ),
            # EmbeddingRetriever
            filters_embedding=self.filters_embedding,
            top_k_embedding=self.top_k_embedding,
            num_candidates=self.num_candidates,
            filter_policy_embedding=(
                self.filter_policy_embedding.value
                if isinstance(self.filter_policy_embedding, FilterPolicy)
                else self.filter_policy_embedding
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
    def from_dict(cls, data: dict[str, Any]) -> "ElasticsearchHybridRetriever":
        """
        Deserialize an ElasticsearchHybridRetriever from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized ElasticsearchHybridRetriever instance.
        """
        # deserialize the document store
        doc_store = ElasticsearchDocumentStore.from_dict(data["init_parameters"]["document_store"])
        data["init_parameters"]["document_store"] = doc_store

        # deserialize the embedder
        # ToDo: in the future we should use the upcoming generic `deserialize_component_inplace` function
        deserialize_chatgenerator_inplace(data["init_parameters"], key="embedder")

        # deserialize the filter policies
        if "filter_policy_bm25" in data["init_parameters"]:
            filter_policy_bm25 = FilterPolicy.from_str(data["init_parameters"]["filter_policy_bm25"])
            data["init_parameters"]["filter_policy_bm25"] = filter_policy_bm25

        if "filter_policy_embedding" in data["init_parameters"]:
            filter_policy_embedding = FilterPolicy.from_str(data["init_parameters"]["filter_policy_embedding"])
            data["init_parameters"]["filter_policy_embedding"] = filter_policy_embedding

        if "join_mode" in data["init_parameters"]:
            join_mode = JoinMode.from_str(data["init_parameters"]["join_mode"])
            data["init_parameters"]["join_mode"] = join_mode

        return default_from_dict(cls, data)
