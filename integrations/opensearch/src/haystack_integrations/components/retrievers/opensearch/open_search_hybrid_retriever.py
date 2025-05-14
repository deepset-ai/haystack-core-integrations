# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

from haystack import DeserializationError, Pipeline, default_from_dict, default_to_dict, logging
from haystack.components.embedders.types import TextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.joiners.document_joiner import JoinMode
from haystack.core.serialization import component_from_dict, import_class_by_name
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever, OpenSearchEmbeddingRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

# Use LazyImport to conditionally import haystack super_component, which is only available in haystack-ai>=2.14.0
with LazyImport(
    "To use the OpenSearchHybridRetriever you need a more recent version of haystack. Run 'pip install haystack-ai>=2.14.0'"  # noqa: E501
) as haystack_imports:
    from haystack import super_component

# Trigger an error message when the OpenSearchHybridRetriever is imported without haystack-ai>=2.13.0
haystack_imports.check()


@super_component
class OpenSearchHybridRetriever:
    """
    A hybrid retriever that combines embedding-based and keyword-based retrieval from OpenSearch.

    This component requires haystack-ai>=2.13.0 to work properly.
    """

    def __init__(
        self,
        document_store: OpenSearchDocumentStore,
        *,
        embedder: TextEmbedder,
        # OpenSearchBM25Retriever
        filters_bm25: Optional[Dict[str, Any]] = None,
        fuzziness: Union[int, str] = "AUTO",
        top_k_bm25: int = 10,
        scale_score: bool = False,
        all_terms_must_match: bool = False,
        filter_policy_bm25: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
        custom_query_bm25: Optional[Dict[str, Any]] = None,
        # OpenSearchEmbeddingRetriever
        filters_embedding: Optional[Dict[str, Any]] = None,
        top_k_embedding: int = 10,
        filter_policy_embedding: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
        custom_query_embedding: Optional[Dict[str, Any]] = None,
        # DocumentJoiner
        join_mode: Union[str, JoinMode] = JoinMode.RECIPROCAL_RANK_FUSION,
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        sort_by_score: bool = True,
        # extra kwargs
        **kwargs,
    ):
        """
        Initialize the OpenSearchHybridRetriever, a super component to retrieve documents from OpenSearch using
        both embedding-based and keyword-based retrieval methods.

        We don't explicitly define all the init parameters of the components in the constructor, for each
        of the components, since that would be around 20+ parameters. Instead, we define the most important ones
        and pass the rest as kwargs. This is to keep the constructor clean and easy to read.

        If you need to pass extra parameters to the components, you can do so by passing them as kwargs. It expects
        a dictionary with the component name as the key and the parameters as the value. The component name should be:

            - "bm25_retriever" -> OpenSearchBM25Retriever
            - "embedding_retriever" -> OpenSearchEmbeddingRetriever

        :param document_store:
            The OpenSearchDocumentStore to use for retrieval.

        :param embedder:
            A TextEmbedder to use for embedding the query.
            See `haystack.components.embedders.types.protocol.TextEmbedder` for more information.

        :param filters_bm25:
            Filters for the BM25 retriever.

        :param fuzziness:
            The fuzziness for the BM25 retriever.

        :param top_k_bm25:
            The number of results to return from the BM25 retriever.

        :param scale_score:
            Whether to scale the score for the BM25 retriever.

        :param all_terms_must_match:
            Whether all terms must match for the BM25 retriever.

        :param filter_policy_bm25:
            The filter policy for the BM25 retriever.

        :param custom_query_bm25:
            A custom query for the BM25 retriever.

        :param filters_embedding:
            Filters for the embedding retriever.

        :param top_k_embedding:
            The number of results to return from the embedding retriever.

        :param filter_policy_embedding:
            The filter policy for the embedding retriever.

        :param custom_query_embedding:
            A custom query for the embedding retriever.

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

            - "bm25_retriever" -> OpenSearchBM25Retriever
            - "embedding_retriever" -> OpenSearchEmbeddingRetriever


        """
        self.document_store = document_store
        self.embedder = embedder

        # OpenSearchBM25Retriever
        self.filters_bm25 = filters_bm25
        self.fuzziness = fuzziness
        self.top_k_bm25 = top_k_bm25
        self.scale_score = scale_score
        self.all_terms_must_match = all_terms_must_match
        self.filter_policy_bm25 = filter_policy_bm25
        self.custom_query_bm25 = custom_query_bm25

        # OpenSearchEmbeddingRetriever
        self.filters_embedding = filters_embedding
        self.top_k_embedding = top_k_embedding
        self.filter_policy_embedding = filter_policy_embedding
        self.custom_query_embedding = custom_query_embedding

        # DocumentJoiner
        self.join_mode = join_mode
        self.weights = weights
        self.top_k = top_k
        self.sort_by_score = sort_by_score

        init_args = {
            "bm25_retriever": {
                "document_store": self.document_store,
                "filters": self.filters_bm25,
                "fuzziness": self.fuzziness,
                "top_k": self.top_k_bm25,
                "scale_score": self.scale_score,
                "all_terms_must_match": self.all_terms_must_match,
                "filter_policy": self.filter_policy_bm25,
                "custom_query": self.custom_query_bm25,
            },
            "embedding_retriever": {
                "document_store": self.document_store,
                "filters": self.filters_embedding,
                "top_k": self.top_k_embedding,
                "filter_policy": self.filter_policy_embedding,
                "custom_query": self.custom_query_embedding,
            },
            "document_joiner": {
                "join_mode": self.join_mode,
                "weights": self.weights,
                "top_k": self.top_k,
                "sort_by_score": self.sort_by_score,
            },
        }

        # kwargs must contain 'bm25_retriever' and 'embedding_retriever' as top-level keys
        if kwargs and not all(key in kwargs for key in ["bm25_retriever", "embedding_retriever"]):
            raise ValueError(
                "kwargs can only contain 'bm25_retriever', 'embedding_retriever' as top-level keys."
            )

        self.extra_args = kwargs

        # handle extra kwargs for the bm25 and embedding retrievers and the doc store as init param
        if "bm25_retriever" in kwargs:
            init_args["bm25_retriever"].update(kwargs["bm25_retriever"])
            init_args["bm25_retriever"]["document_store"] = self.document_store
        if "embedding_retriever" in kwargs:
            init_args["embedding_retriever"].update(kwargs["embedding_retriever"])
            init_args["embedding_retriever"]["document_store"] = self.document_store

        self.pipeline = self._create_pipeline(init_args)

    def _create_pipeline(self, data: dict[str, Any]) -> Pipeline:
        """
        Create the pipeline for the OpenSearchHybridRetriever.
        """
        embedding_retriever = OpenSearchEmbeddingRetriever(**data["embedding_retriever"])
        bm25_retriever = OpenSearchBM25Retriever(**data["bm25_retriever"])
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
        }
        self.output_mapping = {"document_joiner.documents": "documents"}

        return hybrid_retrieval

    def to_dict(self):
        """
        Serialize OpenSearchHybridRetriever to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            # DocumentStore
            document_store=self.document_store.to_dict(),
            embedder=self.embedder.to_dict(),
            filters_bm25=self.filters_bm25,
            fuzziness=self.fuzziness,
            top_k_bm25=self.top_k_bm25,
            scale_score=self.scale_score,
            all_terms_must_match=self.all_terms_must_match,
            filter_policy_bm25=self.filter_policy_bm25.value,
            custom_query_bm25=self.custom_query_bm25,
            # OpenSearchEmbeddingRetriever
            filters_embedding=self.filters_embedding,
            top_k_embedding=self.top_k_embedding,
            filter_policy_embedding=self.filter_policy_embedding.value,
            custom_query_embedding=self.custom_query_embedding,
            # DocumentJoiner
            join_mode=self.join_mode.value,
            weights=self.weights,
            top_k=self.top_k,
            sort_by_score=self.sort_by_score,
            # extra kwargs
            **self.extra_args,
        )

    @classmethod
    def from_dict(cls, data):

        # deserialize the document store
        doc_store = OpenSearchDocumentStore.from_dict(data["init_parameters"]["document_store"])
        data["init_parameters"]["document_store"] = doc_store

        # deserialize the embedder
        try:
            text_embedder_class = import_class_by_name(data["init_parameters"]["embedder"]["type"])
        except ImportError as e:
            msg = f"Class '{data['init_parameters']['embedder']['type']}' not correctly imported"
            raise DeserializationError(msg) from e

        data["init_parameters"]["embedder"] = component_from_dict(
            cls=text_embedder_class, data=data["init_parameters"]["embedder"], name="embedder"
        )

        # deserialize the embedders filtering policy
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
