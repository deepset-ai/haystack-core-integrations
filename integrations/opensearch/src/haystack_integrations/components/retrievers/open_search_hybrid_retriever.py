# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Literal, Optional, Union

from haystack import Pipeline, default_from_dict, default_to_dict, logging
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.joiners.document_joiner import JoinMode
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.document_stores.types import FilterPolicy
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice

from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever, OpenSearchEmbeddingRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

logger = logging.getLogger(__name__)

# Use LazyImport to conditionally import haystack super_component, which is only available in newer versions
with LazyImport(
    "To use the OpenSearchHybridRetriever you need more recent version of haystack. Run 'pip install haystack-ai>=2.13.0'"  # noqa: E501
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
        # SentenceTransformer
        text_embedder_model: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[ComponentDevice] = None,
        normalize_embeddings: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
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
        join_mode: Union[str, JoinMode] = JoinMode.CONCATENATE,
        weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
        sort_by_score: bool = True,
        # ChatPromptBuilder
        template: Optional[List[ChatMessage]] = None,
        # OpenAIChatGenerator
        generator_model: str = "gpt-4o-mini",
        streaming_callback: Optional[StreamingCallbackT] = None,
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
        # AnswerBuilder
        pattern: Optional[str] = None,
        reference_pattern: Optional[str] = None,
        # extra kwargs
        **kwargs,
    ):
        """
        Initialize the OpenSearchHybridRetriever, a super component to retrieve documents from OpenSearch using
        both embedding-based and keyword-based retrieval methods.

        This super component is tied to a SentenceTransformersTextEmbedder and an OpenAIChatGenerator.

        We don't explicitly define all the init parameters of the components in the constructor, for each
        of the components, since that would be around 50 parameters. Instead, we define the most important ones
        and pass the rest as kwargs. This is to keep the constructor clean and easy to read.
        """

        self.document_store = document_store

        # SentenceTransformersTextEmbedder
        self.text_embedder_model = text_embedder_model
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.model_kwargs = model_kwargs or {}
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.config_kwargs = config_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        self.backend = backend

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

        # ChatPromptBuilder
        self.template = template

        # OpenAIChatGenerator
        self.generator_model = generator_model
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url
        self.organization = organization
        self.generation_kwargs = generation_kwargs or {}
        self.http_client_kwargs = http_client_kwargs or {}

        # AnswerBuilder
        self.pattern = pattern
        self.reference_pattern = reference_pattern

        init_args = {
            "text_embedder": {
                "model": self.text_embedder_model,
                "device": self.device,
                "normalize_embeddings": self.normalize_embeddings,
                "model_kwargs": self.model_kwargs,
                "tokenizer_kwargs": self.tokenizer_kwargs,
                "config_kwargs": self.config_kwargs,
                "encode_kwargs": self.encode_kwargs,
                "backend": self.backend,
            },
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
            "chat_prompt_builder": {
                "template": self.template,
            },
            "generator": {
                "model": self.generator_model,
                "streaming_callback": self.streaming_callback,
                "api_base_url": self.api_base_url,
                "organization": self.organization,
                "generation_kwargs": self.generation_kwargs,
                "http_client_kwargs": self.http_client_kwargs,
            },
            "answer_builder": {"pattern": self.pattern, "reference_pattern": self.reference_pattern},
        }

        # look for extra kwargs for each component and add the document store as init param for the retrievers
        # the DocumentJoiner and the AnswerBuilder have no extra kwargs since all the params are already
        # exposed in the constructor
        if "text_embedder" in kwargs:
            init_args["text_embedder"].update(kwargs["text_embedder"])
        if "bm25_retriever" in kwargs:
            init_args["bm25_retriever"].update(kwargs["bm25_retriever"])
            init_args["bm25_retriever"]["document_store"] = self.document_store
        if "embedding_retriever" in kwargs:
            init_args["embedding_retriever"].update(kwargs["embedding_retriever"])
            init_args["embedding_retriever"]["document_store"] = self.document_store
        if "chat_prompt_builder" in kwargs:
            init_args["chat_prompt_builder"].update(kwargs["chat_prompt_builder"])
        if "generator" in kwargs:
            init_args["generator"].update(kwargs["generator"])

        self.pipeline = self._create_pipeline(init_args)

    def _create_pipeline(self, data: dict[str, Any]) -> Pipeline:
        """
        Create the pipeline for the OpenSearchHybridRetriever.

        If the template for the ChatPromptBuilder is not provided, a default template is used.
        """
        text_embedder = SentenceTransformersTextEmbedder(**data["text_embedder"])
        embedding_retriever = OpenSearchEmbeddingRetriever(**data["embedding_retriever"])
        bm25_retriever = OpenSearchBM25Retriever(**data["bm25_retriever"])
        document_joiner = DocumentJoiner(**data["document_joiner"])
        generator = OpenAIChatGenerator(**data["generator"])
        answer_builder = AnswerBuilder(**data["answer_builder"])

        # if no args for the ChatPromptBuilder are provided or if not template is provided, use the default template
        if not data["chat_prompt_builder"]["template"]:
            default_template = [
                ChatMessage.from_system(
                    "You are a helpful AI assistant. Answer the following question based on the given context information "  # noqa: E501
                    "only. If the context is empty or just a '\n' answer with None, example: 'None'."
                ),
                ChatMessage.from_user(
                    """
                    Context:
                    {% for document in documents %}
                        {{ document.content }}
                    {% endfor %}

                    Question: {{question}}
                    """
                ),
            ]
            data["chat_prompt_builder"]["template"] = default_template

        chat_prompt_builder = ChatPromptBuilder(
            **data["chat_prompt_builder"], required_variables=["question", "documents"]
        )

        hybrid_retrieval = Pipeline()
        hybrid_retrieval.add_component("text_embedder", text_embedder)
        hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
        hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
        hybrid_retrieval.add_component("document_joiner", document_joiner)
        hybrid_retrieval.add_component("prompt_builder", chat_prompt_builder)
        hybrid_retrieval.add_component("llm", generator)
        hybrid_retrieval.add_component("answer_builder", answer_builder)

        hybrid_retrieval.connect("text_embedder", "embedding_retriever")
        hybrid_retrieval.connect("bm25_retriever", "document_joiner")
        hybrid_retrieval.connect("embedding_retriever", "document_joiner")
        hybrid_retrieval.connect("document_joiner.documents", "prompt_builder.documents")
        hybrid_retrieval.connect("prompt_builder", "llm")
        hybrid_retrieval.connect("llm.replies", "answer_builder.replies")

        # Define how pipeline inputs/outputs map to subcomponent inputs/outputs
        self.input_mapping = {
            # The pipeline input "query" feeds into each of the retrievers, the prompt builder, and the answer builder
            "query": ["text_embedder.text", "bm25_retriever.query", "prompt_builder.question", "answer_builder.query"],
        }
        # The pipeline output "answers" comes from "answer_builder.answers"
        self.output_mapping = {"answer_builder.answers": "answers"}

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
            # SentenceTransformer
            text_embedder_model=self.text_embedder_model,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            config_kwargs=self.config_kwargs,
            encode_kwargs=self.encode_kwargs,
            backend=self.backend,
            # OpenSearchBM25Retriever
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
            # ChatPromptBuilder
            template=self.template,
            # OpenAIChatGenerator
            generator_model=self.generator_model,
            streaming_callback=self.streaming_callback,
            api_base_url=self.api_base_url,
            organization=self.organization,
            generation_kwargs=self.generation_kwargs,
            http_client_kwargs=self.http_client_kwargs,
            # AnswerBuilder
            pattern=self.pattern,
            reference_pattern=self.reference_pattern,
        )

    @classmethod
    def from_dict(cls, data):
        # deserialize the document store
        doc_store = OpenSearchDocumentStore.from_dict(data["init_parameters"]["document_store"])
        data["init_parameters"]["document_store"] = doc_store
        return default_from_dict(cls, data)
