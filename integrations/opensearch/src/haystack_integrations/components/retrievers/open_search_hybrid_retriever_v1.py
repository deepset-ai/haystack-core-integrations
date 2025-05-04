# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import  Optional, Dict, Any

from haystack import Pipeline, logging
from haystack.components.builders import ChatPromptBuilder, AnswerBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport

from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever
from haystack_integrations.components.retrievers.opensearch import OpenSearchEmbeddingRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

logger = logging.getLogger(__name__)

# Use LazyImport to conditionally import haystack super_component, which is only available in newer versions
with LazyImport("To use the OpenSearchHybridRetriever you need more recent version of haystack. Run 'pip install haystack-ai>=2.13.0' ") as haystack_imports:   # noqa: F401
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
            # component-specific kwargs
            text_embedder_kwargs: Optional[Dict[str, Any]] = None,
            bm25_retriever_kwargs: Optional[Dict[str, Any]] = None,
            embedding_retriever_kwargs: Optional[Dict[str, Any]] = None,
            document_joiner_kwargs: Optional[Dict[str, Any]] = None,
            chat_prompt_builder_kwargs: Optional[Dict[str, Any]] = None,
            generator_kwargs: Optional[Dict[str, Any]] = None,
            answer_builder_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the OpenSearchHybridRetriever, a super component to retrieve documents from OpenSearch using
        both embedding-based and keyword-based retrieval methods.

        This super component is tied to a SentenceTransformersTextEmbedder and an OpenAIChatGenerator.

        :param document_store: The OpenSearch document store to use for retrieval, which must be an instance of
        OpenSearchDocumentStore. See https://docs.haystack.deepset.ai/reference/integrations-opensearch#opensearchdocumentstore   # noqa: F401
        for more details.

        :param text_embedder_kwargs: Parameters for the SentenceTransformersTextEmbedder.
        Defaults are: {"model": "sentence-transformers/all-mpnet-base-v2", "progress_bar": False}
        See: https://docs.haystack.deepset.ai/reference/components-embedders#sentence-transformers-text-embedder

        :param bm25_retriever_kwargs: Parameters for the OpenSearchBM25Retriever. Defaults: {"top_k": 10}
        See: https://docs.haystack.deepset.ai/reference/integrations-opensearch#opensearchbm25retriever

        :param embedding_retriever_kwargs: Parameters for the OpenSearchEmbeddingRetriever. Defaults are {"top_k": 10}.
        See: https://docs.haystack.deepset.ai/reference/integrations-opensearch#opensearchembeddingretriever

        :param document_joiner_kwargs: Parameters for the DocumentJoiner. Defaults are: {"join_mode": JoinMode.CONCATENATE}
        See: https://docs.haystack.deepset.ai/reference/components-joiners#document-joiner

        :param chat_prompt_builder_kwargs: Parameters for the ChatPromptBuilder.
                Default: Uses a default template for question answering

        :param generator_kwargs: Parameters for the OpenAIChatGenerator.
                Default: {}

        :param answer_builder_kwargs: Parameters for the AnswerBuilder.
                Default: {}
        """
        self.document_store = document_store
        
        # SentenceTransformer default kwargs for each component
        self.text_embedder_kwargs = text_embedder_kwargs or {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "progress_bar": False
        }
        # BM25Retriever default kwargs
        self.bm25_retriever_kwargs = bm25_retriever_kwargs or {"top_k": 10}
        self.bm25_retriever_kwargs['document_store'] = self.document_store

        # EmbeddingRetriever default kwargs
        self.embedding_retriever_kwargs = embedding_retriever_kwargs or {"top_k": 10}
        self.embedding_retriever_kwargs['document_store'] = self.document_store

        # DocumentJoiner, ChatPromptBuilder, OpenAIChatGenerator, and AnswerBuilder default kwargs
        self.document_joiner_kwargs = document_joiner_kwargs or {"join_mode": "concatenate"}
        self.chat_prompt_builder_kwargs = chat_prompt_builder_kwargs
        self.generator_kwargs = generator_kwargs or {}
        self.answer_builder_kwargs = answer_builder_kwargs or {}

        self.pipeline = self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        """
        Create the pipeline for the OpenSearchHybridRetriever.
        """
        text_embedder = SentenceTransformersTextEmbedder(**self.text_embedder_kwargs)
        embedding_retriever = OpenSearchEmbeddingRetriever(**self.embedding_retriever_kwargs)
        bm25_retriever = OpenSearchBM25Retriever(**self.bm25_retriever_kwargs)
        document_joiner = DocumentJoiner(**self.document_joiner_kwargs)
        generator = OpenAIChatGenerator(**self.generator_kwargs)
        answer_builder = AnswerBuilder(**self.answer_builder_kwargs)

        # if no args for the ChatPromptBuilder are provided or if no template is provided, use the default template
        self.chat_prompt_builder_kwargs = self.chat_prompt_builder_kwargs or {}
        if not self.chat_prompt_builder_kwargs or "template" not in self.chat_prompt_builder_kwargs:
            default_template = [
                ChatMessage.from_system(
                    "You are a helpful AI assistant. Answer the following question based on the given context information "
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
                )
            ]
            self.chat_prompt_builder_kwargs["template"] = default_template

        # check if we are passing a serialised template
        if isinstance(self.chat_prompt_builder_kwargs["template"], list):
            template = []
            for item in self.chat_prompt_builder_kwargs["template"]:
                if isinstance(item, dict):
                    template.append(ChatMessage.from_dict(item))
                else:
                    template.append(item)
            self.chat_prompt_builder_kwargs["template"] = template

        chat_prompt_builder = ChatPromptBuilder(
            **self.chat_prompt_builder_kwargs,
            required_variables=["question", "documents"]
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

        return hybrid_retrieval

    def to_dict(self):

        serialised = {
            "init_parameters": {
                "document_store": self.document_store.to_dict(),
                "text_embedder_kwargs": self.text_embedder_kwargs,
                "bm25_retriever_kwargs": self.bm25_retriever_kwargs,
                "embedding_retriever_kwargs": self.embedding_retriever_kwargs,
                "document_joiner_kwargs": self.document_joiner_kwargs,
                "chat_prompt_builder_kwargs": self.chat_prompt_builder_kwargs,
                "generator_kwargs": self.generator_kwargs,
                "answer_builder_kwargs": self.answer_builder_kwargs
            },
            "type": "haystack_integrations.components.retrievers.open_search_hybrid_retriever",
        }

        # serialise the template
        if isinstance(self.chat_prompt_builder_kwargs['template'], list):
            template = []
            for item in self.chat_prompt_builder_kwargs['template']:
                if isinstance(item, ChatMessage):
                    template.append(item.to_dict())
                else:
                    template.append(item)
            serialised["init_parameters"]["chat_prompt_builder_kwargs"]["template"] = template

        # connected the serialise the document store to the retrievers kwargs
        doc_store = serialised["init_parameters"]["document_store"]
        serialised["init_parameters"]["bm25_retriever_kwargs"]['document_store'] = doc_store
        serialised["init_parameters"]["embedding_retriever_kwargs"]['document_store'] = doc_store

        return serialised

    @classmethod
    def from_dict(cls, data):
        document_store = OpenSearchDocumentStore.from_dict(data['init_parameters']["document_store"])
        return cls(
            document_store=document_store,
            text_embedder_kwargs=data["init_parameters"]["text_embedder_kwargs"],
            bm25_retriever_kwargs=data["init_parameters"]["bm25_retriever_kwargs"],
            embedding_retriever_kwargs=data["init_parameters"]["embedding_retriever_kwargs"],
            document_joiner_kwargs=data["init_parameters"]["document_joiner_kwargs"],
            chat_prompt_builder_kwargs=data["init_parameters"]["chat_prompt_builder_kwargs"],
            generator_kwargs=data["init_parameters"]["generator_kwargs"],
            answer_builder_kwargs=data["init_parameters"]["answer_builder_kwargs"]
        )
