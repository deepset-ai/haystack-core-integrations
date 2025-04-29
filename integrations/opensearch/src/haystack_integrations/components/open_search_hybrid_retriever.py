# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import Pipeline, logging
from haystack.components.builders import ChatPromptBuilder, AnswerBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport

from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever
from haystack_integrations.components.retrievers.opensearch import OpenSearchEmbeddingRetriever

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
            test: str = "test",
            document_store,
            embedding_model: str,
            top_k,
            template:
    ):


        self.pipeline = self._create_pipeline()

    def _create_pipeline(self, embedding_model: str, top_k: int, template: str, document_store) -> Pipeline:
        text_embedder = SentenceTransformersTextEmbedder(model=embedding_model, progress_bar=False)
        embedding_retriever = OpenSearchEmbeddingRetriever(document_store, top_k=top_k)
        bm25_retriever = OpenSearchBM25Retriever(document_store, top_k=top_k)
        document_joiner = DocumentJoiner(join_mode="concatenate")

        default = [
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

        template = template if template else default

        hybrid_retrieval = Pipeline()
        hybrid_retrieval.add_component("text_embedder", text_embedder)
        hybrid_retrieval.add_component("embedding_retriever", embedding_retriever)
        hybrid_retrieval.add_component("bm25_retriever", bm25_retriever)
        hybrid_retrieval.add_component("document_joiner", document_joiner)
        hybrid_retrieval.add_component("prompt_builder", ChatPromptBuilder(
            template=template, required_variables=["question", "documents"])
        )
        hybrid_retrieval.add_component("llm", OpenAIChatGenerator())
        hybrid_retrieval.add_component("answer_builder", AnswerBuilder())
        hybrid_retrieval.connect("text_embedder", "embedding_retriever")
        hybrid_retrieval.connect("bm25_retriever", "document_joiner")
        hybrid_retrieval.connect("embedding_retriever", "document_joiner")
        hybrid_retrieval.connect("document_joiner.documents", "prompt_builder.documents")
        hybrid_retrieval.connect("prompt_builder", "llm")
        hybrid_retrieval.connect("llm.replies", "answer_builder.replies")

        return hybrid_retrieval