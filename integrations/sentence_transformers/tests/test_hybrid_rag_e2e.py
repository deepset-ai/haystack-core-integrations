# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest
from haystack import Document, Pipeline
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners.document_joiner import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.components.rankers.sentence_transformers import SentenceTransformersSimilarityRanker

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY", None),
    reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
)
def test_hybrid_retrieval_rag_pipeline(tmp_path):
    document_store = InMemoryDocumentStore()

    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Mario and I live in the capital of Italy."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component(
        instance=SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL), name="document_embedder"
    )
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="document_writer")
    indexing_pipeline.connect("document_embedder", "document_writer")
    indexing_pipeline.run({"document_embedder": {"documents": documents}})
    assert document_store.count_documents() == len(documents)

    prompt_template = [
        ChatMessage.from_user(
            "Given these documents, answer the question.\n"
            "Documents:\n"
            "{% for doc in documents %}{{ doc.content }}\n{% endfor %}\n"
            "Question: {{ question }}\n"
            "Answer:"
        )
    ]

    rag_pipeline = Pipeline()
    rag_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="bm25_retriever")
    rag_pipeline.add_component(instance=SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL), name="text_embedder")
    rag_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=document_store), name="embedding_retriever"
    )
    rag_pipeline.add_component(instance=DocumentJoiner(), name="joiner")
    rag_pipeline.add_component(
        instance=SentenceTransformersSimilarityRanker(model=RANKER_MODEL, top_k=3), name="ranker"
    )
    rag_pipeline.add_component(
        instance=ChatPromptBuilder(template=prompt_template, required_variables="*"), name="prompt_builder"
    )
    rag_pipeline.add_component(instance=OpenAIChatGenerator(), name="llm")
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

    rag_pipeline.connect("bm25_retriever", "joiner")
    rag_pipeline.connect("text_embedder", "embedding_retriever")
    rag_pipeline.connect("embedding_retriever", "joiner")
    rag_pipeline.connect("joiner", "ranker")
    rag_pipeline.connect("ranker.documents", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("ranker.documents", "answer_builder.documents")

    # Serialize and deserialize the pipeline to make sure the migrated components round-trip correctly
    with open(tmp_path / "test_hybrid_rag_pipeline.json", "w") as f:
        json.dump(rag_pipeline.to_dict(), f)
    with open(tmp_path / "test_hybrid_rag_pipeline.json") as f:
        rag_pipeline = Pipeline.from_dict(json.load(f))

    questions = ["Who lives in Paris?", "Who lives in Rome?"]
    spywords = ["Jean", "Giorgio"]

    for question, spyword in zip(questions, spywords, strict=True):
        result = rag_pipeline.run(
            {
                "bm25_retriever": {"query": question},
                "text_embedder": {"text": question},
                "ranker": {"query": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
            }
        )

        answers = result["answer_builder"]["answers"]
        assert len(answers) == 1
        generated_answer = answers[0]
        assert spyword in generated_answer.data
        assert generated_answer.query == question
        assert generated_answer.documents
        assert generated_answer.meta
