# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# A minimal RAG pipeline using Eden AI as the (EU-hosted, multi-provider) chat backend.
#
# Prerequisites:
#   pip install edenai-haystack
#   export EDENAI_API_KEY="<your Eden AI API key>"

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.generators.edenai import EdenAIChatGenerator

# 1. Tiny in-memory knowledge base
document_store = InMemoryDocumentStore()
document_store.write_documents(
    [
        Document(content="Eden AI is a unified API that routes to 500+ AI models with EU data residency."),
        Document(content="Haystack is an open-source framework by deepset to build LLM and RAG applications."),
        Document(content="The capital of France is Paris."),
    ]
)

# 2. Build the RAG pipeline
template = [
    ChatMessage.from_user(
        "Given these documents, answer the question.\n"
        "Documents:\n{% for doc in documents %}{{ doc.content }}\n{% endfor %}\n"
        "Question: {{ question }}\nAnswer:"
    )
]

pipeline = Pipeline()
pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template, required_variables=["question"]))
# Swap the model for any Eden AI `provider/model`, e.g. "mistral/mistral-large-latest" for an EU-hosted model.
pipeline.add_component("llm", EdenAIChatGenerator(model="openai/gpt-4o-mini"))

pipeline.connect("retriever", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "llm.messages")

question = "What is Eden AI?"
result = pipeline.run({"retriever": {"query": question}, "prompt_builder": {"question": question}})
print(result["llm"]["replies"][0].text)
