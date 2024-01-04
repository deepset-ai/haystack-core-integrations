from haystack import Pipeline
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.components.builders.prompt_builder import PromptBuilder

from ollama_haystack import OllamaGenerator

template = """
Given the following information, answer the question.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: What's the official language of {{ country }}?
"""
pipe = Pipeline()
document_store = InMemoryDocumentStore()

pipe.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", OllamaGenerator(model='orca-mini'))
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

pipe.run({"prompt_builder": {"country": "Ghana"}})
