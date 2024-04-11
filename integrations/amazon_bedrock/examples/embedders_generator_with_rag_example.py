# To run this example, you will need to
# 1) set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_DEFAULT_REGION` environment variables
# 2) enable access to the selected model in Amazon Bedrock
# Note: if you change the model, update the model-specific inference parameters.

from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.embedders.amazon_bedrock import (
    AmazonBedrockDocumentEmbedder,
    AmazonBedrockTextEmbedder,
)
from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockGenerator

generator_model_name = "amazon.titan-text-lite-v1"
embedder_model_name = "amazon.titan-embed-text-v1"

prompt_template = """
Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Given the context above, answer the question.
Write a full detailed answer.
Provide explanation of why the answer is relevant to the question.
If you cannot answer the question, output "I do not know".

Question: {{ question }}?
"""

docs = [
    Document(content="User ABC is using Amazon SageMaker to train ML models."),
    Document(content="User XYZ is using Amazon EC2 instances to train ML models."),
]


doc_embedder = AmazonBedrockDocumentEmbedder(model=embedder_model_name)
docs_with_embeddings = doc_embedder.run(docs)["documents"]

doc_store = InMemoryDocumentStore()
doc_store.write_documents(docs_with_embeddings)


pipe = Pipeline()
pipe.add_component("text_embedder", AmazonBedrockTextEmbedder(embedder_model_name))
pipe.add_component("retriever", InMemoryEmbeddingRetriever(doc_store, top_k=1))
pipe.add_component("prompt_builder", PromptBuilder(prompt_template))
pipe.add_component(
    "generator",
    AmazonBedrockGenerator(
        model=generator_model_name,
        # model-specific inference parameters
        generation_kwargs={
            "maxTokenCount": 1024,
            "temperature": 0.0,
        },
    ),
)
pipe.connect("text_embedder", "retriever")
pipe.connect("retriever", "prompt_builder")
pipe.connect("prompt_builder", "generator")


question = "Which user is using IaaS services for Machine Learning?"
results = pipe.run(
    {
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question},
    }
)
results["generator"]["replies"]
