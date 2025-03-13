# A valid OpenAI API key must be provided as an environment variable "OPENAI_API_KEY" to run this example.

import os
from getpass import getpass

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key:")


from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import OpenAITextEmbedder, OpenAIDocumentEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.generators import OpenAIGenerator
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import AnswerBuilder
from haystack_integrations.components.evaluators.ragas import RagasEvaluator

from ragas.llms import HaystackLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness


dataset = [
    "OpenAI is one of the most recognized names in the large language model space, known for its GPT series of models. These models excel at generating human-like text and performing tasks like creative writing, answering questions, and summarizing content. GPT-4, their latest release, has set benchmarks in understanding context and delivering detailed responses.",
    "Anthropic is well-known for its Claude series of language models, designed with a strong focus on safety and ethical AI behavior. Claude is particularly praised for its ability to follow complex instructions and generate text that aligns closely with user intent.",
    "DeepMind, a division of Google, is recognized for its cutting-edge Gemini models, which are integrated into various Google products like Bard and Workspace tools. These models are renowned for their conversational abilities and their capacity to handle complex, multi-turn dialogues.",
    "Meta AI is best known for its LLaMA (Large Language Model Meta AI) series, which has been made open-source for researchers and developers. LLaMA models are praised for their ability to support innovation and experimentation due to their accessibility and strong performance.",
    "Meta AI with it's LLaMA models aims to democratize AI development by making high-quality models available for free, fostering collaboration across industries. Their open-source approach has been a game-changer for researchers without access to expensive resources.",
    "Microsoft’s Azure AI platform is famous for integrating OpenAI’s GPT models, enabling businesses to use these advanced models in a scalable and secure cloud environment. Azure AI powers applications like Copilot in Office 365, helping users draft emails, generate summaries, and more.",
    "Amazon’s Bedrock platform is recognized for providing access to various language models, including its own models and third-party ones like Anthropic’s Claude and AI21’s Jurassic. Bedrock is especially valued for its flexibility, allowing users to choose models based on their specific needs.",
    "Cohere is well-known for its language models tailored for business use, excelling in tasks like search, summarization, and customer support. Their models are recognized for being efficient, cost-effective, and easy to integrate into workflows.",
    "AI21 Labs is famous for its Jurassic series of language models, which are highly versatile and capable of handling tasks like content creation and code generation. The Jurassic models stand out for their natural language understanding and ability to generate detailed and coherent responses.",
    "In the rapidly advancing field of artificial intelligence, several companies have made significant contributions with their large language models. Notable players include OpenAI, known for its GPT Series (including GPT-4); Anthropic, which offers the Claude Series; Google DeepMind with its Gemini Models; Meta AI, recognized for its LLaMA Series; Microsoft Azure AI, which integrates OpenAI’s GPT Models; Amazon AWS (Bedrock), providing access to various models including Claude (Anthropic) and Jurassic (AI21 Labs); Cohere, which offers its own models tailored for business use; and AI21 Labs, known for its Jurassic Series. These companies are shaping the landscape of AI by providing powerful models with diverse capabilities.",
]

# Initialize components for RAG pipeline
document_store = InMemoryDocumentStore()
docs = [Document(content=doc) for doc in dataset]

document_embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
text_embedder = OpenAITextEmbedder(model="text-embedding-3-small")

docs_with_embeddings = document_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

retriever = InMemoryEmbeddingRetriever(document_store, top_k=2)

template = [
    ChatMessage.from_user(
        """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
    )
]

prompt_builder = ChatPromptBuilder(template=template)
chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")

# Setting the RagasEvaluator Component
llm = OpenAIGenerator(model="gpt-4o-mini")
evaluator_llm = HaystackLLMWrapper(llm)

ragas_evaluator = RagasEvaluator(
    ragas_metrics=[AnswerRelevancy(), ContextPrecision(), Faithfulness()], evaluator_llm=evaluator_llm
)

# Creating the Pipeline
rag_pipeline = Pipeline()

# Adding the components
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", chat_generator)
rag_pipeline.add_component("answer_builder", AnswerBuilder())
rag_pipeline.add_component("ragas_evaluator", ragas_evaluator)

# Connecting the components
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder")
rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("retriever", "answer_builder.documents")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("retriever", "answer_builder.documents")
rag_pipeline.connect("retriever", "ragas_evaluator.documents")
rag_pipeline.connect("llm.replies", "ragas_evaluator.response")

# Run the pipeline
question = "What makes Meta AI’s LLaMA models stand out?"

reference = "Meta AI’s LLaMA models stand out for being open-source, supporting innovation and experimentation due to their accessibility and strong performance."


result = rag_pipeline.run(
    {
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question},
        "answer_builder": {"query": question},
        "ragas_evaluator": {"query": question, "reference": reference},
        # Each metric expects a specific set of parameters as input. Refer to the
        # Ragas class' documentation for more details.
    }
)

print(result['answer_builder']['answers'][0].data, '\n')
print(result['ragas_evaluator']['result'])
