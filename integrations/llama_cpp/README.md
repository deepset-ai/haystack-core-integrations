# llama-cpp-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/llama-cpp-haystack.svg)](https://pypi.org/project/llama-cpp-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llama-cpp-haystack.svg)](https://pypi.org/project/llama-cpp-haystack)

-----

Custom component for [Haystack](https://github.com/deepset-ai/haystack) (2.x) for running LLMs using the [Llama.cpp](https://github.com/ggerganov/llama.cpp) LLM framework. This implementation leverages the [Python Bindings for llama.cpp](https://github.com/abetlen/llama-cpp-python).

**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [License](#license)

## Installation

```bash
pip install llama-cpp-haystack
```

The default install behaviour is to build `llama.cpp` for CPU only on Linux and Windows and use Metal on MacOS.

To install using the other backends, first install [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) using the instructions on their [installation documentation](https://github.com/abetlen/llama-cpp-python#installation) and then install [llama-cpp-haystack](https://github.com/deepset-ai/haystack-core-integrations/tree/main/integrations/llama_cpp).


For example, to use `llama-cpp-haystack` with the cuBLAS backend:

```bash
export LLAMA_CUBLAS=1
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
pip install llama-cpp-haystack
```

## Usage

You can utilize the `LlamaCppGenerator` to load models quantized using llama.cpp (GGUF) for text generation.

Information about the supported models and model parameters can be found on the llama.cpp [documentation](https://llama-cpp-python.readthedocs.io/en/latest).

The GGUF versions of popular models can be downloaded from [HuggingFace](https://huggingface.co/models?library=gguf).

### Passing additional model parameters

The `model_path`, `n_ctx`, `n_batch` arguments have been exposed for convenience and can be directly passed to the Generator during initialization as keyword arguments.  

The `model_kwargs` parameter can be used to pass additional arguments when initializing the model. In case of duplication, these kwargs override `model_path`, `n_ctx`, and `n_batch` init parameters.

See Llama.cpp's [model documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.__init__) for more information on the available model arguments.

For example, to offload the model to GPU during initialization:

```python
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator

generator = LlamaCppGenerator(
    model_path="/content/openchat-3.5-1210.Q3_K_S.gguf", 
    n_ctx=512,
    n_batch=128,
    model_kwargs={"n_gpu_layers": -1}
)
generator.warm_up()

input = "Who is the best American actor?"
prompt = f"GPT4 Correct User: {input} <|end_of_turn|> GPT4 Correct Assistant:"

result = generator.run(prompt, generation_kwargs={"max_tokens": 128})
generated_text = result["replies"][0]

print(generated_text)
```
### Passing generation parameters

The `generation_kwargs` parameter can be used to pass additional generation arguments like `max_tokens`, `temperature`, `top_k`, `top_p`, etc to the model during inference. 

See Llama.cpp's [`create_completion` documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_completion) for more information on the available generation arguments.

For example, to set the `max_tokens` and `temperature`:

```python
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator

generator = LlamaCppGenerator(
    model_path="/content/openchat-3.5-1210.Q3_K_S.gguf",
    n_ctx=512,
    n_batch=128,
    generation_kwargs={"max_tokens": 128, "temperature": 0.1},
)
generator.warm_up()

input = "Who is the best American actor?"
prompt = f"GPT4 Correct User: {input} <|end_of_turn|> GPT4 Correct Assistant:"

result = generator.run(prompt)
generated_text = result["replies"][0]

print(generated_text)
```
The `generation_kwargs` can also be passed to the `run` method of the generator directly:

```python
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator

generator = LlamaCppGenerator(
    model_path="/content/openchat-3.5-1210.Q3_K_S.gguf",
    n_ctx=512,
    n_batch=128,
)
generator.warm_up()

input = "Who is the best American actor?"
prompt = f"GPT4 Correct User: {input} <|end_of_turn|> GPT4 Correct Assistant:"

result = generator.run(
    prompt,
    generation_kwargs={"max_tokens": 128, "temperature": 0.1},
)
generated_text = result["replies"][0]

print(generated_text)
```

## Example

Below is the example Retrieval Augmented Generation pipeline that uses the [Simple Wikipedia](https://huggingface.co/datasets/pszemraj/simple_wikipedia) Dataset from HuggingFace. You can find more examples in the [`examples`](https://github.com/deepset-ai/haystack-core-integrations/tree/main/integrations/llama_cpp/examples) folder.


Load the dataset:

```python
# Install HuggingFace Datasets using "pip install datasets"
from datasets import load_dataset
from haystack import Document, Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores import InMemoryDocumentStore

# Import LlamaCppGenerator
from haystack_integrations.components.generators.llama_cpp import LlamaCppGenerator

# Load first 100 rows of the Simple Wikipedia Dataset from HuggingFace
dataset = load_dataset("pszemraj/simple_wikipedia", split="validation[:100]")

docs = [
    Document(
        content=doc["text"],
        meta={
            "title": doc["title"],
            "url": doc["url"],
        },
    )
    for doc in dataset
]
```

Index the documents to the `InMemoryDocumentStore` using the `SentenceTransformersDocumentEmbedder` and `DocumentWriter`:

```python
doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

# Indexing Pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=doc_embedder, name="DocEmbedder")
indexing_pipeline.add_component(instance=DocumentWriter(document_store=doc_store), name="DocWriter")
indexing_pipeline.connect(connect_from="DocEmbedder", connect_to="DocWriter")

indexing_pipeline.run({"DocEmbedder": {"documents": docs}})
```

Create the Retrieval Augmented Generation (RAG) pipeline and add the `LlamaCppGenerator` to it:

```python
# Prompt Template for the https://huggingface.co/openchat/openchat-3.5-1210 LLM
prompt_template = """GPT4 Correct User: Answer the question using the provided context.
Question: {{question}}
Context:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
<|end_of_turn|>
GPT4 Correct Assistant:
"""

rag_pipeline = Pipeline()

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

# Load the LLM using LlamaCppGenerator
model_path = "openchat-3.5-1210.Q3_K_S.gguf"
generator = LlamaCppGenerator(model_path=model_path, n_ctx=4096, n_batch=128)

rag_pipeline.add_component(
    instance=text_embedder,
    name="text_embedder",
)
rag_pipeline.add_component(instance=InMemoryEmbeddingRetriever(document_store=doc_store, top_k=3), name="retriever")
rag_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
rag_pipeline.add_component(instance=generator, name="llm")
rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

rag_pipeline.connect("text_embedder", "retriever")
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("retriever", "answer_builder.documents")
```

Run the pipeline:

```python
question = "Which year did the Joker movie release?"
result = rag_pipeline.run(
    {
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question},
        "llm": {"generation_kwargs": {"max_tokens": 128, "temperature": 0.1}},
        "answer_builder": {"query": question},
    }
)

generated_answer = result["answer_builder"]["answers"][0]
print(generated_answer.data)
# The Joker movie was released on October 4, 2019.
```

## License

`llama-cpp-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
