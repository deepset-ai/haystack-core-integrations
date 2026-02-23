# dspy-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/dspy-haystack.svg)](https://pypi.org/project/dspy-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dspy-haystack.svg)](https://pypi.org/project/dspy-haystack)

An integration between [DSPy](https://github.com/stanfordnlp/dspy) and [Haystack](https://haystack.deepset.ai/).

DSPy is a framework for algorithmically optimizing prompts for Language Models by applying classical machine learning concepts (training data, evaluation metrics, optimization).

This integration provides:
- **DSPyChatGenerator** — a Haystack ChatGenerator component that uses DSPy signatures and modules for structured generation

## Installation

```bash
pip install dspy-haystack
```

## Quick Start

### DSPyChatGenerator

A Haystack chat generator that uses DSPy signatures for structured generation with built-in reasoning patterns (Chain-of-Thought, Predict, ReAct).

```python
from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.dspy import DSPyChatGenerator
import dspy

# Define a DSPy signature
class QASignature(dspy.Signature):
    """Answer questions accurately and concisely."""
    question = dspy.InputField(desc="The user's question")
    answer = dspy.OutputField(desc="A clear, concise answer")

# Create the generator
generator = DSPyChatGenerator(
    model="openai/gpt-5-mini",
    signature=QASignature,
    module_type="ChainOfThought"
)

# Use in pipeline
pipeline = Pipeline()
pipeline.add_component("llm", generator)

messages = [ChatMessage.from_user("What is the capital of France?")]
result = pipeline.run({"llm": {"messages": messages}})
print(result["llm"]["replies"][0].text)
```

You can also use string signatures for quick prototyping:

```python
generator = DSPyChatGenerator(
    model="openai/gpt-5-mini",
    signature="question -> answer",
    module_type="Predict"
)
```

## License

`dspy-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
