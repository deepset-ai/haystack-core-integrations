import dspy
from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.dspy import DSPyChatGenerator


class QASignature(dspy.Signature):
    """Answer questions accurately and concisely."""

    question = dspy.InputField(desc="The user's question")
    answer = dspy.OutputField(desc="A clear, concise answer")


def basic_qa_example():
    """Simple question-answering with Chain-of-Thought reasoning."""

    generator = DSPyChatGenerator(
        model="openai/gpt-4o-mini",
        signature=QASignature,
        module_type="ChainOfThought",
        output_field="answer",
    )

    pipeline = Pipeline()
    pipeline.add_component("llm", generator)

    messages = [ChatMessage.from_user("What causes rainbows to appear?")]
    result = pipeline.run({"llm": {"messages": messages}})

    print(f"Question: {messages[0].text}")
    print(f"Answer: {result['llm']['replies'][0].text}\n")


def string_signature_example():
    """Using a simple string signature instead of a class."""
    generator = DSPyChatGenerator(
        model="openai/gpt-4o-mini",
        signature="question -> answer",
        module_type="Predict",
        output_field="answer",
    )

    pipeline = Pipeline()
    pipeline.add_component("llm", generator)

    messages = [ChatMessage.from_user("What is the capital of Japan?")]
    result = pipeline.run({"llm": {"messages": messages}})

    print(f"Question: {messages[0].text}")
    print(f"Answer: {result['llm']['replies'][0].text}\n")


if __name__ == "__main__":
    basic_qa_example()
    string_signature_example()