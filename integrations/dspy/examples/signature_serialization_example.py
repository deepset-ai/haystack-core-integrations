from typing import Literal

import dspy
import pydantic
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.dspy import DSPySignatureChatGenerator

class Source(pydantic.BaseModel):
    """A single cited source."""
    title: str
    url: str
    relevance: float


class StructuredAnswer(pydantic.BaseModel):
    """A rich answer with metadata."""
    summary: str
    confidence: Literal["low", "medium", "high"]
    sources: list[Source]
    key_facts: list[str]


class ResearchSignature(dspy.Signature):
    """Research a topic and return a structured, cited answer."""
    question: str = dspy.InputField(desc="The research question")
    context: str = dspy.InputField(desc="Background material or documents to ground the answer")
    answer: StructuredAnswer = dspy.OutputField(desc="A structured answer with sources and confidence")
    follow_up_questions: list[str] = dspy.OutputField(desc="Suggested follow-up questions for deeper research")


def print_signature_fields(sig):
    """Pretty-print input/output fields and their types."""
    print("  Input fields:")
    for name, field in sig.input_fields.items():
        annotation = field.annotation if hasattr(field, "annotation") else "str"
        print(f"    {name}: {annotation}")
    print("  Output fields:")
    for name, field in sig.output_fields.items():
        annotation = field.annotation if hasattr(field, "annotation") else "str"
        print(f"    {name}: {annotation}")


def main():
    generator = DSPySignatureChatGenerator(
        model="openai/gpt-5-mini",
        signature=ResearchSignature,
        module_type="ChainOfThought",
        output_field="answer",
        input_mapping={"question": "question", "context": "context"},
    )

    print("=== Original generator ===")
    print(f"  signature class: {generator.signature.__name__}")
    print_signature_fields(generator.signature)

    data = generator.to_dict()
    sig_value = data["init_parameters"]["signature"]
    print(f"\n=== Serialized ===")
    print(f"  signature value: {sig_value}")
    print(f"  all init params: {list(data['init_parameters'].keys())}")

    restored = DSPySignatureChatGenerator.from_dict(data)
    print(f"\n=== Restored generator ===")
    print(f"  signature class: {restored.signature.__name__}")
    print(f"  same class?    : {restored.signature is ResearchSignature}")
    print_signature_fields(restored.signature)

    messages = [ChatMessage.from_user("What are the main causes of coral reef bleaching?")]
    result = restored.run(
        messages=messages,
        context=(
            "Coral bleaching occurs when corals expel their symbiotic algae (zooxanthellae) "
            "due to stress. Major stressors include elevated sea surface temperatures, ocean "
            "acidification from increased CO2 absorption, pollution runoff, and overexposure "
            "to sunlight. The Great Barrier Reef experienced mass bleaching events in 2016, "
            "2017, 2020, and 2022, primarily driven by marine heatwaves."
        ),
    )
    print(f"\n=== Inference ===")
    print(f"  Question: {messages[0].text}")
    print(f"  Answer  : {result['replies'][0].text}")


if __name__ == "__main__":
    main()
