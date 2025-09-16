"""
TogetherAI Generator Example

This example demonstrates how to use the TogetherAIGenerator component with the Haystack framework.
It shows basic usage, advanced configuration, streaming, pipeline integration, and async operations.

Prerequisites:
- Set environment variables: TOGETHER_AI_API_KEY
- Install: pip install haystack-together_ai
"""

from haystack.utils import Secret

from haystack_integrations.components.generators.together_ai.generator import TogetherAIGenerator

generator = TogetherAIGenerator(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    api_key=Secret.from_env_var("TOGETHER_AI_API_KEY"),
    generation_kwargs={"max_tokens": 100, "temperature": 0.7, "top_p": 0.9},
)

system_prompt = "You are a helpful assistant. Provide clear, direct answers."

questions = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "List benefits of renewable energy.",
]

for question in questions:
    print(f"\nQuestion: {question}")
    result = generator.run(prompt=question, system_prompt=system_prompt)

    response = result["replies"][0]
    metadata = result["meta"][0]

    print(f"Answer: {response}")
    print(f"Tokens used: {metadata.get('usage', {}).get('total_tokens', 'N/A')}")
    print("-" * 60)
