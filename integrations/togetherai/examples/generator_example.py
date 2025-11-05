# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# This example demonstrates how to use the TogetherAIGenerator component
# To run this example, you will need to
# set `TOGETHER_API_KEY` environment variable


from haystack.utils import Secret

from haystack_integrations.components.generators.togetherai.generator import TogetherAIGenerator

generator = TogetherAIGenerator(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    api_key=Secret.from_env_var("TOGETHER_API_KEY"),
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
