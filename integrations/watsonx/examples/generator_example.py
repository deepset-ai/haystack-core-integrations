# In order to run this example, you will need watsonx credentials.
import os

from haystack import Pipeline
from haystack.utils import Secret

from haystack_integrations.components.generators.watsonx.generator import WatsonxGenerator

# Basic Usage of WatsonxGenerator
generator = WatsonxGenerator(
    model="ibm/granite-13b-instruct-v2",
    api_key=Secret.from_env_var("WATSONX_API_KEY"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    generation_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "decoding_method": "sample",
        "concurrency_limit": 5,
    },
)
prompts = ["Who is the president of the USA?", "What is the tallest mountain?"]

for prompt in prompts:
    result = generator.run(prompt)


## Result
## Output will vary based on model behavior
## Who is the president of the USA? → trump
## What is the tallest mountain? → The tallest mountain is Mount Everest, which is 8,848 meters (29,029 feet) tall.

# Initialize with streaming enabled
streaming_generator = WatsonxGenerator(
    api_key=Secret.from_token(os.environ["WATSONX_API_KEY"]),
    model="ibm/granite-13b-instruct-v2",
    project_id=os.environ["WATSONX_PROJECT_ID"],
    generation_kwargs={"max_new_tokens": 450, "temperature": 0.9, "top_p": 0.9, "repetition_penalty": 1.2},
)

# Create pipeline
streaming_pipeline = Pipeline()
streaming_pipeline.add_component("generator", streaming_generator)

# Run with streaming
prompt = "Write a short story about an AI assistant."
result = streaming_pipeline.run({"generator": {"prompt": prompt, "stream": True}})

# Process streaming results


##Result for steaming generator
##Full response: The AI assistant is a friendly and helpful robot that can help you with many tasks, such as setting reminders,
## managing your calendar, and creating to-do lists. You can also use the AI assistant to search for information online, or just have fun conversations with it.
