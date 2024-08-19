# To run this example, you will need to set a `ANTHROPIC_API_KEY` environment variable.

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator

msg = ChatMessage.from_system(
    "You are a prompt expert who answers questions based on the given documents.\n"
    "Here are the documents:\n"
    "{% for d in documents %} \n"
    "    {{d.content}} \n"
    "{% endfor %}"
)

fetch_pipeline = Pipeline()
fetch_pipeline.add_component("fetcher", LinkContentFetcher())
fetch_pipeline.add_component("converter", HTMLToDocument())
fetch_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=[msg], variables=["documents"]))

fetch_pipeline.connect("fetcher", "converter")
fetch_pipeline.connect("converter", "prompt_builder")

result = fetch_pipeline.run(
    data={
        "fetcher": {"urls": ["https://ar5iv.labs.arxiv.org/html/2310.04406"]},
    }
)

# Now we have our document fetched as a ChatMessage
final_prompt_msg = result["prompt_builder"]["prompt"][0]

# We add a cache control header to the prompt message
final_prompt_msg.meta["cache_control"] = {"type": "ephemeral"}


# Build QA pipeline
qa_pipeline = Pipeline()
qa_pipeline.add_component("llm", AnthropicChatGenerator(
    api_key=Secret.from_env_var("ANTHROPIC_API_KEY"),
    streaming_callback=print_streaming_chunk,
    generation_kwargs={"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}},
))

questions = ["Why is Monte-Carlo Tree Search used in LATS",
             "Summarize LATS selection, expansion, evaluation, simulation, backpropagation, and reflection"]

# Answer the questions using prompt caching (i.e. the entire document is cached, we run the question against it)
for question in questions:
    print("Question: " + question)
    qa_pipeline.run(
        data={
            "llm": {"messages": [final_prompt_msg,
                                 ChatMessage.from_user("Given these documents, answer the question:" + question)]},
        }
    )
    print("\n")

