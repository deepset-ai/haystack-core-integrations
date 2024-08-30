# To run this example, you will need to set a `ANTHROPIC_API_KEY` environment variable.

import time

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils import Secret

from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator

enable_prompt_caching = True

msg = ChatMessage.from_system(
    "You are a prompt expert who answers questions based on the given documents.\n"
    "Here are the documents:\n"
    "{% for d in documents %} \n"
    "    {{d.content}} \n"
    "{% endfor %}"
)


def measure_and_print_streaming_chunk():
    first_token_time = None

    def stream_callback(chunk: StreamingChunk) -> None:
        nonlocal first_token_time
        if first_token_time is None:
            first_token_time = time.time()
        print(chunk.content, flush=True, end="")

    return stream_callback, lambda: first_token_time


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
if enable_prompt_caching:
    final_prompt_msg.meta["cache_control"] = {"type": "ephemeral"}

generation_kwargs = {"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}} if enable_prompt_caching else {}
claude_llm = AnthropicChatGenerator(
    api_key=Secret.from_env_var("ANTHROPIC_API_KEY"),
    generation_kwargs=generation_kwargs,
)

# Build QA pipeline
qa_pipeline = Pipeline()
qa_pipeline.add_component("llm", claude_llm)

questions = [
    "What's this paper about?",
    "What's the main contribution of this paper?",
    "How can findings from this paper be applied to real-world problems?",
]

# Answer the questions using prompt caching (i.e. the entire document is cached, we run the question against it)
for question in questions:
    print("Question: " + question)
    start_time = time.time()
    streaming_callback, get_first_token_time = measure_and_print_streaming_chunk()
    claude_llm.streaming_callback = streaming_callback

    result = qa_pipeline.run(
        data={
            "llm": {
                "messages": [
                    final_prompt_msg,
                    ChatMessage.from_user("Given these documents, answer the question:" + question),
                ]
            },
        }
    )

    end_time = time.time()
    total_time = end_time - start_time
    time_to_first_token = get_first_token_time() - start_time

    print(f"\nTotal time: {total_time:.2f} seconds")
    print(f"Time to first token: {time_to_first_token:.2f} seconds")
    print(f"Cache usage: {result['llm']['replies'][0].meta.get('usage')}")
    print("\n" + "=" * 50)
