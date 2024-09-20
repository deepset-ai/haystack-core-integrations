# To run this example, you will need to set a `ANTHROPIC_API_KEY` environment variable.

import time

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils import Secret

from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator

# Advanced: We can also cache the HTTP GET requests for the HTML content to avoid repeating requests
# that fetched the same content.
# This type of caching requires requests_cache library to be installed
# Uncomment the following two lines to caching the HTTP requests

# import requests_cache
# requests_cache.install_cache("anthropic_demo")

ENABLE_PROMPT_CACHING = True  # Toggle this to enable or disable prompt caching


def create_streaming_callback():
    first_token_time = None

    def stream_callback(chunk: StreamingChunk) -> None:
        nonlocal first_token_time
        if first_token_time is None:
            first_token_time = time.time()
        print(chunk.content, flush=True, end="")

    return stream_callback, lambda: first_token_time


# Until prompt caching graduates from beta, we need to set the anthropic-beta header
generation_kwargs = {"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}} if ENABLE_PROMPT_CACHING else {}

claude_llm = AnthropicChatGenerator(
    api_key=Secret.from_env_var("ANTHROPIC_API_KEY"), generation_kwargs=generation_kwargs
)

pipe = Pipeline()
pipe.add_component("fetcher", LinkContentFetcher())
pipe.add_component("converter", HTMLToDocument())
pipe.add_component("prompt_builder", ChatPromptBuilder(variables=["documents"]))
pipe.add_component("llm", claude_llm)
pipe.connect("fetcher", "converter")
pipe.connect("converter", "prompt_builder")
pipe.connect("prompt_builder.prompt", "llm.messages")

system_message = ChatMessage.from_system(
    "Claude is an AI assistant that answers questions based on the given documents.\n"
    "Here are the documents:\n"
    "{% for d in documents %} \n"
    "    {{d.content}} \n"
    "{% endfor %}"
)

if ENABLE_PROMPT_CACHING:
    system_message.meta["cache_control"] = {"type": "ephemeral"}

questions = [
    "What's this paper about?",
    "What's the main contribution of this paper?",
    "How can findings from this paper be applied to real-world problems?",
]

for question in questions:
    print(f"Question: {question}")
    start_time = time.time()
    streaming_callback, get_first_token_time = create_streaming_callback()
    # reset LLM streaming callback to initialize new timers in streaming mode
    claude_llm.streaming_callback = streaming_callback

    result = pipe.run(
        data={
            "fetcher": {"urls": ["https://ar5iv.labs.arxiv.org/html/2310.04406"]},
            "prompt_builder": {"template": [system_message, ChatMessage.from_user(f"Answer the question: {question}")]},
        }
    )

    end_time = time.time()
    total_time = end_time - start_time
    time_to_first_token = get_first_token_time() - start_time
    print("\n" + "-" * 100)
    print(f"Total generation time: {total_time:.2f} seconds")
    print(f"Time to first token: {time_to_first_token:.2f} seconds")
    # first time we create a prompt cache usage key 'cache_creation_input_tokens' will have a value of the number of
    # tokens used to create the prompt cache
    # on first subsequent cache hit we'll see a usage key 'cache_read_input_tokens' having a value of the number of
    # tokens read from the cache
    token_stats = result["llm"]["replies"][0].meta.get("usage")
    if token_stats.get("cache_creation_input_tokens", 0) > 0:
        print("Cache created! ", end="")
    elif token_stats.get("cache_read_input_tokens", 0) > 0:
        print("Cache hit! ", end="")
    else:
        print("Cache not used, something is wrong with the prompt caching setup. ", end="")
    print(f"Cache usage details: {token_stats}")
    print("\n" + "=" * 100)
