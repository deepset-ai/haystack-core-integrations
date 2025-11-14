import os
from typing import Any, Callable, Optional

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import StreamingChunk
from haystack.tools import (
    ToolsType,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
)
from haystack.utils import deserialize_callable, serialize_callable

from anthropic import AnthropicVertex, AsyncAnthropicVertex

from .chat_generator import AnthropicChatGenerator

logger = logging.getLogger(__name__)


@component
class AnthropicVertexChatGenerator(AnthropicChatGenerator):
    """

    Enables text generation using state-of-the-art Claude 3 LLMs via the Anthropic Vertex AI API.
    It supports models such as `Claude 3.5 Sonnet`, `Claude 3 Opus`, `Claude 3 Sonnet`, and `Claude 3 Haiku`,
    accessible through the Vertex AI API endpoint.

    To use AnthropicVertexChatGenerator, you must have a GCP project with Vertex AI enabled.
    Additionally, ensure that the desired Anthropic model is activated in the Vertex AI Model Garden.
    Before making requests, you may need to authenticate with GCP using `gcloud auth login`.
    For more details, refer to the [guide] (https://docs.anthropic.com/en/api/claude-on-vertex-ai).

    Any valid text generation parameters for the Anthropic messaging API can be passed to
    the AnthropicVertex API. Users can provide these parameters directly to the component via
    the `generation_kwargs` parameter in `__init__` or the `run` method.

    For more details on the parameters supported by the Anthropic API, refer to the
    Anthropic Message API [documentation](https://docs.anthropic.com/en/api/messages).

    ```python
    from haystack_integrations.components.generators.anthropic import AnthropicVertexChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]
    client = AnthropicVertexChatGenerator(
                model="claude-sonnet-4@20250514",
                project_id="your-project-id", region="your-region"
            )
    response = client.run(messages)
    print(response)

    >> {'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=
    >> "Natural Language Processing (NLP) is a field of artificial intelligence that
    >> focuses on enabling computers to understand, interpret, and generate human language. It involves developing
    >> techniques and algorithms to analyze and process text or speech data, allowing machines to comprehend and
    >> communicate in natural languages like English, Spanish, or Chinese.")],
    >> _name=None, _meta={'model': 'claude-sonnet-4@20250514', 'index': 0, 'finish_reason': 'end_turn',
    >> 'usage': {'input_tokens': 15, 'output_tokens': 64}})]}
    ```

    For more details on supported models and their capabilities, refer to the Anthropic
    [documentation](https://docs.anthropic.com/claude/docs/intro-to-claude).

    """

    def __init__(
        self,
        region: str,
        project_id: str,
        model: str = "claude-sonnet-4@20250514",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
        ignore_tools_thinking_messages: bool = True,
        tools: Optional[ToolsType] = None,
        *,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Creates an instance of AnthropicVertexChatGenerator.

        :param region: The region where the Anthropic model is deployed. Defaults to "us-central1".
        :param project_id: The GCP project ID where the Anthropic model is deployed.
        :param model: The name of the model to use.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param generation_kwargs: Other parameters to use for the model. These parameters are all sent directly to
            the AnthropicVertex endpoint. See Anthropic [documentation](https://docs.anthropic.com/claude/reference/messages_post)
            for more details.

            Supported generation_kwargs parameters are:
            - `system`: The system message to be passed to the model.
            - `max_tokens`: The maximum number of tokens to generate.
            - `metadata`: A dictionary of metadata to be passed to the model.
            - `stop_sequences`: A list of strings that the model should stop generating at.
            - `temperature`: The temperature to use for sampling.
            - `top_p`: The top_p value to use for nucleus sampling.
            - `top_k`: The top_k value to use for top-k sampling.
            - `extra_headers`: A dictionary of extra headers to be passed to the model (i.e. for beta features).
        :param ignore_tools_thinking_messages: Anthropic's approach to tools (function calling) resolution involves a
            "chain of thought" messages before returning the actual function names and parameters in a message. If
            `ignore_tools_thinking_messages` is `True`, the generator will drop so-called thinking messages when tool
            use is detected. See the Anthropic [tools](https://docs.anthropic.com/en/docs/tool-use#chain-of-thought-tool-use)
            for more details.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset, that the model can use.
            Each tool should have a unique name.
        :param timeout:
            Timeout for Anthropic client calls. If not set, it defaults to the default set by the Anthropic client.
        :param max_retries:
            Maximum number of retries to attempt for failed requests. If not set, it defaults to the default set by
            the Anthropic client.
        """
        _check_duplicate_tool_names(flatten_tools_or_toolsets(tools))
        self.region = region or os.environ.get("REGION")
        self.project_id = project_id or os.environ.get("PROJECT_ID")
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.streaming_callback = streaming_callback
        self.ignore_tools_thinking_messages = ignore_tools_thinking_messages
        self.tools = tools
        self.timeout = timeout
        self.max_retries = max_retries

        client_kwargs: dict[str, Any] = {"region": self.region, "project_id": self.project_id}
        # We do this since timeout=None is not the same as not setting it in Anthropic
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        # We do this since max_retries must be an int when passing to Anthropic
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries

        # mypy is not happy that we override the type of the clients
        self.client = AnthropicVertex(**client_kwargs)  # type: ignore[assignment]
        self.async_client = AsyncAnthropicVertex(**client_kwargs)  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None

        return default_to_dict(
            self,
            region=self.region,
            project_id=self.project_id,
            model=self.model,
            streaming_callback=callback_name,
            generation_kwargs=self.generation_kwargs,
            ignore_tools_thinking_messages=self.ignore_tools_thinking_messages,
            tools=serialize_tools_or_toolset(self.tools),
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnthropicVertexChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)

        return default_from_dict(cls, data)
