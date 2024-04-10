import logging
from typing import Any, Callable, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

with LazyImport(message="Run 'pip install cohere'") as cohere_import:
    import cohere
logger = logging.getLogger(__name__)


@component
class CohereChatGenerator:
    """
    Enables text generation using Cohere's chat endpoint.

    This component is designed to inference Cohere's chat models.

    Users can pass any text generation parameters valid for the `cohere.Client,chat` method
    directly to this component via the `**generation_kwargs` parameter in __init__ or the `**generation_kwargs`
    parameter in `run` method.

    Invocations are made using 'cohere' package.
    See [Cohere API](https://docs.cohere.com/reference/chat) for more details.

    Example usage:
    ```python
    from haystack_integrations.components.generators.cohere import CohereChatGenerator

    component = CohereChatGenerator(api_key=Secret.from_token("test-api-key"))
    response = component.run(chat_messages)

    assert response["replies"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
        model: str = "command",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the CohereChatGenerator instance.

        :param api_key: the API key for the Cohere API.
        :param model: The name of the model to use. Available models are: [command, command-light, command-nightly,
            command-nightly-light].
        :param streaming_callback: a callback function to be called with the streaming response.
        :param api_base_url: the base URL of the Cohere API.
        :param generation_kwargs: additional model parameters. These will be used during generation. Refer to
            https://docs.cohere.com/reference/chat for more details.
            Some of the parameters are:
            - 'chat_history': A list of previous messages between the user and the model, meant to give the model
               conversational context for responding to the user's message.
            - 'preamble_override': When specified, the default Cohere preamble will be replaced with the provided one.
            - 'conversation_id': An alternative to chat_history. Previous conversations can be resumed by providing
               the conversation's identifier. The contents of message and the model's response will be stored
               as part of this conversation.If a conversation with this id does not already exist,
               a new conversation will be created.
            - 'prompt_truncation': Defaults to AUTO when connectors are specified and OFF in all other cases.
               Dictates how the prompt will be constructed.
            - 'connectors': Accepts {"id": "web-search"}, and/or the "id" for a custom connector, if you've created one.
                When specified, the model's reply will be enriched with information found by
                quering each of the connectors (RAG).
            - 'documents': A list of relevant documents that the model can use to enrich its reply.
            - 'search_queries_only': Defaults to false. When true, the response will only contain a
               list of generated search queries, but no search will take place, and no reply from the model to the
               user's message will be generated.
            - 'citation_quality': Defaults to "accurate". Dictates the approach taken to generating citations
                as part of the RAG flow by allowing the user to specify whether they want
                "accurate" results or "fast" results.
            - 'temperature': A non-negative float that tunes the degree of randomness in generation. Lower temperatures
                mean less random generations.
        """
        cohere_import.check()

        if not api_base_url:
            api_base_url = cohere.COHERE_API_URL
        if generation_kwargs is None:
            generation_kwargs = {}
        self.api_key = api_key
        self.model = model
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url
        self.generation_kwargs = generation_kwargs
        self.model_parameters = kwargs
        self.client = cohere.Client(
            api_key=self.api_key.resolve_value(), api_url=self.api_base_url, client_name="haystack"
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
                Dictionary with serialized data.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            api_key=self.api_key.to_dict(),
            generation_kwargs=self.generation_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohereChatGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
               Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        deserialize_secrets_inplace(init_params, ["api_key"])
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    def _message_to_dict(self, message: ChatMessage) -> Dict[str, str]:
        role = "User" if message.role == ChatRole.USER else "Chatbot"
        chat_message = {"user_name": role, "text": message.content}
        return chat_message

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param messages: list of `ChatMessage` instances representing the input messages.
        :param generation_kwargs: additional keyword arguments for text generation. These parameters will
            potentially override the parameters passed in the __init__ method.
            For more details on the parameters supported by the Cohere API, refer to the
            Cohere [documentation](https://docs.cohere.com/reference/chat).
        :returns: A dictionary with the following keys:
            - `replies`: a list of `ChatMessage` instances representing the generated responses.
        """
        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        chat_history = [self._message_to_dict(m) for m in messages[:-1]]
        response = self.client.chat(
            message=messages[-1].content,
            model=self.model,
            stream=self.streaming_callback is not None,
            chat_history=chat_history,
            **generation_kwargs,
        )
        if self.streaming_callback:
            for chunk in response:
                if chunk.event_type == "text-generation":
                    stream_chunk = self._build_chunk(chunk)
                    self.streaming_callback(stream_chunk)
            chat_message = ChatMessage.from_assistant(content=response.texts)
            chat_message.meta.update(
                {
                    "model": self.model,
                    "usage": response.token_count,
                    "index": 0,
                    "finish_reason": response.finish_reason,
                    "documents": response.documents,
                    "citations": response.citations,
                }
            )
        else:
            chat_message = self._build_message(response)
        return {"replies": [chat_message]}

    def _build_chunk(self, chunk) -> StreamingChunk:
        """
        Converts the response from the Cohere API to a StreamingChunk.
        :param chunk: The chunk returned by the OpenAI API.
        :param choice: The choice returned by the OpenAI API.
        :returns: The StreamingChunk.
        """
        chat_message = StreamingChunk(content=chunk.text, meta={"index": chunk.index, "event_type": chunk.event_type})
        return chat_message

    def _build_message(self, cohere_response):
        """
        Converts the non-streaming response from the Cohere API to a ChatMessage.
        :param cohere_response: The completion returned by the Cohere API.
        :returns: The ChatMessage.
        """
        content = cohere_response.text
        message = ChatMessage.from_assistant(content=content)
        message.meta.update(
            {
                "model": self.model,
                "usage": cohere_response.token_count,
                "index": 0,
                "finish_reason": None,
                "documents": cohere_response.documents,
                "citations": cohere_response.citations,
            }
        )
        return message
