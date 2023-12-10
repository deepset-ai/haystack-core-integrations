import logging
import os
from typing import Any, Callable, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.components.generators.utils import deserialize_callback_handler, serialize_callback_handler
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install cohere'") as cohere_import:
    import cohere
logger = logging.getLogger(__name__)


class CohereChatGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "command",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        cohere_import.check()

        if not api_key:
            api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            error = "CohereChatGenerator needs an API key to run. Either provide it as init parameter or set the env var COHERE_API_KEY."  # noqa: E501
            raise ValueError(error)

        if not api_base_url:
            api_base_url = cohere.COHERE_API_URL
        if generation_kwargs is None:
            generation_kwargs = {}
        self.api_key = api_key
        self.model_name = model_name
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url
        self.generation_kwargs = generation_kwargs
        self.model_parameters = kwargs
        self.client = cohere.Client(api_key=self.api_key, api_url=self.api_base_url)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        :return: The serialized component as a dictionary.
        """
        callback_name = serialize_callback_handler(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model_name=self.model_name,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohereChatGenerator":
        """
        Deserialize this component from a dictionary.
        :param data: The dictionary representation of this component.
        :return: The deserialized component instance.
        """
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callback_handler(serialized_callback_handler)
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        message = [message.content for message in messages]
        response = self.client.chat(
            message=message[0], model=self.model_name, stream=self.streaming_callback is not None, **generation_kwargs
        )
        if self.streaming_callback:
            for chunk in response:
                if chunk.event_type == "text-generation":
                    stream_chunk = self._build_chunk(chunk)
                    self.streaming_callback(stream_chunk)
            chat_message = ChatMessage(content=response.texts, role=None, name=None)
            chat_message.metadata.update(
                {
                    "token_count": response.token_count,
                    "finish_reason": response.finish_reason,
                    "documents": response.documents,
                    "citations": response.citations,
                    "chat-history": response.chat_history,
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
        :return: The StreamingChunk.
        """
        # if chunk.event_type == "text-generation":
        chat_message = StreamingChunk(
            content=chunk.text, metadata={"index": chunk.index, "event_type": chunk.event_type}
        )
        return chat_message

    def _build_message(self, cohere_response):
        """
        Converts the non-streaming response from the Cohere API to a ChatMessage.
        :param cohere_response: The completion returned by the Cohere API.
        :return: The ChatMessage.
        """
        content = cohere_response.text
        message = ChatMessage(content=content, role=None, name=None)
        message.metadata.update(
            {
                "token_count": cohere_response.token_count,
                "meta": cohere_response.meta,
                "citations": cohere_response.citations,
                "documents": cohere_response.documents,
                "chat-history": cohere_response.chat_history,
            }
        )
        return message
