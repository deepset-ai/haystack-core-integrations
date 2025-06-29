# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import component, logging
from haystack.dataclasses import ChatMessage, StreamingCallbackT

from .chat.chat_generator import WatsonxChatGenerator

logger = logging.getLogger(__name__)


@component
class WatsonxGenerator(WatsonxChatGenerator):
    """
    Enables text completions using IBM's watsonx.ai foundation models.

    This component extends WatsonxChatGenerator to provide the standard Generator interface
    that works with prompt strings instead of ChatMessage objects. It inherits all the
    functionality from WatsonxChatGenerator while adapting the input/output format.

    The generator works with IBM's foundation models including:
    - granite-13b-chat-v2
    - llama-2-70b-chat
    - llama-3-70b-instruct
    - Other watsonx.ai chat models

    You can customize the generation behavior by passing parameters to the
    watsonx.ai API through the `generation_kwargs` argument. These parameters
    are passed directly to the watsonx.ai inference endpoint.

    For details on watsonx.ai API parameters, see
    [IBM watsonx.ai documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-parameters.html).

    ### Usage example

    ```python
    from haystack_integrations.components.generators.watsonx.generator import WatsonxGenerator
    from haystack.utils import Secret

    generator = WatsonxGenerator(
        api_key=Secret.from_env_var('WATSONX_API_KEY'),
        model='ibm/granite-13b-chat-v2',
        project_id=Secret.from_env_var('WATSONX_PROJECT_ID'),
    )
    
    response = generator.run(
        prompt="Explain quantum computing in simple terms",
        system_prompt="You are a helpful physics teacher."
    )
    print(response)
    ```
    Output:
    ```
    {
        'replies': ['Quantum computing uses quantum-mechanical phenomena like....'],
        'meta': [{'model': 'ibm/granite-13b-chat-v2', 'project_id': 'your-project-id',
                  'usage': {'prompt_tokens': 12, 'completion_tokens': 45, 'total_tokens': 57}}]
    }
    ```

    The component also supports streaming responses and function calling through
    watsonx.ai's tools parameter.
    """

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text completions synchronously.

        :param prompt: The input prompt for text generation
        :param system_prompt: Optional system prompt to set context/instructions
        :param streaming_callback: Optional callback function for streaming responses
        :param generation_kwargs: Additional generation parameters
        :return: Dictionary with 'replies' (List[str]) and 'meta' (List[Dict[str, Any]])
        """
        messages = self._prepare_messages(prompt, system_prompt)
        
        chat_response = WatsonxChatGenerator.run(
            self,
            messages=messages,
            generation_kwargs=generation_kwargs,
            streaming_callback=streaming_callback
        )
        
        return self._convert_chat_response_to_generator_format(chat_response)

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    async def run_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text completions asynchronously.

        :param prompt: The input prompt for text generation
        :param system_prompt: Optional system prompt to set context/instructions
        :param streaming_callback: Optional callback function for streaming responses
        :param generation_kwargs: Additional generation parameters
        :return: Dictionary with 'replies' (List[str]) and 'meta' (List[Dict[str, Any]])
        """
        messages = self._prepare_messages(prompt, system_prompt)
        
        chat_response = await WatsonxChatGenerator.run_async(
            self,
            messages=messages,
            generation_kwargs=generation_kwargs,
            streaming_callback=streaming_callback
        )
        
        return self._convert_chat_response_to_generator_format(chat_response)

    def _prepare_messages(self, prompt: str, system_prompt: Optional[str] = None) -> List[ChatMessage]:
        """
        Convert prompt and system_prompt to ChatMessage format.
        
        :param prompt: The user prompt
        :param system_prompt: Optional system prompt
        :return: List of ChatMessage objects
        """
        messages = []
        
        if system_prompt:
            messages.append(ChatMessage.from_system(system_prompt))
        
        messages.append(ChatMessage.from_user(prompt))
        
        return messages

    def _convert_chat_response_to_generator_format(self, chat_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert ChatGenerator response format to Generator format.
        
        :param chat_response: Response from WatsonxChatGenerator
        :return: Response in Generator format with replies and meta lists
        """
        replies = []
        meta = []
        
        for chat_message in chat_response.get("replies", []):
            text_content = chat_message.text if hasattr(chat_message, 'text') else str(chat_message)
            replies.append(text_content)
            
            message_meta = getattr(chat_message, 'meta', {}) or {}
            meta.append(message_meta)
        
        return {"replies": replies, "meta": meta}