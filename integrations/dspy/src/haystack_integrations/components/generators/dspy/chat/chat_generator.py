from typing import Any, Callable, Dict, List, Optional, Type, Union

import dspy
from haystack import component
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils import Secret

from haystack_integrations.components.generators.dspy.generator import DSPyGenerator


@component
class DSPyChatGenerator(DSPyGenerator):
    """
    A Haystack chat generator component that uses DSPy signatures and modules
    for structured generation.

    Accepts and returns ``ChatMessage`` objects, making it compatible with
    Haystack chat pipelines.

    Usage example:

    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.dspy import DSPyChatGenerator
    import dspy

    class QASignature(dspy.Signature):
        question = dspy.InputField(desc="The user's question")
        answer = dspy.OutputField(desc="A clear, concise answer")

    generator = DSPyChatGenerator(
        model="openai/gpt-5-mini",
        signature=QASignature,
        module_type="ChainOfThought",
    )

    messages = [ChatMessage.from_user("What is the capital of France?")]
    result = generator.run(messages=messages)
    print(result["replies"][0].text)
    ```
    """

    def __init__(
        self,
        signature: Union[str, Type[dspy.Signature]],
        model: str = "openai/gpt-5-mini",
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        module_type: str = "ChainOfThought",
        output_field: str = "answer",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        streaming_callback: Optional[Callable] = None,
    ):
        """
        Initialize the DSPyChatGenerator.

        :param signature: DSPy signature defining I/O structure. Can be a string
            like ``"question -> answer"`` or a ``dspy.Signature`` subclass.
        :param model: Model identifier (e.g. ``"openai/gpt-5-mini"``).
        :param api_key: API key for the LLM provider.
        :param module_type: DSPy module type: ``"Predict"``, ``"ChainOfThought"``, or ``"ReAct"``.
        :param output_field: Which signature output field to use as the reply.
        :param generation_kwargs: Additional generation parameters (temperature, max_tokens, etc.).
        :param input_mapping: Maps DSPy signature input field names to run kwarg names.
        :param streaming_callback: Callback for streaming responses.
        """
        DSPyGenerator.__init__(
            self,
            signature=signature,
            model=model,
            api_key=api_key,
            module_type=module_type,
            output_field=output_field,
            generation_kwargs=generation_kwargs,
            input_mapping=input_mapping,
            streaming_callback=streaming_callback,
        )

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the DSPy module on the given messages.

        :param messages: List of chat messages. The last user message is used as input.
        :param generation_kwargs: Optional runtime generation parameters.
        :param kwargs: Additional keyword arguments mapped to signature input fields.
        :returns: A dictionary with ``replies`` (list of ChatMessage) and ``meta`` (list of dicts).
        """
        if not messages:
            msg = "The 'messages' parameter cannot be empty."
            raise ValueError(msg)

        prompt = self._extract_last_user_message(messages)
        result = DSPyGenerator.run(self, prompt=prompt, generation_kwargs=generation_kwargs, **kwargs)

        replies = [ChatMessage.from_assistant(text=text) for text in result["replies"]]

        return {"replies": replies, "meta": result["meta"]}

    @component.output_types(replies=List[ChatMessage])
    async def run_async(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Asynchronously run the DSPy module on the given messages.

        Uses DSPy's native ``acall`` for true async I/O.

        :param messages: List of chat messages. The last user message is used as input.
        :param generation_kwargs: Optional runtime generation parameters.
        :param kwargs: Additional keyword arguments mapped to signature input fields.
        :returns: A dictionary with ``replies`` (list of ChatMessage) and ``meta`` (list of dicts).
        """
        if not messages:
            msg = "The 'messages' parameter cannot be empty."
            raise ValueError(msg)

        prompt = self._extract_last_user_message(messages)
        result = await DSPyGenerator.run_async(self, prompt=prompt, generation_kwargs=generation_kwargs, **kwargs)

        replies = [ChatMessage.from_assistant(text=text) for text in result["replies"]]

        return {"replies": replies, "meta": result["meta"]}

    @staticmethod
    def _extract_last_user_message(messages: List[ChatMessage]) -> str:
        """Extract the text of the last user message from a list of chat messages."""
        for msg in reversed(messages):
            if msg.role == ChatRole.USER:
                return msg.text

        # Fallback to last message if no user message found
        return messages[-1].text
