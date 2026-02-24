from collections.abc import Callable
from typing import Any

import dspy
from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils import Secret, deserialize_secrets_inplace

VALID_MODULE_TYPES = {"Predict", "ChainOfThought", "ReAct"}


def _configure_dspy_lm(model: str, api_key: str, **kwargs: Any) -> dspy.LM:
    """
    Create and configure a DSPy language model.

    :param model: Model identifier (e.g. ``"openai/gpt-5-mini"``).
    :param api_key: Resolved API key string.
    :param kwargs: Additional keyword arguments passed to ``dspy.LM``.
    :returns: The configured ``dspy.LM`` instance.
    """
    lm = dspy.LM(model=model, api_key=api_key, **kwargs)
    dspy.configure(lm=lm)
    return lm


def _get_dspy_module_class(module_type: str):
    """
    Map a module type string to the corresponding DSPy module class.

    :param module_type: One of ``"Predict"``, ``"ChainOfThought"``, or ``"ReAct"``.
    :returns: The DSPy module class.
    :raises ValueError: If the module type is not recognized.
    """
    mapping = {
        "Predict": dspy.Predict,
        "ChainOfThought": dspy.ChainOfThought,
        "ReAct": dspy.ReAct,
    }
    if module_type not in mapping:
        msg = f"Invalid module_type '{module_type}'. Must be one of {sorted(VALID_MODULE_TYPES)}"
        raise ValueError(msg)
    return mapping[module_type]


@component
class DSPyChatGenerator:
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
        signature: str | type[dspy.Signature],
        model: str = "openai/gpt-5-mini",
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        module_type: str = "ChainOfThought",
        output_field: str = "answer",
        generation_kwargs: dict[str, Any] | None = None,
        input_mapping: dict[str, str] | None = None,
        streaming_callback: Callable | None = None,
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
        if module_type not in VALID_MODULE_TYPES:
            msg = f"Invalid module_type '{module_type}'. Must be one of {sorted(VALID_MODULE_TYPES)}"
            raise ValueError(msg)

        self.signature = signature
        self.model = model
        self.api_key = api_key
        self.module_type = module_type
        self.output_field = output_field
        self.generation_kwargs = generation_kwargs or {}
        self.input_mapping = input_mapping
        self.streaming_callback = streaming_callback

        self._lm = _configure_dspy_lm(
            model=self.model,
            api_key=self.api_key.resolve_value(),
            **self.generation_kwargs,
        )

        module_class = _get_dspy_module_class(self.module_type)
        self._module = module_class(self.signature)

    def _build_dspy_inputs(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Build the input dict for the DSPy module call."""
        if self.input_mapping:
            dspy_inputs = {}
            for sig_field, source in self.input_mapping.items():
                if source in kwargs:
                    dspy_inputs[sig_field] = kwargs[source]
                else:
                    dspy_inputs[sig_field] = prompt
            return dspy_inputs

        input_fields = self._get_input_field_names()
        dspy_inputs = {input_fields[0]: prompt}

        for field in input_fields[1:]:
            if field in kwargs:
                dspy_inputs[field] = kwargs[field]

        return dspy_inputs

    def _get_input_field_names(self) -> list[str]:
        """Get input field names from the signature."""
        if isinstance(self.signature, str):
            input_part = self.signature.split("->")[0].strip()
            return [f.strip() for f in input_part.split(",")]
        return list(self.signature.input_fields.keys())

    @staticmethod
    def _extract_last_user_message(messages: list[ChatMessage]) -> str:
        """Extract the text of the last user message from a list of chat messages."""
        for msg in reversed(messages):
            if msg.role == ChatRole.USER:
                return msg.text
        return messages[-1].text

    def _signature_to_string(self) -> str:
        """Convert the signature to a string representation for serialization."""
        if isinstance(self.signature, str):
            return self.signature
        input_names = list(self.signature.input_fields.keys())
        output_names = list(self.signature.output_fields.keys())
        return ", ".join(input_names) + " -> " + ", ".join(output_names)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        kwargs: dict[str, Any] = {
            "signature": self._signature_to_string(),
            "model": self.model,
            "module_type": self.module_type,
            "output_field": self.output_field,
            "generation_kwargs": self.generation_kwargs,
            "input_mapping": self.input_mapping,
        }
        try:
            kwargs["api_key"] = self.api_key.to_dict()
        except ValueError:
            pass
        return default_to_dict(self, **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DSPyChatGenerator":
        """Deserialize a component from a dictionary."""
        init_params = data.get("init_parameters", {})
        deserialize_secrets_inplace(init_params, ["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
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
        dspy_inputs = self._build_dspy_inputs(prompt, **kwargs)

        if generation_kwargs:
            prediction = self._module(**dspy_inputs, config=generation_kwargs)
        else:
            prediction = self._module(**dspy_inputs)

        output_text = getattr(prediction, self.output_field, str(prediction))

        replies = [ChatMessage.from_assistant(text=output_text)]
        return {"replies": replies}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
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
        dspy_inputs = self._build_dspy_inputs(prompt, **kwargs)

        if generation_kwargs:
            prediction = await self._module.acall(**dspy_inputs, config=generation_kwargs)
        else:
            prediction = await self._module.acall(**dspy_inputs)

        output_text = getattr(prediction, self.output_field, str(prediction))

        replies = [ChatMessage.from_assistant(text=output_text)]
        return {"replies": replies}
