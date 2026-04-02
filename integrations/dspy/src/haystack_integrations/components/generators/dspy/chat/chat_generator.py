import importlib
from typing import Any

import dspy
from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole

VALID_MODULE_TYPES = {"Predict", "ChainOfThought", "ReAct"}


def _create_dspy_lm(model: str, api_base: str | None = None, **kwargs: Any) -> dspy.LM:
    """
    Create a DSPy language model instance.

    :param model: Model identifier (e.g. `"openai/gpt-5-mini"`).
    :param api_base: Optional base URL for the API (useful for local models).
    :param kwargs: Additional keyword arguments passed to `dspy.LM`.
    :returns: The configured `dspy.LM` instance.
    """
    lm_kwargs: dict[str, Any] = {"model": model, **kwargs}
    if api_base is not None:
        lm_kwargs["api_base"] = api_base
    return dspy.LM(**lm_kwargs)


def _get_dspy_module_class(module_type: str) -> type:
    """
    Map a module type string to the corresponding DSPy module class.

    :param module_type: One of `"Predict"`, `"ChainOfThought"`, or `"ReAct"`.
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
class DSPySignatureChatGenerator:
    """
    A Haystack chat generator component that uses DSPy signatures and modules for structured generation.

    Accepts and returns `ChatMessage` objects, making it compatible with
    Haystack chat pipelines.

    The API key is read automatically from environment variables by DSPy/litellm
    (e.g. `OPENAI_API_KEY`). Use `api_base` for local or self-hosted models.

    Usage example:

    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.dspy import DSPySignatureChatGenerator
    import dspy

    class QASignature(dspy.Signature):
        question = dspy.InputField(desc="The user's question")
        answer = dspy.OutputField(desc="A clear, concise answer")

    generator = DSPySignatureChatGenerator(
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
        api_base: str | None = None,
        module_type: str = "ChainOfThought",
        output_field: str = "answer",
        generation_kwargs: dict[str, Any] | None = None,
        module_kwargs: dict[str, Any] | None = None,
        input_mapping: dict[str, str] | None = None,
    ):
        """
        Initialize the DSPySignatureChatGenerator.

        :param signature: DSPy signature defining I/O structure. Can be a string
            like `"question -> answer"` or a `dspy.Signature` subclass.
        :param model: Model identifier (e.g. `"openai/gpt-5-mini"`).
        :param api_base: Optional base URL for the API (useful for local models).
        :param module_type: DSPy module type: `"Predict"`, `"ChainOfThought"`, or `"ReAct"`.
        :param output_field: Which signature output field to use as the reply.
        :param generation_kwargs: Additional generation parameters (temperature, max_tokens, etc.).
        :param module_kwargs: Additional keyword arguments passed to the DSPy module constructor.
            For example, use `{"tools": [tool1, tool2]}` when using the `"ReAct"` module type.
        :param input_mapping: Maps DSPy signature input field names to `run()` kwarg names.
            For example, if your signature has an input field `"context"` but your pipeline
            provides it as `"documents"`, use `{"context": "documents"}`. When not provided,
            the first input field receives the last user message text, and remaining fields
            are matched by name from `**kwargs`.
        """
        if module_type not in VALID_MODULE_TYPES:
            msg = f"Invalid module_type '{module_type}'. Must be one of {sorted(VALID_MODULE_TYPES)}"
            raise ValueError(msg)

        self.signature = signature
        self.model = model
        self.api_base = api_base
        self.module_type = module_type
        self.output_field = output_field
        self.generation_kwargs = generation_kwargs or {}
        self.module_kwargs = module_kwargs or {}
        self.input_mapping = input_mapping

        self._lm = _create_dspy_lm(
            model=self.model,
            api_base=self.api_base,
            **self.generation_kwargs,
        )

        module_class = _get_dspy_module_class(self.module_type)
        self._module = module_class(self.signature, **self.module_kwargs)
        self._module.set_lm(self._lm)

    def _build_dspy_inputs(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
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
                if not msg.text:
                    err = "The last user message has no text content."
                    raise ValueError(err)
                return msg.text
        err = "No user message found in 'messages'."
        raise ValueError(err)

    @staticmethod
    def _serialize_signature(signature: str | type[dspy.Signature]) -> dict[str, str]:
        """
        Serialize the signature to a dictionary.

        String signatures are stored as
        `{"type": "str", "value": "question -> answer"}`.
        `dspy.Signature` subclasses are stored as
        `{"type": "class", "value": "mymodule.QASignature"}`.
        """
        if isinstance(signature, str):
            return {"type": "str", "value": signature}
        return {"type": "class", "value": f"{signature.__module__}.{signature.__qualname__}"}

    @staticmethod
    def _deserialize_signature(data: dict[str, str]) -> str | type[dspy.Signature]:
        """
        Deserialize a signature from a dictionary.

        Accepts `{"type": "str", "value": "question -> answer"}` or
        `{"type": "class", "value": "mymodule.QASignature"}`.
        """
        signature_type = data["type"]
        value = data["value"]

        if signature_type == "str":
            return value

        if signature_type == "class":
            module_path, class_name = value.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        msg = f"Unknown signature type '{signature_type}'. Must be 'str' or 'class'."
        raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        kwargs: dict[str, Any] = {
            "signature": self._serialize_signature(self.signature),
            "model": self.model,
            "api_base": self.api_base,
            "module_type": self.module_type,
            "output_field": self.output_field,
            "generation_kwargs": self.generation_kwargs,
            "module_kwargs": self.module_kwargs,
            "input_mapping": self.input_mapping,
        }
        return default_to_dict(self, **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DSPySignatureChatGenerator":
        """Deserialize a component from a dictionary."""
        init_params = data.get("init_parameters", {})

        signature = init_params.get("signature")
        if isinstance(signature, dict):
            init_params["signature"] = cls._deserialize_signature(signature)

        return default_from_dict(cls, data)

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run the DSPy module on the given messages.

        :param messages: List of chat messages. The last user message is used as input.
        :param generation_kwargs: Optional runtime generation parameters.
        :param kwargs: Additional keyword arguments mapped to signature input fields.
        :returns: A dictionary with `replies` (list of ChatMessage).
        """
        if not messages:
            msg = "The 'messages' parameter cannot be empty."
            raise ValueError(msg)

        prompt = self._extract_last_user_message(messages)
        dspy_inputs = self._build_dspy_inputs(prompt, **kwargs)

        prediction = self._module(**dspy_inputs, config=generation_kwargs or {})

        if not hasattr(prediction, self.output_field):
            available = list(prediction.keys())
            msg = f"Output field '{self.output_field}' not found in prediction. Available fields: {available}"
            raise ValueError(msg)
        output_text = getattr(prediction, self.output_field)

        replies = [ChatMessage.from_assistant(text=output_text)]
        return {"replies": replies}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage],
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Asynchronously run the DSPy module on the given messages.

        Uses DSPy's native `acall` for true async I/O.

        :param messages: List of chat messages. The last user message is used as input.
        :param generation_kwargs: Optional runtime generation parameters.
        :param kwargs: Additional keyword arguments mapped to signature input fields.
        :returns: A dictionary with `replies` (list of ChatMessage).
        """
        if not messages:
            msg = "The 'messages' parameter cannot be empty."
            raise ValueError(msg)

        prompt = self._extract_last_user_message(messages)
        dspy_inputs = self._build_dspy_inputs(prompt, **kwargs)

        prediction = await self._module.acall(**dspy_inputs, config=generation_kwargs or {})

        if not hasattr(prediction, self.output_field):
            available = list(prediction.keys())
            msg = f"Output field '{self.output_field}' not found in prediction. Available fields: {available}"
            raise ValueError(msg)
        output_text = getattr(prediction, self.output_field)

        replies = [ChatMessage.from_assistant(text=output_text)]
        return {"replies": replies}
