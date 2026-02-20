from typing import Any, Callable, Dict, List, Optional, Type, Union

import dspy
from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace


@component
class DSPyGenerator:
    """
    A Haystack generator component that uses DSPy signatures and modules
    for structured generation.

    Takes a string prompt and returns string replies.

    Usage example:

    ```python
    from haystack_integrations.components.generators.dspy import DSPyGenerator
    import dspy

    class QASignature(dspy.Signature):
        question = dspy.InputField(desc="The user's question")
        answer = dspy.OutputField(desc="A clear, concise answer")

    generator = DSPyGenerator(
        model="openai/gpt-5-mini",
        signature=QASignature,
        module_type="ChainOfThought",
    )

    result = generator.run(prompt="What is the capital of France?")
    print(result["replies"][0])
    ```
    """

    VALID_MODULE_TYPES = {"Predict", "ChainOfThought", "ReAct"}

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
        Initialize the DSPyGenerator.

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
        if module_type not in self.VALID_MODULE_TYPES:
            msg = f"Invalid module_type '{module_type}'. Must be one of {sorted(self.VALID_MODULE_TYPES)}"
            raise ValueError(msg)

        self.signature = signature
        self.model = model
        self.api_key = api_key
        self.module_type = module_type
        self.output_field = output_field
        self.generation_kwargs = generation_kwargs or {}
        self.input_mapping = input_mapping
        self.streaming_callback = streaming_callback

        self._lm = dspy.LM(
            model=self.model,
            api_key=self.api_key.resolve_value(),
            **self.generation_kwargs,
        )
        dspy.configure(lm=self._lm)

        module_class = self._get_module_class(self.module_type)
        self._module = module_class(self.signature)

    @component.output_types(replies=List[str])
    def run(
        self,
        prompt: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the DSPy module on the given prompt.

        :param prompt: The input prompt string.
        :param generation_kwargs: Optional runtime generation parameters that override
            the defaults for this call only.
        :param kwargs: Additional keyword arguments mapped to signature input fields.
        :returns: A dictionary with ``replies`` (list of strings) and ``meta`` (list of dicts).
        """
        dspy_inputs = self._build_dspy_inputs(prompt, **kwargs)

        if generation_kwargs:
            prediction = self._module(**dspy_inputs, config=generation_kwargs)
        else:
            prediction = self._module(**dspy_inputs)

        output_text = getattr(prediction, self.output_field, str(prediction))
        meta = {"model": self.model, "module_type": self.module_type}

        return {"replies": [output_text], "meta": [meta]}

    @component.output_types(replies=List[str])
    async def run_async(
        self,
        prompt: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Asynchronously run the DSPy module on the given prompt.

        Uses DSPy's native ``acall`` for true async I/O.

        :param prompt: The input prompt string.
        :param generation_kwargs: Optional runtime generation parameters that override
            the defaults for this call only.
        :param kwargs: Additional keyword arguments mapped to signature input fields.
        :returns: A dictionary with ``replies`` (list of strings) and ``meta`` (list of dicts).
        """
        dspy_inputs = self._build_dspy_inputs(prompt, **kwargs)

        if generation_kwargs:
            prediction = await self._module.acall(**dspy_inputs, config=generation_kwargs)
        else:
            prediction = await self._module.acall(**dspy_inputs)

        output_text = getattr(prediction, self.output_field, str(prediction))
        meta = {"model": self.model, "module_type": self.module_type}

        return {"replies": [output_text], "meta": [meta]}

    def _build_dspy_inputs(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Build the input dict for the DSPy module call."""
        if self.input_mapping:
            dspy_inputs = {}
            for sig_field, source in self.input_mapping.items():
                if source in kwargs:
                    dspy_inputs[sig_field] = kwargs[source]
                else:
                    dspy_inputs[sig_field] = prompt
            return dspy_inputs

        # Default: map prompt to the first input field
        input_fields = self._get_input_field_names()
        dspy_inputs = {input_fields[0]: prompt}

        # Pass any additional kwargs that match remaining input fields
        for field in input_fields[1:]:
            if field in kwargs:
                dspy_inputs[field] = kwargs[field]

        return dspy_inputs

    def _get_input_field_names(self) -> List[str]:
        """Get input field names from the signature."""
        if isinstance(self.signature, str):
            input_part = self.signature.split("->")[0].strip()
            return [f.strip() for f in input_part.split(",")]
        return list(self.signature.input_fields.keys())

    @staticmethod
    def _get_module_class(module_type: str):
        """Map a module type string to the corresponding DSPy module class."""
        mapping = {
            "Predict": dspy.Predict,
            "ChainOfThought": dspy.ChainOfThought,
            "ReAct": dspy.ReAct,
        }
        return mapping[module_type]

    def _signature_to_string(self) -> str:
        """Convert the signature to a string representation for serialization."""
        if isinstance(self.signature, str):
            return self.signature
        input_names = list(self.signature.input_fields.keys())
        output_names = list(self.signature.output_fields.keys())
        return ", ".join(input_names) + " -> " + ", ".join(output_names)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this component to a dictionary."""
        kwargs: Dict[str, Any] = {
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
            # Token-based secrets cannot be serialized; omit them.
            pass
        return default_to_dict(self, **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DSPyGenerator":
        """Deserialize a component from a dictionary."""
        init_params = data.get("init_parameters", {})
        deserialize_secrets_inplace(init_params, ["api_key"])
        return default_from_dict(cls, data)
