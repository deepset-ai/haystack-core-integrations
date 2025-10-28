import json
from collections.abc import Callable
from typing import Any

from haystack import DeserializationError, component, default_from_dict, default_to_dict

from deepeval.evaluate import evaluate
from deepeval.evaluate.types import EvaluationResult
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from .metrics import (
    METRIC_DESCRIPTORS,
    DeepEvalMetric,
    InputConverters,
)


@component
class DeepEvalEvaluator:
    """
    A component that uses the [DeepEval framework](https://docs.confident-ai.com/docs/evaluation-introduction)
    to evaluate inputs against a specific metric. Supported metrics are defined by `DeepEvalMetric`.

    Usage example:
    ```python
    from haystack_integrations.components.evaluators.deepeval import DeepEvalEvaluator, DeepEvalMetric

    evaluator = DeepEvalEvaluator(
        metric=DeepEvalMetric.FAITHFULNESS,
        metric_params={"model": "gpt-4"},
    )
    output = evaluator.run(
        questions=["Which is the most popular global sport?"],
        contexts=[
            [
                "Football is undoubtedly the world's most popular sport with"
                "major events like the FIFA World Cup and sports personalities"
                "like Ronaldo and Messi, drawing a followership of more than 4"
                "billion people."
            ]
        ],
        responses=["Football is the most popular sport with around 4 billion" "followers worldwide"],
    )
    print(output["results"])
    ```
    """

    _backend_metric: BaseMetric
    # Wrapped for easy mocking.
    _backend_callable: Callable[[list[LLMTestCase], BaseMetric], EvaluationResult]

    def __init__(
        self,
        metric: str | DeepEvalMetric,
        metric_params: dict[str, Any] | None = None,
    ):
        """
        Construct a new DeepEval evaluator.

        :param metric:
            The metric to use for evaluation.
        :param metric_params:
            Parameters to pass to the metric's constructor.
            Refer to the `RagasMetric` class for more details
            on required parameters.
        """
        self.metric = metric if isinstance(metric, DeepEvalMetric) else DeepEvalMetric.from_str(metric)
        self.metric_params = metric_params
        self.descriptor = METRIC_DESCRIPTORS[self.metric]

        self._init_backend()
        expected_inputs = self.descriptor.input_parameters
        component.set_input_types(self, **expected_inputs)

    @component.output_types(results=list[list[dict[str, Any]]])
    def run(self, **inputs: Any) -> dict[str, Any]:
        """
        Run the DeepEval evaluator on the provided inputs.

        :param inputs:
            The inputs to evaluate. These are determined by the
            metric being calculated. See `DeepEvalMetric` for more
            information.
        :returns:
            A dictionary with a single `results` entry that contains
            a nested list of metric results. Each input can have one or more
            results, depending on the metric. Each result is a dictionary
            containing the following keys and values:
            - `name` - The name of the metric.
            - `score` - The score of the metric.
            - `explanation` - An optional explanation of the score.
        """
        InputConverters.validate_input_parameters(self.metric, self.descriptor.input_parameters, inputs)
        converted_inputs: list[LLMTestCase] = list(self.descriptor.input_converter(**inputs))  # type: ignore

        results = self._backend_callable(converted_inputs, self._backend_metric)
        converted_results = [
            [result.to_dict() for result in self.descriptor.output_converter(x)] for x in results.test_results
        ]

        return {"results": converted_results}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        :raises DeserializationError:
            If the component cannot be serialized.
        """

        def check_serializable(obj: Any) -> bool:
            try:
                json.dumps(obj)
                return True
            except (TypeError, OverflowError):
                return False

        if not check_serializable(self.metric_params):
            msg = "DeepEval evaluator cannot serialize the metric parameters"
            raise DeserializationError(msg)

        return default_to_dict(
            self,
            metric=self.metric,
            metric_params=self.metric_params,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeepEvalEvaluator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)

    @staticmethod
    def _invoke_deepeval(test_cases: list[LLMTestCase], metric: BaseMetric) -> EvaluationResult:
        return evaluate(test_cases=test_cases, metrics=[metric])

    def _init_backend(self):
        """
        Initialize the DeepEval backend.
        """
        if self.descriptor.init_parameters is not None:
            if self.metric_params is None:
                msg = f"DeepEval metric '{self.metric}' expected init parameters but got none"
                raise ValueError(msg)
            elif not all(k in self.descriptor.init_parameters for k in self.metric_params.keys()):
                msg = (
                    f"Invalid init parameters for DeepEval metric '{self.metric}'. "
                    f"Expected: {list(self.descriptor.init_parameters.keys())}"
                )

                raise ValueError(msg)
        backend_metric_params = dict(self.metric_params) if self.metric_params is not None else {}

        # This shouldn't matter at all as we aren't asserting the outputs, but just in case...
        backend_metric_params["threshold"] = 0.0
        self._backend_metric = self.descriptor.backend(**backend_metric_params)
        self._backend_callable = DeepEvalEvaluator._invoke_deepeval
