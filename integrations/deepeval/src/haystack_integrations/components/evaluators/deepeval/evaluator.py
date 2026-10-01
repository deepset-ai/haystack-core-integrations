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
    MetricDescriptor,
    _metric_name,
)


@component
class DeepEvalEvaluator:
    """
    A component that uses DeepEval to evaluate inputs against a specific metric.

    Uses the [DeepEval framework](https://docs.confident-ai.com/docs/evaluation-introduction).
    Supported built-in metrics are defined by `DeepEvalMetric`. Alternatively, any
    initialized DeepEval metric can be passed directly, which makes it possible to use
    custom metrics that subclass `deepeval.metrics.BaseMetric`.

    Note that a component configured with an already initialized metric cannot be
    serialized: a metric instance carries runtime state that cannot be reliably
    reconstructed. Use one of the `DeepEvalMetric` built-ins if the pipeline needs
    to survive `to_dict`/`from_dict`.

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

    Usage example with a custom metric:
    ```python
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase, SingleTurnParams

    class ResponseLengthMetric(BaseMetric):
        _required_params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]

        def __init__(self, max_words: int = 20) -> None:
            self.max_words = max_words

        def measure(self, test_case: LLMTestCase) -> float:
            words = len((test_case.actual_output or "").split())
            self.score = min(1.0, words / self.max_words)
            self.reason = f"The response contains {words} words"
            return self.score

        async def a_measure(self, test_case: LLMTestCase) -> float:
            return self.measure(test_case)

        @property
        def __name__(self) -> str:
            return "Response Length"

    evaluator = DeepEvalEvaluator(metric=ResponseLengthMetric(max_words=10))
    output = evaluator.run(
        questions=["Which is the most popular global sport?"],
        responses=["Football"],
    )
    print(output["results"])
    ```
    """

    metric: DeepEvalMetric | BaseMetric
    descriptor: MetricDescriptor
    _backend_metric: BaseMetric
    # Wrapped for easy mocking.
    _backend_callable: Callable[[list[LLMTestCase], BaseMetric], EvaluationResult]

    def __init__(
        self,
        metric: str | DeepEvalMetric | BaseMetric,
        metric_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Construct a new DeepEval evaluator.

        :param metric:
            The metric to use for evaluation. Either one of the built-in metrics
            defined by `DeepEvalMetric` (or its string value), or an already
            initialized DeepEval metric such as a custom subclass of
            `deepeval.metrics.BaseMetric`. In the latter case, the inputs expected
            by the component are derived from the test case parameters that the
            metric declares as required.
        :param metric_params:
            Parameters to pass to the metric's constructor.
            Refer to the `DeepEvalMetric` class for more details
            on required parameters. Not supported when `metric` is an
            already initialized metric, which is expected to be fully configured.
        """
        if isinstance(metric, BaseMetric):
            if metric_params is not None:
                msg = (
                    f"DeepEval metric '{_metric_name(metric)}' is already initialized, "
                    f"'metric_params' must not be provided"
                )
                raise ValueError(msg)
            self.metric = metric
            self.descriptor = MetricDescriptor.for_custom_metric(metric)
        else:
            self.metric = metric if isinstance(metric, DeepEvalMetric) else DeepEvalMetric.from_str(metric)
            self.descriptor = METRIC_DESCRIPTORS[self.metric]
        self.metric_params = metric_params

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

        Only evaluators using a built-in metric can be serialized; metric instances
        provided by the user cannot be reconstructed from a dictionary.

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

        if isinstance(self.metric, BaseMetric):
            msg = (
                f"DeepEval evaluator cannot serialize the metric instance '{_metric_name(self.metric)}'. "
                f"Use one of the built-in metrics defined by 'DeepEvalMetric' to serialize the component"
            )
            raise DeserializationError(msg)

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

    def _init_backend(self) -> None:
        """
        Initialize the DeepEval backend.
        """
        self._backend_callable = DeepEvalEvaluator._invoke_deepeval

        if isinstance(self.metric, BaseMetric):
            # Metrics provided by the user are already initialized, so we use them as-is
            # instead of instantiating the backend ourselves.
            self._backend_metric = self.metric
            return

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
