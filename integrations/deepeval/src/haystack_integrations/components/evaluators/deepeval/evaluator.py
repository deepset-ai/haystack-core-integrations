import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

from haystack import DeserializationError, component, default_from_dict, default_to_dict

from deepeval.evaluate import evaluate
from deepeval.evaluate.types import EvaluationResult, MetricData, TestResult
from deepeval.metrics import BaseMetric
from deepeval.metrics.utils import copy_metrics
from deepeval.test_case import LLMTestCase

from .metrics import (
    METRIC_DESCRIPTORS,
    DeepEvalMetric,
    InputConverters,
)


@component
class DeepEvalEvaluator:
    """
    A component that uses DeepEval to evaluate inputs against a specific metric.

    Uses the [DeepEval framework](https://docs.confident-ai.com/docs/evaluation-introduction).
    Supported metrics are defined by `DeepEvalMetric`.

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
    _backend_callable_async: Callable[[list[LLMTestCase], BaseMetric], Awaitable[EvaluationResult]]

    def __init__(
        self,
        metric: str | DeepEvalMetric,
        metric_params: dict[str, Any] | None = None,
    ) -> None:
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
        converted_results = self._convert_results(results)

        return {"results": converted_results}

    @component.output_types(results=list[list[dict[str, Any]]])
    async def run_async(self, **inputs: Any) -> dict[str, Any]:
        """
        Run the DeepEval evaluator asynchronously on the provided inputs.

        Each test case is evaluated concurrently using the metric's async
        `a_measure` method. A separate metric copy is used per test case
        because DeepEval metrics keep state (`score`, `reason`) on the
        metric instance.

        :param inputs:
            The inputs to evaluate. These are determined by the
            metric being calculated. See `DeepEvalMetric` for more
            information.
        :returns:
            A dictionary with a single `results` entry that contains
            a nested list of metric results. The shape matches the
            output of the `run` method.
        """
        InputConverters.validate_input_parameters(self.metric, self.descriptor.input_parameters, inputs)
        converted_inputs: list[LLMTestCase] = list(self.descriptor.input_converter(**inputs))  # type: ignore

        results = await self._backend_callable_async(converted_inputs, self._backend_metric)
        converted_results = self._convert_results(results)

        return {"results": converted_results}

    def _convert_results(self, results: EvaluationResult) -> list[list[dict[str, Any]]]:
        """Convert an ``EvaluationResult`` to the evaluator's output format."""
        return [[result.to_dict() for result in self.descriptor.output_converter(x)] for x in results.test_results]

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

    @staticmethod
    async def _invoke_deepeval_async(
        test_cases: list[LLMTestCase],
        metric: BaseMetric,
        max_concurrent: int = 4,
    ) -> EvaluationResult:
        """Evaluate `test_cases` concurrently using the metric's `a_measure`."""

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _evaluate_one(test_case: LLMTestCase) -> TestResult:
            async with semaphore:
                # DeepEval metrics keep their result state on the instance, so each
                # concurrent evaluation needs its own copy.
                metric_copy = cast(BaseMetric, copy_metrics([metric])[0])
                await metric_copy.a_measure(test_case)
                metric_data = MetricData.model_construct(
                    name=metric_copy.__class__.__name__,
                    score=metric_copy.score,
                    reason=metric_copy.reason,
                )
                return TestResult(
                    name=test_case.name or "",
                    success=False,
                    conversational=False,
                    metrics_data=[metric_data],
                    input=test_case.input,
                    actual_output=test_case.actual_output,
                    expected_output=test_case.expected_output,
                    context=test_case.context,
                    retrieval_context=cast(list[str] | None, test_case.retrieval_context),
                )

        results = await asyncio.gather(*[_evaluate_one(tc) for tc in test_cases])
        return EvaluationResult(test_results=list(results), confident_link=None, test_run_id=None)

    def _init_backend(self) -> None:
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
        self._backend_callable_async = DeepEvalEvaluator._invoke_deepeval_async
