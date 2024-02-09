import json
from typing import Any, Callable, Dict, List, Optional, Union

from haystack import DeserializationError, component, default_from_dict, default_to_dict

from deepeval.evaluate import TestResult, evaluate
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
    A component that uses the DeepEval framework to evaluate inputs against a specific metric.

    The supported metrics are defined by :class:`DeepEvalMetric`. The inputs of the component
    metric-dependent.
    """

    _backend_metric: BaseMetric
    # Wrapped for easy mocking.
    _backend_callable: Callable[[List[LLMTestCase], BaseMetric], List[TestResult]]

    def __init__(
        self,
        metric: Union[str, DeepEvalMetric],
        metric_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Construct a new DeepEval evaluator.

        :param metric:
            The metric to use for evaluation.
        :param metric_params:
            Parameters to pass to the metric's constructor.
        """
        self.metric = metric if isinstance(metric, DeepEvalMetric) else DeepEvalMetric.from_str(metric)
        self.metric_params = metric_params
        self.descriptor = METRIC_DESCRIPTORS[self.metric]

        self._init_backend()
        expected_inputs = self.descriptor.input_parameters
        component.set_input_types(self, **expected_inputs)

    @component.output_types(results=List[List[Dict[str, Any]]])
    def run(self, **inputs) -> Dict[str, Any]:
        """
        Run the DeepEval evaluator.

        Example:
        ```python
        pipeline = Pipeline()
        evaluator = DeepEvalEvaluator(
            metric=DeepEvalMetric.ANSWER_RELEVANCY,
            metric_params={"model": "gpt-4"},
        )
        pipeline.add_component("evaluator", evaluator)

        # Each metric expects a specific set of parameters as input. Refer to the
        # DeepEvalMetric class' documentation for more details.
        output = pipeline.run({"evaluator": {
            "questions": ["question],
            "contexts": [["context"]],
            "responses": ["response"]
        }})
        ```

        :param inputs:
            The inputs to evaluate. These are determined by the
            metric being calculated. See :class:`DeepEvalMetric` for more
            information.
        :returns:
            A nested list of metric results. Each input can have one or more
            results, depending on the metric. Each result is a dictionary
            containing the following keys and values:
                * `name` - The name of the metric.
                * `score` - The score of the metric.
                * `explanation` - An optional explanation of the score.
        """
        InputConverters.validate_input_parameters(self.metric, self.descriptor.input_parameters, inputs)
        converted_inputs: List[LLMTestCase] = list(self.descriptor.input_converter(**inputs))  # type: ignore

        results = self._backend_callable(converted_inputs, self._backend_metric)
        converted_results = [[result.to_dict() for result in self.descriptor.output_converter(x)] for x in results]

        return {"results": converted_results}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """

        def check_serializable(obj: Any):
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
    def from_dict(cls, data: Dict[str, Any]) -> "DeepEvalEvaluator":
        """
        Deserialize a component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        """
        return default_from_dict(cls, data)

    @staticmethod
    def _invoke_deepeval(test_cases: List[LLMTestCase], metric: BaseMetric) -> List[TestResult]:
        return evaluate(test_cases, [metric])

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
