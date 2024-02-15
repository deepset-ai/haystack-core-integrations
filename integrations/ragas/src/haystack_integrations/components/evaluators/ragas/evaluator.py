import json
from typing import Any, Callable, Dict, List, Optional, Union

from datasets import Dataset  # type: ignore
from haystack import DeserializationError, component, default_from_dict, default_to_dict

from ragas import evaluate  # type: ignore
from ragas.evaluation import Result  # type: ignore
from ragas.metrics.base import Metric  # type: ignore

from .metrics import (
    METRIC_DESCRIPTORS,
    InputConverters,
    MetricParamsValidator,
    OutputConverters,
    RagasMetric,
)


@component
class RagasEvaluator:
    """
    A component that uses the Ragas framework to evaluate inputs against a specific metric.

    The supported metrics are defined by `RagasMetric`.
    Most of them require an OpenAI API key to be provided as an environment variable "OPENAI_API_KEY".
    The inputs of the component are metric-dependent.
    The output is a nested list of evaluation results where each inner list contains the results for a single input.
    """

    # Wrapped for easy mocking.
    _backend_callable: Callable
    _backend_metric: Metric

    def __init__(
        self,
        metric: Union[str, RagasMetric],
        metric_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Construct a new Ragas evaluator.

        :param metric:
            The metric to use for evaluation.
        :param metric_params:
            Parameters to pass to the metric's constructor.
        """
        self.metric = metric if isinstance(metric, RagasMetric) else RagasMetric.from_str(metric)
        self.metric_params = metric_params or {}
        self.descriptor = METRIC_DESCRIPTORS[self.metric]

        self._init_backend()
        self._init_metric()

        expected_inputs = self.descriptor.input_parameters
        component.set_input_types(self, **expected_inputs)

    def _init_backend(self):
        """
        Initialize the Ragas backend and validate inputs.
        """
        self._backend_callable = RagasEvaluator._invoke_evaluate

    def _init_metric(self):
        MetricParamsValidator.validate_metric_parameters(
            self.metric, self.descriptor.init_parameters, self.metric_params
        )
        self._backend_metric = self.descriptor.backend(**self.metric_params)

    @staticmethod
    def _invoke_evaluate(dataset: Dataset, metric: Metric) -> Result:
        return evaluate(dataset, [metric])

    @component.output_types(results=List[List[Dict[str, Any]]])
    def run(self, **inputs) -> Dict[str, Any]:
        """
        Run the Ragas evaluator.

        Example:
        ```python
        p = Pipeline()
        evaluator = RagasEvaluator(
            metric=RagasMetric.CONTEXT_PRECISION,
        )
        p.add_component("evaluator", evaluator)

        results = p.run({"evaluator": {"questions": QUESTIONS, "contexts": CONTEXTS, "ground_truths": GROUND_TRUTHS}})
        ```

        :param inputs:
            The inputs to evaluate. These are determined by the
            metric being calculated. See :class:`RagasMetric` for more
            information.
        :returns:
            A nested list of metric results. Each input can have one or more
            results, depending on the metric. Each result is a dictionary
            containing the following keys and values:
                * `name` - The name of the metric.
                * `score` - The score of the metric.
        """
        InputConverters.validate_input_parameters(self.metric, self.descriptor.input_parameters, inputs)
        converted_inputs: List[Dict[str, str]] = list(self.descriptor.input_converter(**inputs))  # type: ignore

        dataset = Dataset.from_list(converted_inputs)
        results = self._backend_callable(dataset=dataset, metric=self._backend_metric)

        OutputConverters.validate_outputs(results)
        converted_results = [
            [result.to_dict()] for result in self.descriptor.output_converter(results, self.metric, self.metric_params)
        ]

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
            msg = "Ragas evaluator cannot serialize the metric parameters"
            raise DeserializationError(msg)

        return default_to_dict(
            self,
            metric=self.metric,
            metric_params=self.metric_params,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RagasEvaluator":
        """
        Deserialize a component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        """
        return default_from_dict(cls, data)
