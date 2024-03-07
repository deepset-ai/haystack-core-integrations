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
    A component that uses the [Ragas framework](https://docs.ragas.io/) to evaluate
    inputs against a specific metric. Supported metrics are defined by `RagasMetric`.

    Usage example:
    ```python
    from haystack_integrations.components.evaluators.ragas import RagasEvaluator, RagasMetric

    evaluator = RagasEvaluator(
        metric=RagasMetric.CONTEXT_PRECISION,
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
        ground_truths=["Football is the most popular sport with around 4 billion" "followers worldwide"],
    )
    print(output["results"])
    ```
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
            Refer to the `RagasMetric` class for more details
            on required parameters.
        """
        self.metric = metric if isinstance(metric, RagasMetric) else RagasMetric.from_str(metric)
        self.metric_params = metric_params or {}
        self.descriptor = METRIC_DESCRIPTORS[self.metric]

        self._init_backend()
        self._init_metric()

        expected_inputs = self.descriptor.input_parameters
        component.set_input_types(self, **expected_inputs)

    def _init_backend(self):
        self._backend_callable = RagasEvaluator._invoke_evaluate

    def _init_metric(self):
        MetricParamsValidator.validate_metric_parameters(
            self.metric, self.descriptor.required_init_parameters, self.descriptor.optional_init_parameters, self.metric_params
        )
        self._backend_metric = self.descriptor.backend(**self.metric_params)

    @staticmethod
    def _invoke_evaluate(dataset: Dataset, metric: Metric) -> Result:
        return evaluate(dataset, [metric])

    @component.output_types(results=List[List[Dict[str, Any]]])
    def run(self, **inputs) -> Dict[str, Any]:
        """
        Run the Ragas evaluator on the provided inputs.

        :param inputs:
            The inputs to evaluate. These are determined by the
            metric being calculated. See `RagasMetric` for more
            information.
        :returns:
            A dictionary with a single `results` entry that contains
            a nested list of metric results. Each input can have one or more
            results, depending on the metric. Each result is a dictionary
            containing the following keys and values:
            - `name` - The name of the metric.
            - `score` - The score of the metric.
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
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        :raises DeserializationError:
            If the component cannot be serialized.
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
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)
