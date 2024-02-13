import json
from typing import Any, Callable, Dict, List, Optional, Union

from datasets import Dataset
from haystack import DeserializationError, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from ragas import evaluate
from ragas.evaluation import Result
from ragas.metrics import AspectCritique
from ragas.metrics.base import Metric

from .metrics import (
    METRIC_ASPECTS,
    METRIC_DESCRIPTORS,
    InputConverters,
    OutputConverters,
    RagasMetric,
)


@component
class RagasEvaluator:
    """
    A component that uses the Ragas framework to evaluate inputs against a specific metric.

    The supported metrics are defined by :class:`RagasMetric`. The inputs of the component are
    metric-dependent. The output is a nested list of evaluation results where each inner list
    contains the results for a single input.
    """

    # Wrapped for easy mocking.
    _backend_callable: Callable

    def __init__(
        self,
        metric: Union[str, RagasMetric],
        metric_params: Optional[Dict[str, Any]] = None,
        *,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
    ):
        """
        Construct a new Ragas evaluator.

        :param metric:
            The metric to use for evaluation.
        :param metric_params:
            Parameters to pass to the metric's constructor.
        :param api_key:
            The API key to use.
        """
        self.metric = metric if isinstance(metric, RagasMetric) else RagasMetric.from_str(metric)
        self.metric_params = metric_params
        self.descriptor = METRIC_DESCRIPTORS[self.metric]
        self.api_key = api_key

        self._init_backend()

        expected_inputs = self.descriptor.input_parameters
        component.set_input_types(self, **expected_inputs)

    @staticmethod
    def _invoke_evaluate(dataset: Dataset, metric: Metric) -> Result:
        return evaluate(dataset, [metric])

    def _init_backend(self):
        """
        Initialize the Ragas backend and validate inputs.
        """
        if self.metric == RagasMetric.ASPECT_CRITIQUE:
            if not self.metric_params:
                msg = (
                    f"Invalid init parameters for Ragas metric '{self.metric}'. "
                    f"Expected metric parameters describing the aspect to critique but got none."
                )
                raise ValueError(msg)
            if "aspect" in self.metric_params and ("name" in self.metric_params or "definition" in self.metric_params):
                msg = (
                    f"Invalid init parameters for Ragas metric '{self.metric}'. "
                    f"If a predefined aspect is selected, no additional metric parameters are allowed."
                )
                raise ValueError(msg)
            elif "name" in self.metric_params and "definition" not in self.metric_params:
                msg = (
                    f"Invalid init parameters for Ragas metric '{self.metric}'. "
                    f"If a name of a custom aspect is provided, a definition must be provided as well."
                )
                raise ValueError(msg)
            elif "definition" in self.metric_params and "name" not in self.metric_params:
                msg = (
                    f"Invalid init parameters for Ragas metric '{self.metric}'. "
                    f"If a definition of a custom aspect is provided, a name must be provided as well."
                )
                raise ValueError(msg)
        elif self.metric_params:
            msg = (
                f"Unexpected init parameters for Ragas metric '{self.metric}'. "
                f"Additional parameters only supported for AspectCritique."
            )
            raise ValueError(msg)
        self._backend_callable = RagasEvaluator._invoke_evaluate

    @component.output_types(results=List[List[Dict[str, Any]]])
    def run(self, **inputs) -> Dict[str, Any]:
        """
        Run the Ragas evaluator.

        Example:
        ```python
        p = Pipeline()
        evaluator = RagasEvaluator(
            metric=RagasMetric.CONTEXT_PRECISION,
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
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
        metric = None
        if self.metric == RagasMetric.ASPECT_CRITIQUE and self.metric_params:
            if "aspect" in self.metric_params:
                metric = METRIC_ASPECTS[self.metric_params["aspect"]]
            else:
                metric = AspectCritique(**self.metric_params)
        else:
            metric = self.descriptor.backend
        results = self._backend_callable(dataset=dataset, metric=metric)

        OutputConverters.validate_outputs(results)
        converted_results = [
            [result.to_dict()] for result in OutputConverters.extract_results(results, self.metric, self.metric_params)
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
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RagasEvaluator":
        """
        Deserialize a component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        """
        deserialize_secrets_inplace(data["init_parameters"], ["api_key"])
        return default_from_dict(cls, data)
