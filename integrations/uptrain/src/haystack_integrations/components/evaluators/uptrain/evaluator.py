import json
from typing import Any, Dict, List, Optional, Union

from haystack import DeserializationError, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from uptrain import APIClient, EvalLLM, Evals  # type: ignore
from uptrain.framework.evals import ParametricEval

from .metrics import (
    METRIC_DESCRIPTORS,
    InputConverters,
    OutputConverters,
    UpTrainMetric,
)


@component
class UpTrainEvaluator:
    """
    A component that uses the [UpTrain framework](https://docs.uptrain.ai/getting-started/introduction)
    to evaluate inputs against a specific metric. Supported metrics are defined by `UpTrainMetric`.

    Usage example:
    ```python
    from haystack_integrations.components.evaluators.uptrain import UpTrainEvaluator, UpTrainMetric
    from haystack.utils import Secret

    evaluator = UpTrainEvaluator(
        metric=UpTrainMetric.FACTUAL_ACCURACY,
        api="openai",
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
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

    _backend_metric: Union[Evals, ParametricEval]
    _backend_client: Union[APIClient, EvalLLM]

    def __init__(
        self,
        metric: Union[str, UpTrainMetric],
        metric_params: Optional[Dict[str, Any]] = None,
        *,
        api: str = "openai",
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        api_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Construct a new UpTrain evaluator.

        :param metric:
            The metric to use for evaluation.
        :param metric_params:
            Parameters to pass to the metric's constructor.
            Refer to the `UpTrainMetric` class for more details
            on required parameters.
        :param api:
            The API to use for evaluation. Supported APIs:
            `openai`, `uptrain`.
        :param api_key:
            The API key to use.
        :param api_params:
            Additional parameters to pass to the API client.
            Required parameters for the UpTrain API: `project_name`.
        """
        self.metric = metric if isinstance(metric, UpTrainMetric) else UpTrainMetric.from_str(metric)
        self.metric_params = metric_params
        self.descriptor = METRIC_DESCRIPTORS[self.metric]
        self.api = api
        self.api_key = api_key
        self.api_params = api_params

        self._init_backend()
        expected_inputs = self.descriptor.input_parameters
        component.set_input_types(self, **expected_inputs)

    @component.output_types(results=List[List[Dict[str, Any]]])
    def run(self, **inputs) -> Dict[str, Any]:
        """
        Run the UpTrain evaluator on the provided inputs.

        :param inputs:
            The inputs to evaluate. These are determined by the
            metric being calculated. See `UpTrainMetric` for more
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
        # The backend requires random access to the data, so we can't stream it.
        InputConverters.validate_input_parameters(self.metric, self.descriptor.input_parameters, inputs)
        converted_inputs: List[Dict[str, str]] = list(self.descriptor.input_converter(**inputs))  # type: ignore

        eval_args = {"data": converted_inputs, "checks": [self._backend_metric]}
        if self.api_params is not None:
            eval_args.update({k: v for k, v in self.api_params.items() if k not in eval_args})

        results: List[Dict[str, Any]]
        if isinstance(self._backend_client, EvalLLM):
            results = self._backend_client.evaluate(**eval_args)
        else:
            results = self._backend_client.log_and_evaluate(**eval_args)

        OutputConverters.validate_outputs(results)
        converted_results = [
            [result.to_dict() for result in self.descriptor.output_converter(x, self.metric_params)] for x in results
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

        if not check_serializable(self.api_params) or not check_serializable(self.metric_params):
            msg = "UpTrain evaluator cannot serialize the API/metric parameters"
            raise DeserializationError(msg)

        return default_to_dict(
            self,
            metric=self.metric,
            metric_params=self.metric_params,
            api=self.api,
            api_key=self.api_key.to_dict(),
            api_params=self.api_params,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UpTrainEvaluator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], ["api_key"])
        return default_from_dict(cls, data)

    def _init_backend(self):
        if isinstance(self.descriptor.backend, Evals):
            if self.metric_params is not None:
                msg = (
                    f"Uptrain metric '{self.metric}' received the following unexpected init parameters:"
                    f"{self.metric_params}"
                )
                raise ValueError(msg)
            backend_metric = self.descriptor.backend
        else:
            assert issubclass(self.descriptor.backend, ParametricEval)
            if self.metric_params is None:
                msg = f"Uptrain metric '{self.metric}' expected init parameters but got none"
                raise ValueError(msg)
            elif not all(k in self.descriptor.init_parameters for k in self.metric_params.keys()):
                msg = (
                    f"Invalid init parameters for UpTrain metric '{self.metric}'. "
                    f"Expected: {list(self.descriptor.init_parameters.keys())}"
                )

                raise ValueError(msg)
            backend_metric = self.descriptor.backend(**self.metric_params)

        supported_apis = ("openai", "uptrain")
        if self.api not in supported_apis:
            msg = f"Unsupported API '{self.api}' for UpTrain evaluator. Supported APIs: {supported_apis}"
            raise ValueError(msg)

        api_key = self.api_key.resolve_value()
        assert api_key is not None
        if self.api == "openai":
            backend_client = EvalLLM(openai_api_key=api_key)
            if self.api_params is not None:
                msg = "OpenAI API does not support additional parameters"
                raise ValueError(msg)
        elif self.api == "uptrain":
            if self.api_params is None or "project_name" not in self.api_params:
                msg = "UpTrain API requires a 'project_name' API parameter"
                raise ValueError(msg)
            backend_client = APIClient(uptrain_api_key=api_key)

        self._backend_metric = backend_metric
        self._backend_client = backend_client
