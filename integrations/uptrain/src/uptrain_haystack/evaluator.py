import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from haystack import DeserializationError, component, default_from_dict, default_to_dict
from uptrain import APIClient, EvalLLM, Evals
from uptrain.framework.evals import ParametricEval

from uptrain_haystack.metrics import (
    METRIC_DESCRIPTORS,
    InputConverters,
    OutputConverters,
    UpTrainMetric,
    UpTrainMetricResult,
)


@component
class UpTrainEvaluator:
    """
    A component that uses the UpTrain framework to evaluate inputs against a specific metric.

    The supported metrics are defined by :class:`UpTrainMetric`. The inputs of the component
    metric-dependent. The output is a list of :class:`UpTrainEvaluatorOutput` objects, each
    containing a single input and the result of the evaluation performed on it.
    """

    _backend_metric: Union[Evals, ParametricEval]
    _backend_client: Union[APIClient, EvalLLM]

    def __init__(
        self,
        metric: Union[str, UpTrainMetric],
        metric_params: Optional[Dict[str, Any]] = None,
        *,
        api: str = "openai",
        api_key_env_var: Optional[str] = "OPENAI_API_KEY",
        api_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Construct a new UpTrain evaluator.

        :param metric:
            The metric to use for evaluation.
        :param metric_params:
            Parameters to pass to the metric's constructor.
        :param api:
            The API to use for evaluation.

            Supported APIs: "openai", "uptrain".
        :param api_key_env_var:
            The name of the environment variable containing the API key.
        :param api_params:
            Additional parameters to pass to the API client.
        """
        self.metric = metric if isinstance(metric, UpTrainMetric) else UpTrainMetric.from_str(metric)
        self.metric_params = metric_params
        self.descriptor = METRIC_DESCRIPTORS[self.metric]
        self.api = api
        self.api_key_env_var = api_key_env_var
        self.api_params = api_params

        self._init_backend()
        expected_inputs = self.descriptor.input_parameters
        component.set_input_types(self, **expected_inputs)

    @component.output_types(output=List["UpTrainEvaluatorOutput"])
    def run(self, **inputs) -> Dict[str, Any]:
        """
        Run the UpTrain evaluator.

        :param inputs:
            The inputs to evaluate. Match the input parameters of the metric.
        :returns:
            A list of :class:`UpTrainEvaluatorOutput` objects, each containing
            a single input and the result of the evaluation performed on it.
        """
        # The backend requires random access to the data, so we can't stream it.
        InputConverters.validate_input_parameters(self.metric, self.descriptor.input_parameters, inputs)
        converted_inputs: List[Dict[str, str]] = list(self.descriptor.input_converter(**inputs))  # type: ignore

        eval_args = {"data": converted_inputs, "checks": [self._backend_metric]}
        if self.api_params is not None:
            eval_args.update({k: v for k, v in self.api_params.items() if k not in eval_args})

        if isinstance(self._backend_client, EvalLLM):
            results = self._backend_client.evaluate(**eval_args)
        else:
            results = self._backend_client.log_and_evaluate(**eval_args)

        OutputConverters.validate_outputs(results)
        converted_results = [self.descriptor.output_converter(x, self.metric_params) for x in results]
        output = [
            UpTrainEvaluatorOutput(input_item, result_item)
            for input_item, result_item in zip(converted_inputs, converted_results)  # type: ignore
        ]

        return {"output": output}

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

        if not check_serializable(self.api_params) or not check_serializable(self.metric_params):
            msg = "UpTrain evaluator cannot serialize the API/metric parameters"
            raise DeserializationError(msg)

        return default_to_dict(
            self,
            metric=self.metric,
            metric_params=self.metric_params,
            api=self.api,
            api_key_env_var=self.api_key_env_var,
            api_params=self.api_params,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UpTrainEvaluator":
        """
        Deserialize a component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        """
        return default_from_dict(cls, data)

    def _init_backend(self):
        """
        Initialize the UpTrain backend.
        """
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

        api_key = os.environ.get(self.api_key_env_var)
        if api_key is None:
            msg = f"Missing API key environment variable '{self.api_key_env_var}' for UpTrain evaluator"
            raise ValueError(msg)

        if self.api == "openai":
            backend_client = EvalLLM(openai_api_key=api_key)
        elif self.api == "uptrain":
            backend_client = APIClient(uptrain_api_key=api_key)

        self._backend_metric = backend_metric
        self._backend_client = backend_client


@dataclass(frozen=True)
class UpTrainEvaluatorOutput:
    """
    Output of the UpTrain evaluator component.

    :param input:
        The input that was evaluated.
    :param result:
        The result of the evaluation. Can contain
        multiple results depending on the metric.
    """

    input: Dict[str, Any]
    result: List[UpTrainMetricResult]
