import dataclasses
import inspect
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

from uptrain import CritiqueTone, Evals, GuidelineAdherence, ResponseMatching
from uptrain.framework.evals import ParametricEval


class UpTrainMetric(Enum):
    """
    Metrics supported by UpTrain.
    """

    #: Context relevance.
    #: Inputs - `questions: List[str], contexts: List[str]`
    CONTEXT_RELEVANCE = "context_relevance"

    #: Factual accuracy.
    #: Inputs - `questions: List[str], contexts: List[str], responses: List[str]`
    FACTUAL_ACCURACY = "factual_accuracy"

    #: Response relevance.
    #: Inputs - `questions: List[str], responses: List[str]`
    RESPONSE_RELEVANCE = "response_relevance"

    #: Response completeness.
    #: Inputs - `questions: List[str], responses: List[str]`
    RESPONSE_COMPLETENESS = "response_completeness"

    #: Response completeness with respect to context.
    #: Inputs - `questions: List[str], contexts: List[str], responses: List[str]`
    RESPONSE_COMPLETENESS_WRT_CONTEXT = "response_completeness_wrt_context"

    #: Response consistency.
    #: Inputs - `questions: List[str], contexts: List[str], responses: List[str]`
    RESPONSE_CONSISTENCY = "response_consistency"

    #: Response conciseness.
    #: Inputs - `questions: List[str], responses: List[str]`
    RESPONSE_CONCISENESS = "response_conciseness"

    #: Language critique.
    #: Inputs - `responses: List[str]`
    CRITIQUE_LANGUAGE = "critique_language"

    #: Tone critique.
    #: Inputs - `responses: List[str]`
    CRITIQUE_TONE = "critique_tone"

    #: Guideline adherence.
    #: Inputs - `questions: List[str], responses: List[str]`
    GUIDELINE_ADHERENCE = "guideline_adherence"

    #: Response matching.
    #: Inputs - `responses: List[str], ground_truths: List[str]`
    RESPONSE_MATCHING = "response_matching"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str) -> "UpTrainMetric":
        """
        Create a metric type from a string.

        :param string:
            The string to convert.
        :returns:
            The metric.
        """
        enum_map = {e.value: e for e in UpTrainMetric}
        metric = enum_map.get(string)
        if metric is None:
            msg = f"Unknown UpTrain metric '{string}'. Supported metrics: {list(enum_map.keys())}"
            raise ValueError(msg)
        return metric


@dataclass(frozen=True)
class MetricResult:
    """
    Result of a metric evaluation.

    :param name:
        The name of the metric.
    :param score:
        The score of the metric.
    :param explanation:
        An optional explanation of the metric.
    """

    name: str
    score: float
    explanation: Optional[str] = None

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class MetricDescriptor:
    """
    Descriptor for a metric.

    :param metric:
        The metric.
    :param backend:
        The associated UpTrain metric class.
    :param input_parameters:
        Parameters accepted by the metric. This is used
        to set the input types of the evaluator component.
    :param input_converter:
        Callable that converts input parameters to the UpTrain input format.
    :param output_converter:
        Callable that converts the UpTrain output format to our output format.
    :param init_parameters:
        Additional parameters that need to be passed to the metric class during initialization.
    """

    metric: UpTrainMetric
    backend: Union[Evals, Type[ParametricEval]]
    input_parameters: Dict[str, Type]
    input_converter: Callable[[Any], Iterable[Dict[str, str]]]
    output_converter: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], List[MetricResult]]
    init_parameters: Optional[Dict[str, Type[Any]]] = None

    @classmethod
    def new(
        cls,
        metric: UpTrainMetric,
        backend: Union[Evals, Type[ParametricEval]],
        input_converter: Callable[[Any], Iterable[Dict[str, str]]],
        output_converter: Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], List[MetricResult]]] = None,
        *,
        init_parameters: Optional[Dict[str, Type]] = None,
    ) -> "MetricDescriptor":
        input_converter_signature = inspect.signature(input_converter)
        input_parameters = {}
        for name, param in input_converter_signature.parameters.items():
            if name in ("cls", "self"):
                continue
            elif param.kind not in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                continue
            input_parameters[name] = param.annotation

        return cls(
            metric=metric,
            backend=backend,
            input_parameters=input_parameters,
            input_converter=input_converter,
            output_converter=output_converter if output_converter is not None else OutputConverters.default(metric),
            init_parameters=init_parameters,
        )


class InputConverters:
    """
    Converters for input parameters.

    The signature of the converter functions serves as the ground-truth of the
    expected input parameters of a given metric. They are also responsible for validating
    the input parameters and converting them to the format expected by UpTrain.
    """

    @staticmethod
    def _validate_input_elements(**kwargs):
        for k, collection in kwargs.items():
            if not isinstance(collection, list):
                msg = (
                    f"UpTrain evaluator expected input '{k}' to be a collection of type 'list', "
                    f"got '{type(collection).__name__}' instead"
                )
                raise ValueError(msg)
            elif not all(isinstance(x, str) for x in collection):
                msg = f"UpTrain evaluator expects inputs to be of type 'str' in '{k}'"
                raise ValueError(msg)

        same_length = len({len(x) for x in kwargs.values()}) == 1
        if not same_length:
            msg = f"Mismatching counts in the following inputs: {({k: len(v) for k, v in kwargs.items()})}"
            raise ValueError(msg)

    @staticmethod
    def validate_input_parameters(metric: UpTrainMetric, expected: Dict[str, Any], received: Dict[str, Any]):
        for param, _ in expected.items():
            if param not in received:
                msg = f"UpTrain evaluator expected input parameter '{param}' for metric '{metric}'"
                raise ValueError(msg)

    @staticmethod
    def question_context_response(
        questions: List[str], contexts: List[str], responses: List[str]
    ) -> Iterable[Dict[str, str]]:
        InputConverters._validate_input_elements(questions=questions, contexts=contexts, responses=responses)
        for q, c, r in zip(questions, contexts, responses):  # type: ignore
            yield {"question": q, "context": c, "response": r}

    @staticmethod
    def question_context(
        questions: List[str],
        contexts: List[str],
    ) -> Iterable[Dict[str, str]]:
        InputConverters._validate_input_elements(questions=questions, contexts=contexts)
        for q, c in zip(questions, contexts):  # type: ignore
            yield {"question": q, "context": c}

    @staticmethod
    def question_response(
        questions: List[str],
        responses: List[str],
    ) -> Iterable[Dict[str, str]]:
        InputConverters._validate_input_elements(questions=questions, responses=responses)
        for q, r in zip(questions, responses):  # type: ignore
            yield {"question": q, "response": r}

    @staticmethod
    def response(
        responses: List[str],
    ) -> Iterable[Dict[str, str]]:
        InputConverters._validate_input_elements(responses=responses)
        for r in responses:
            yield {"response": r}

    @staticmethod
    def response_ground_truth(
        responses: List[str],
        ground_truths: List[str],
    ) -> Iterable[Dict[str, str]]:
        InputConverters._validate_input_elements(ground_truths=ground_truths, responses=responses)
        for r, gt in zip(responses, ground_truths):  # type: ignore
            yield {"response": r, "ground_truth": gt}


class OutputConverters:
    """
    Converters for results returned by UpTrain.

    They are responsible for converting the results to our output format.
    """

    @staticmethod
    def validate_outputs(outputs: List[Dict[str, Any]]):
        msg = None
        if not isinstance(outputs, list):
            msg = f"Expected response from UpTrain evaluator to be a 'list', got '{type(outputs).__name__}'"
        elif not all(isinstance(x, dict) for x in outputs):
            msg = "UpTrain evaluator expects outputs to be a list of `dict`s"
        elif not all(isinstance(y, str) for x in outputs for y in x.keys()):
            msg = "UpTrain evaluator expects keys in the output dicts to be `str`"
        elif not all(isinstance(y, (float, str)) for x in outputs for y in x.values()):
            msg = "UpTrain evaluator expects values in the output dicts to be either `str` or `float`"

        if msg is not None:
            raise ValueError(msg)

    @staticmethod
    def _extract_default_results(output: Dict[str, Any], metric_name: str) -> MetricResult:
        try:
            score_key = f"score_{metric_name}"
            explanation_key = f"explanation_{metric_name}"
            return MetricResult(name=metric_name, score=output[score_key], explanation=output.get(explanation_key))
        except KeyError as e:
            msg = f"UpTrain evaluator did not return an expected output for metric '{metric_name}'"
            raise ValueError(msg) from e

    @staticmethod
    def default(
        metric: UpTrainMetric,
    ) -> Callable[[Dict[str, Any], Optional[Dict[str, Any]]], List[MetricResult]]:
        def inner(
            output: Dict[str, Any], metric_params: Optional[Dict[str, Any]], metric: UpTrainMetric  # noqa: ARG001
        ) -> List[MetricResult]:
            return [OutputConverters._extract_default_results(output, str(metric))]

        return partial(inner, metric=metric)

    @staticmethod
    def critique_language(
        output: Dict[str, Any], metric_params: Optional[Dict[str, Any]]  # noqa: ARG004
    ) -> List[MetricResult]:
        out = []
        for expected_key in ("fluency", "coherence", "grammar", "politeness"):
            out.append(OutputConverters._extract_default_results(output, expected_key))
        return out

    @staticmethod
    def critique_tone(
        output: Dict[str, Any], metric_params: Optional[Dict[str, Any]]  # noqa: ARG004
    ) -> List[MetricResult]:
        return [OutputConverters._extract_default_results(output, "tone")]

    @staticmethod
    def guideline_adherence(output: Dict[str, Any], metric_params: Optional[Dict[str, Any]]) -> List[MetricResult]:
        assert metric_params is not None
        return [OutputConverters._extract_default_results(output, f'{metric_params["guideline_name"]}_adherence')]

    @staticmethod
    def response_matching(
        output: Dict[str, Any], metric_params: Optional[Dict[str, Any]]  # noqa: ARG004
    ) -> List[MetricResult]:
        metric_str = "response_match"
        out = [OutputConverters._extract_default_results(output, metric_str)]

        # Enumerate other relevant keys.
        score_key = f"score_{metric_str}"
        for k, v in output.items():
            if k != score_key and metric_str in k and isinstance(v, float):
                out.append(MetricResult(name=k, score=v))
        return out


METRIC_DESCRIPTORS = {
    UpTrainMetric.CONTEXT_RELEVANCE: MetricDescriptor.new(
        UpTrainMetric.CONTEXT_RELEVANCE, Evals.CONTEXT_RELEVANCE, InputConverters.question_context  # type: ignore
    ),
    UpTrainMetric.FACTUAL_ACCURACY: MetricDescriptor.new(
        UpTrainMetric.FACTUAL_ACCURACY, Evals.FACTUAL_ACCURACY, InputConverters.question_context_response  # type: ignore
    ),
    UpTrainMetric.RESPONSE_RELEVANCE: MetricDescriptor.new(
        UpTrainMetric.RESPONSE_RELEVANCE, Evals.RESPONSE_RELEVANCE, InputConverters.question_response  # type: ignore
    ),
    UpTrainMetric.RESPONSE_COMPLETENESS: MetricDescriptor.new(
        UpTrainMetric.RESPONSE_COMPLETENESS, Evals.RESPONSE_COMPLETENESS, InputConverters.question_response  # type: ignore
    ),
    UpTrainMetric.RESPONSE_COMPLETENESS_WRT_CONTEXT: MetricDescriptor.new(
        UpTrainMetric.RESPONSE_COMPLETENESS_WRT_CONTEXT,
        Evals.RESPONSE_COMPLETENESS_WRT_CONTEXT,
        InputConverters.question_context_response,  # type: ignore
    ),
    UpTrainMetric.RESPONSE_CONSISTENCY: MetricDescriptor.new(
        UpTrainMetric.RESPONSE_CONSISTENCY, Evals.RESPONSE_CONSISTENCY, InputConverters.question_context_response  # type: ignore
    ),
    UpTrainMetric.RESPONSE_CONCISENESS: MetricDescriptor.new(
        UpTrainMetric.RESPONSE_CONCISENESS, Evals.RESPONSE_CONCISENESS, InputConverters.question_response  # type: ignore
    ),
    UpTrainMetric.CRITIQUE_LANGUAGE: MetricDescriptor.new(
        UpTrainMetric.CRITIQUE_LANGUAGE,
        Evals.CRITIQUE_LANGUAGE,
        InputConverters.response,
        OutputConverters.critique_language,
    ),
    UpTrainMetric.CRITIQUE_TONE: MetricDescriptor.new(
        UpTrainMetric.CRITIQUE_TONE,
        CritiqueTone,
        InputConverters.response,
        OutputConverters.critique_tone,
        init_parameters={"llm_persona": str},
    ),
    UpTrainMetric.GUIDELINE_ADHERENCE: MetricDescriptor.new(
        UpTrainMetric.GUIDELINE_ADHERENCE,
        GuidelineAdherence,
        InputConverters.question_response,  # type: ignore
        OutputConverters.guideline_adherence,
        init_parameters={"guideline": str, "guideline_name": str, "response_schema": Optional[str]},  # type: ignore
    ),
    UpTrainMetric.RESPONSE_MATCHING: MetricDescriptor.new(
        UpTrainMetric.RESPONSE_MATCHING,
        ResponseMatching,
        InputConverters.response_ground_truth,  # type: ignore
        OutputConverters.response_matching,
        init_parameters={"method": Optional[str]},  # type: ignore
    ),
}
