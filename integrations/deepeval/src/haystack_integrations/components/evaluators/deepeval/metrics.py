import dataclasses
import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any

from deepeval.evaluate.types import TestResult
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase


class DeepEvalMetric(Enum):
    """
    Metrics supported by DeepEval.

    All metrics require a `model` parameter, which specifies
    the model to use for evaluation. Refer to the DeepEval
    documentation for information on the supported models.
    """

    #: Answer relevancy.\
    #: Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str]`
    ANSWER_RELEVANCY = "answer_relevancy"

    #: Faithfulness.\
    #: Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str]`
    FAITHFULNESS = "faithfulness"

    #: Contextual precision.\
    #: Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str], ground_truths: List[str]`\
    #: The ground truth is the expected response.
    CONTEXTUAL_PRECISION = "contextual_precision"

    #: Contextual recall.\
    #: Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str], ground_truths: List[str]`\
    #: The ground truth is the expected response.\
    CONTEXTUAL_RECALL = "contextual_recall"

    #: Contextual relevance.\
    #: Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str]`
    CONTEXTUAL_RELEVANCE = "contextual_relevance"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str) -> "DeepEvalMetric":
        """
        Create a metric type from a string.

        :param string:
            The string to convert.
        :returns:
            The metric.
        """
        enum_map = {e.value: e for e in DeepEvalMetric}
        metric = enum_map.get(string)
        if metric is None:
            msg = f"Unknown DeepEval metric '{string}'. Supported metrics: {list(enum_map.keys())}"
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
    score: float | None = None
    explanation: str | None = None

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class MetricDescriptor:
    """
    Descriptor for a metric.

    :param metric:
        The metric.
    :param backend:
        The associated DeepEval metric class.
    :param input_parameters:
        Parameters accepted by the metric. This is used
        to set the input types of the evaluator component.
    :param input_converter:
        Callable that converts input parameters to the DeepEval input format.
    :param output_converter:
        Callable that converts the DeepEval output format to our output format.
        Accepts a single output parameter and returns a list of results derived from it.
    :param init_parameters:
        Additional parameters that need to be passed to the metric class during initialization.
    """

    metric: DeepEvalMetric
    backend: type[BaseMetric]
    input_parameters: dict[str, type]
    input_converter: Callable[[Any], Iterable[LLMTestCase]]
    output_converter: Callable[[TestResult], list[MetricResult]]
    init_parameters: Mapping[str, type] | None = None

    @classmethod
    def new(
        cls,
        metric: DeepEvalMetric,
        backend: type[BaseMetric],
        input_converter: Callable[[Any], Iterable[LLMTestCase]],
        output_converter: Callable[[TestResult], list[MetricResult]] | None = None,
        *,
        init_parameters: Mapping[str, type] | None = None,
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
    the input parameters and converting them to the format expected by DeepEval.
    """

    @staticmethod
    def _validate_input_elements(**kwargs):
        for k, collection in kwargs.items():
            if not isinstance(collection, list):
                msg = (
                    f"DeepEval evaluator expected input '{k}' to be a collection of type 'list', "
                    f"got '{type(collection).__name__}' instead"
                )
                raise ValueError(msg)
            elif not all(isinstance(x, str) for x in collection) and not all(isinstance(x, list) for x in collection):
                msg = f"DeepEval evaluator expects inputs to be of type 'str' or 'list' in '{k}'"
                raise ValueError(msg)

        same_length = len({len(x) for x in kwargs.values()}) == 1
        if not same_length:
            msg = f"Mismatching counts in the following inputs: { ({k: len(v) for k, v in kwargs.items()}) }"
            raise ValueError(msg)

    @staticmethod
    def validate_input_parameters(metric: DeepEvalMetric, expected: dict[str, Any], received: dict[str, Any]) -> None:
        for param, _ in expected.items():
            if param not in received:
                msg = f"DeepEval evaluator expected input parameter '{param}' for metric '{metric}'"
                raise ValueError(msg)

    @staticmethod
    def question_context_response(
        questions: list[str], contexts: list[list[str]], responses: list[str]
    ) -> Iterable[LLMTestCase]:
        InputConverters._validate_input_elements(questions=questions, contexts=contexts, responses=responses)
        for q, c, r in zip(questions, contexts, responses, strict=True):  # type: ignore
            test_case = LLMTestCase(input=q, actual_output=r, retrieval_context=c)
            yield test_case

    @staticmethod
    def question_context_response_ground_truth(
        questions: list[str], contexts: list[list[str]], responses: list[str], ground_truths: list[str]
    ) -> Iterable[LLMTestCase]:
        InputConverters._validate_input_elements(questions=questions, contexts=contexts, responses=responses)
        for q, c, r, gt in zip(questions, contexts, responses, ground_truths, strict=True):  # type: ignore
            test_case = LLMTestCase(input=q, actual_output=r, retrieval_context=c, expected_output=gt)
            yield test_case


class OutputConverters:
    """
    Converters for results returned by DeepEval.

    They are responsible for converting the results to our output format.
    """

    @staticmethod
    def default(
        metric: DeepEvalMetric,
    ) -> Callable[[TestResult], list[MetricResult]]:
        def inner(output: TestResult, metric: DeepEvalMetric) -> list[MetricResult]:
            metric_name = str(metric)
            assert output.metrics_data
            assert len(output.metrics_data) == 1
            metric_result = output.metrics_data[0]
            out = [MetricResult(name=metric_name, score=metric_result.score, explanation=metric_result.reason)]
            return out

        return partial(inner, metric=metric)


METRIC_DESCRIPTORS = {
    DeepEvalMetric.ANSWER_RELEVANCY: MetricDescriptor.new(
        DeepEvalMetric.ANSWER_RELEVANCY,
        AnswerRelevancyMetric,
        InputConverters.question_context_response,  # type: ignore
        init_parameters={"model": str | None},  # type: ignore
    ),
    DeepEvalMetric.FAITHFULNESS: MetricDescriptor.new(
        DeepEvalMetric.FAITHFULNESS,
        FaithfulnessMetric,
        InputConverters.question_context_response,  # type: ignore
        init_parameters={"model": str | None},  # type: ignore
    ),
    DeepEvalMetric.CONTEXTUAL_PRECISION: MetricDescriptor.new(
        DeepEvalMetric.CONTEXTUAL_PRECISION,
        ContextualPrecisionMetric,
        InputConverters.question_context_response_ground_truth,  # type: ignore
        init_parameters={"model": str | None},  # type: ignore
    ),
    DeepEvalMetric.CONTEXTUAL_RECALL: MetricDescriptor.new(
        DeepEvalMetric.CONTEXTUAL_RECALL,
        ContextualRecallMetric,
        InputConverters.question_context_response_ground_truth,  # type: ignore
        init_parameters={"model": str | None},  # type: ignore
    ),
    DeepEvalMetric.CONTEXTUAL_RELEVANCE: MetricDescriptor.new(
        DeepEvalMetric.CONTEXTUAL_RELEVANCE,
        ContextualRelevancyMetric,
        InputConverters.question_context_response,  # type: ignore
        init_parameters={"model": str | None},  # type: ignore
    ),
}
