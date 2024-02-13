import dataclasses
import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Type

from ragas.evaluation import Result
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness, context_relevancy, answer_similarity, answer_correctness, context_utilization, AspectCritique,
)
from ragas.metrics.base import Metric
from ragas.metrics.critique import harmfulness, correctness, maliciousness, coherence, conciseness


class RagasBaseEnum(Enum):
    """
    Base functionality for a Ragas enum.
    """
    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str) -> "RagasMetric":
        """
        Create a metric type from a string.

        :param string:
            The string to convert.
        :returns:
            The metric.
        """
        enum_map = {e.value: e for e in RagasMetric}
        metric = enum_map.get(string)
        if metric is None:
            msg = f"Unknown Ragas metric '{string}'. Supported metrics: {list(enum_map.keys())}"
            raise ValueError(msg)
        return metric


class RagasMetric(RagasBaseEnum):
    """
    Metrics supported by Ragas.
    """
    # #: Answer correctness
    # Prior to Ragas version 0.1.0rc1, this metric is expected to fail with:
    # ValueError: too many values to unpack (expected 3) due to a bug in Ragas,
    # https://github.com/explodinggradients/ragas/issues/476
    # #: Inputs - `questions: List[str], responses: List[str], ground_truths: List[str]`
    ANSWER_CORRECTNESS = "answer_correctness"

    # #: Faithfulness
    # #: Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str]`
    FAITHFULNESS = "faithfulness"

    # #: Answer similarity
    # #: Inputs - `responses: List[str], ground_truths: List[str]`
    ANSWER_SIMILARITY = "answer_similarity"

    # #: Context precision
    # #: Inputs - `questions: List[str], contexts: List[List[str]], ground_truths: List[str]`
    CONTEXT_PRECISION = "context_precision"

    # #: Context utilization
    # #: Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str]`
    CONTEXT_UTILIZATION = "context_utilization"

    # #: Context recall
    # #: Inputs - `questions: List[str], contexts: List[List[str]], ground_truths: List[str]`
    CONTEXT_RECALL = "context_recall"

    # #: Aspect critique
    # #: Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str]`
    ASPECT_CRITIQUE = "aspect_critique"

    # #: Context relevancy
    # #: Inputs - `questions: List[str], contexts: List[List[str]]`
    CONTEXT_RELEVANCY = "context_relevancy"

    # #: Answer relevancy
    # #: Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str]`
    ANSWER_RELEVANCY = "answer_relevancy"


class RagasMetricAspect(str, RagasBaseEnum):
    """
    Predefined aspects supported by Ragas when evaluating Aspect critique.
    """
    HARMFULNESS = "harmfulness"

    MALICIOUSNESS = "maliciousness"

    COHERENCE = "coherence"

    CORRECTNESS = "correctness"

    CONCISENESS = "conciseness"


@dataclass(frozen=True)
class MetricResult:
    """
    Result of a metric evaluation.

    :param name:
        The name of the metric.
    :param score:
        The score of the metric.
    """

    name: str
    score: float

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class MetricDescriptor:
    """
    Descriptor for a metric.

    :param metric:
        The metric.
    :param backend:
        The associated Ragas metric class.
    :param input_parameters:
        Parameters accepted by the metric. This is used
        to set the input types of the evaluator component.
    :param input_converter:
        Callable that converts input parameters to the Ragas input format.
    :param init_parameters:
        Additional parameters that need to be passed to the metric class during initialization.
    """

    metric: RagasMetric
    backend: Type[Metric]
    input_parameters: Dict[str, Type]
    input_converter: Callable[[Any], Iterable[Dict[str, str]]]
    init_parameters: Optional[Dict[str, Type[Any]]] = None

    @classmethod
    def new(
        cls,
        metric: RagasMetric,
        backend: Type[Metric],
        input_converter: Callable[[Any], Iterable[Dict[str, str]]],
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
            init_parameters=init_parameters,
        )


class InputConverters:
    """
    Converters for input parameters.

    The signature of the converter functions serves as the ground-truth of the
    expected input parameters of a given metric. They are also responsible for validating
    the input parameters and converting them to the format expected by Ragas.
    """

    @staticmethod
    def _validate_input_elements(**kwargs):
        for k, collection in kwargs.items():
            if not isinstance(collection, list):
                msg = (
                    f"Ragas evaluator expected input '{k}' to be a collection of type 'list', "
                    f"got '{type(collection).__name__}' instead"
                )
                raise ValueError(msg)
            elif not all(isinstance(x, str) for x in collection) and not all(isinstance(x, list) for x in collection):
                msg = f"Ragas evaluator expects inputs to be of type 'str' or 'list' in '{k}'"
                raise ValueError(msg)

        same_length = len({len(x) for x in kwargs.values()}) == 1
        if not same_length:
            msg = f"Mismatching counts in the following inputs: {({k: len(v) for k, v in kwargs.items()})}"
            raise ValueError(msg)

    @staticmethod
    def validate_input_parameters(metric: RagasMetric, expected: Dict[str, Any], received: Dict[str, Any], metric_params: Optional[Dict[str, Any]] = None):
        for param, _ in expected.items():
            if param not in received:
                msg = f"Ragas evaluator expected input parameter '{param}' for metric '{metric}'"
                raise ValueError(msg)

    @staticmethod
    def question_context_response(
        questions: List[str], contexts: List[List[str]], responses: List[str]
    ) -> Iterable[Dict[str, str]]:
        InputConverters._validate_input_elements(questions=questions, contexts=contexts, responses=responses)
        for q, c, r in zip(questions, contexts, responses):  # type: ignore
            yield {"question": q, "contexts": c, "answer": r}

    @staticmethod
    def question_context_ground_truth(
        questions: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> Iterable[Dict[str, str]]:
        InputConverters._validate_input_elements(questions=questions, contexts=contexts, ground_truths=ground_truths)
        for q, c, gt in zip(questions, contexts, ground_truths):  # type: ignore
            yield {"question": q, "contexts": c, "ground_truth": gt}

    @staticmethod
    def question_context(
        questions: List[str],
        contexts: List[List[str]],
    ) -> Iterable[Dict[str, str]]:
        InputConverters._validate_input_elements(questions=questions, contexts=contexts)
        for q, c in zip(questions, contexts):  # type: ignore
            yield {"question": q, "contexts": c}

    @staticmethod
    def response_ground_truth(
        responses: List[str],
        ground_truths: List[str],
    ) -> Iterable[Dict[str, str]]:
        InputConverters._validate_input_elements(responses=responses, ground_truths=ground_truths)
        for r, gt in zip(responses, ground_truths):  # type: ignore
            yield {"answer": r, "ground_truth": gt}

    @staticmethod
    def question_response_ground_truth(
        questions: List[str],
        responses: List[str],
        ground_truths: List[str],
    ) -> Iterable[Dict[str, str]]:
        InputConverters._validate_input_elements(questions=questions, ground_truths=ground_truths, responses=responses)
        for q, r, gt in zip(questions, responses, ground_truths):  # type: ignore
            yield {"question": q, "answer": r, "ground_truth": gt}


class OutputConverters:
    """
    Converters for results returned by Ragas.

    They are responsible for converting the results to our output format.
    """

    @staticmethod
    def validate_outputs(outputs: Result):
        if not isinstance(outputs, Result):
            msg = f"Expected response from Ragas evaluator to be a 'Result', got '{type(outputs).__name__}'"
            raise ValueError(msg)

    @staticmethod
    def extract_results(output: Result, metric: RagasMetric, metric_params: Optional[Dict[str, Any]]) -> List[MetricResult]:
        try:
            metric_name = ""
            if metric == RagasMetric.ASPECT_CRITIQUE:
                metric_name = metric_params.get("name") or metric_params.get("aspect").value
            else:
                metric_name = metric.value
            output_scores: List[Dict[str, float]] = output.scores.to_list()
            print(output)
            return [MetricResult(name=metric_name, score=metric_dict[metric_name]) for metric_dict in output_scores]
        except KeyError as e:
            msg = f"Ragas evaluator did not return an expected output for metric '{e.args[0]}'"
            raise ValueError(msg) from e


METRIC_DESCRIPTORS = {
    RagasMetric.ANSWER_CORRECTNESS: MetricDescriptor.new(
        RagasMetric.ANSWER_CORRECTNESS, answer_correctness, InputConverters.question_response_ground_truth  # type: ignore
    ),
    RagasMetric.FAITHFULNESS: MetricDescriptor.new(
        RagasMetric.FAITHFULNESS, faithfulness, InputConverters.question_context_response  # type: ignore
    ),
    RagasMetric.ANSWER_SIMILARITY: MetricDescriptor.new(
        RagasMetric.ANSWER_SIMILARITY, answer_similarity, InputConverters.response_ground_truth  # type: ignore
    ),
    RagasMetric.CONTEXT_PRECISION: MetricDescriptor.new(
        RagasMetric.CONTEXT_PRECISION, context_precision, InputConverters.question_context_ground_truth  # type: ignore
    ),
    RagasMetric.CONTEXT_UTILIZATION: MetricDescriptor.new(
        RagasMetric.CONTEXT_UTILIZATION, context_utilization,
        InputConverters.question_context_response,  # type: ignore
    ),
    RagasMetric.CONTEXT_RECALL: MetricDescriptor.new(
        RagasMetric.CONTEXT_RECALL, context_recall, InputConverters.question_context_ground_truth  # type: ignore
    ),
    RagasMetric.ASPECT_CRITIQUE: MetricDescriptor.new(
        RagasMetric.ASPECT_CRITIQUE, AspectCritique, InputConverters.question_context_response  # type: ignore
    ),
    RagasMetric.CONTEXT_RELEVANCY: MetricDescriptor.new(
        RagasMetric.CONTEXT_RELEVANCY, context_relevancy, InputConverters.question_context  # type: ignore
    ),
    RagasMetric.ANSWER_RELEVANCY: MetricDescriptor.new(
        RagasMetric.ANSWER_RELEVANCY, answer_relevancy, InputConverters.question_context_response  # type: ignore
    ),
}

METRIC_ASPECTS = {
    RagasMetricAspect.HARMFULNESS: harmfulness,
    RagasMetricAspect.MALICIOUSNESS: maliciousness,
    RagasMetricAspect.COHERENCE: coherence,
    RagasMetricAspect.CORRECTNESS: correctness,
    RagasMetricAspect.CONCISENESS: conciseness,
}