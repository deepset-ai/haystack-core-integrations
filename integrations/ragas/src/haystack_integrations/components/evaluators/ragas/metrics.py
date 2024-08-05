import dataclasses
import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

from ragas.evaluation import Result
from ragas.metrics import (  # type: ignore
    AnswerCorrectness,  # type: ignore
    AnswerRelevancy,  # type: ignore
    AnswerSimilarity,  # type: ignore
    AspectCritique,  # type: ignore
    ContextPrecision,  # type: ignore
    ContextRecall,  # type: ignore
    ContextUtilization,  # type: ignore
    Faithfulness,  # type: ignore
)
from ragas.metrics.base import Metric


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

    #: Answer correctness.\
    #: Inputs - `questions: List[str], responses: List[str], ground_truths: List[str]`\
    #: Parameters - `weights: Tuple[float, float]`
    ANSWER_CORRECTNESS = "answer_correctness"

    #: Faithfulness.\
    #: Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str]`
    FAITHFULNESS = "faithfulness"

    #: Answer similarity.\
    #: Inputs - `responses: List[str], ground_truths: List[str]`\
    #: Parameters - `threshold: float`
    ANSWER_SIMILARITY = "answer_similarity"

    #: Context precision.\
    #: Inputs - `questions: List[str], contexts: List[List[str]], ground_truths: List[str]`
    CONTEXT_PRECISION = "context_precision"

    #: Context utilization.
    #: Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str]`\
    CONTEXT_UTILIZATION = "context_utilization"

    #: Context recall.
    #: Inputs - `questions: List[str], contexts: List[List[str]], ground_truths: List[str]`\
    CONTEXT_RECALL = "context_recall"

    #: Aspect critique.
    #: Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str]`\
    #: Parameters - `name: str, definition: str, strictness: int`
    ASPECT_CRITIQUE = "aspect_critique"

    #: Answer relevancy.\
    #: Inputs - `questions: List[str], contexts: List[List[str]], responses: List[str]`\
    #: Parameters - `strictness: int`
    ANSWER_RELEVANCY = "answer_relevancy"


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
    :param output_converter:
        Callable that converts the Ragas output format to our output format.
        Accepts a single output parameter and returns a list of results derived from it.
    :param init_parameters:
        Additional parameters that are allowed to be passed to the metric class during initialization.
    """

    metric: RagasMetric
    backend: Type[Metric]
    input_parameters: Dict[str, Type]
    input_converter: Callable[[Any], Iterable[Dict[str, str]]]
    output_converter: Callable[[Result, RagasMetric, Optional[Dict[str, Any]]], List[MetricResult]]
    init_parameters: Optional[List[str]] = None

    @classmethod
    def new(
        cls,
        metric: RagasMetric,
        backend: Type[Metric],
        input_converter: Callable[[Any], Iterable[Dict[str, str]]],
        output_converter: Optional[
            Callable[[Result, RagasMetric, Optional[Dict[str, Any]]], List[MetricResult]]
        ] = None,
        *,
        init_parameters: Optional[List[str]] = None,
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
            output_converter=output_converter if output_converter is not None else OutputConverters.default,
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
    def _validate_input_elements(**kwargs) -> None:
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
    def validate_input_parameters(
        metric: RagasMetric,
        expected: Dict[str, Any],
        received: Dict[str, Any],
    ) -> None:
        for param, _ in expected.items():
            if param not in received:
                msg = f"Ragas evaluator expected input parameter '{param}' for metric '{metric}'"
                raise ValueError(msg)

    @staticmethod
    def question_context_response(
        questions: List[str], contexts: List[List[str]], responses: List[str]
    ) -> Iterable[Dict[str, Union[str, List[str]]]]:
        InputConverters._validate_input_elements(questions=questions, contexts=contexts, responses=responses)
        for q, c, r in zip(questions, contexts, responses):  # type: ignore
            yield {"question": q, "contexts": c, "answer": r}

    @staticmethod
    def question_context_ground_truth(
        questions: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> Iterable[Dict[str, Union[str, List[str]]]]:
        InputConverters._validate_input_elements(questions=questions, contexts=contexts, ground_truths=ground_truths)
        for q, c, gt in zip(questions, contexts, ground_truths):  # type: ignore
            yield {"question": q, "contexts": c, "ground_truth": gt}

    @staticmethod
    def question_context(
        questions: List[str],
        contexts: List[List[str]],
    ) -> Iterable[Dict[str, Union[str, List[str]]]]:
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
    def validate_outputs(outputs: Result) -> None:
        if not isinstance(outputs, Result):
            msg = f"Expected response from Ragas evaluator to be a 'Result', got '{type(outputs).__name__}'"
            raise ValueError(msg)

    @staticmethod
    def _extract_default_results(output: Result, metric_name: str) -> List[MetricResult]:
        try:
            output_scores: List[Dict[str, float]] = output.scores.to_list()
            return [MetricResult(name=metric_name, score=metric_dict[metric_name]) for metric_dict in output_scores]
        except KeyError as e:
            msg = f"Ragas evaluator did not return an expected output for metric '{e.args[0]}'"
            raise ValueError(msg) from e

    @staticmethod
    def default(output: Result, metric: RagasMetric, _: Optional[Dict]) -> List[MetricResult]:
        metric_name = metric.value
        return OutputConverters._extract_default_results(output, metric_name)

    @staticmethod
    def aspect_critique(output: Result, _: RagasMetric, metric_params: Optional[Dict[str, Any]]) -> List[MetricResult]:
        if metric_params is None:
            msg = "Aspect critique metric requires metric parameters"
            raise ValueError(msg)
        metric_name = metric_params["name"]
        return OutputConverters._extract_default_results(output, metric_name)


METRIC_DESCRIPTORS = {
    RagasMetric.ANSWER_CORRECTNESS: MetricDescriptor.new(
        RagasMetric.ANSWER_CORRECTNESS,
        AnswerCorrectness,
        InputConverters.question_response_ground_truth,  # type: ignore
        init_parameters=["weights"],
    ),
    RagasMetric.FAITHFULNESS: MetricDescriptor.new(
        RagasMetric.FAITHFULNESS,
        Faithfulness,
        InputConverters.question_context_response,  # type: ignore
    ),
    RagasMetric.ANSWER_SIMILARITY: MetricDescriptor.new(
        RagasMetric.ANSWER_SIMILARITY,
        AnswerSimilarity,
        InputConverters.response_ground_truth,  # type: ignore
        init_parameters=["threshold"],
    ),
    RagasMetric.CONTEXT_PRECISION: MetricDescriptor.new(
        RagasMetric.CONTEXT_PRECISION,
        ContextPrecision,
        InputConverters.question_context_ground_truth,  # type: ignore
    ),
    RagasMetric.CONTEXT_UTILIZATION: MetricDescriptor.new(
        RagasMetric.CONTEXT_UTILIZATION,
        ContextUtilization,
        InputConverters.question_context_response,  # type: ignore
    ),
    RagasMetric.CONTEXT_RECALL: MetricDescriptor.new(
        RagasMetric.CONTEXT_RECALL,
        ContextRecall,
        InputConverters.question_context_ground_truth,  # type: ignore
    ),
    RagasMetric.ASPECT_CRITIQUE: MetricDescriptor.new(
        RagasMetric.ASPECT_CRITIQUE,
        AspectCritique,
        InputConverters.question_context_response,  # type: ignore
        OutputConverters.aspect_critique,
        init_parameters=["name", "definition", "strictness"],
    ),
    RagasMetric.ANSWER_RELEVANCY: MetricDescriptor.new(
        RagasMetric.ANSWER_RELEVANCY,
        AnswerRelevancy,
        InputConverters.question_context_response,  # type: ignore
        init_parameters=["strictness"],
    ),
}
