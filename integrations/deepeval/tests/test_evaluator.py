import copy
import os
from dataclasses import dataclass
from typing import ClassVar

import pytest
from deepeval.evaluate.types import EvaluationResult, TestResult
from deepeval.metrics import BaseMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run import MetricData
from haystack import DeserializationError

from haystack_integrations.components.evaluators.deepeval import DeepEvalEvaluator
from haystack_integrations.components.evaluators.deepeval.metrics import (
    DeepEvalMetric,
    InputConverters,
    SingleTurnParams,
)

DEFAULT_QUESTIONS = [
    "Which is the most popular global sport?",
    "Who created the Python language?",
]
DEFAULT_CONTEXTS = [
    [
        (
            "The popularity of sports can be measured in various ways, including TV viewership, social media "
            "presence, number of participants, and economic impact. Football is undoubtedly the world's most popular "
            "sport with major events like the FIFA World Cup and sports personalities like Ronaldo and Messi, "
            "drawing a followership of more than 4 billion people."
        )
    ],
    [
        (
            "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming "
            "language. Its design philosophy emphasizes code readability, and its language constructs aim to help "
            "programmers write clear, logical code for both small and large-scale software projects."
        )
    ],
]
DEFAULT_RESPONSES = [
    "Football is the most popular sport with around 4 billion followers worldwide",
    "Python language was created by Guido van Rossum.",
]
DEFAULT_GROUND_TRUTHS = [
    "Football is the most popular sport with over 4 billion followers worldwide. It's horrible...",
    "Python language was created by the seventh son of the seventh son.",
]


@dataclass(frozen=True)
class Unserializable:
    something: str


@dataclass(frozen=True)
class MockResult:
    score: float
    reason: str | None = None
    score_breakdown: dict[str, float] | None = None


# Only returns results for the passed metrics.
class MockBackend:
    def __init__(self, metric: DeepEvalMetric) -> None:
        self.metric = metric

    def eval(self, test_cases, metric) -> EvaluationResult:
        assert isinstance(metric, BaseMetric)

        output_map = {
            DeepEvalMetric.ANSWER_RELEVANCY: [MockResult(0.5, "1")],
            DeepEvalMetric.FAITHFULNESS: [MockResult(0.1, "2")],
            DeepEvalMetric.CONTEXTUAL_PRECISION: [MockResult(0.2, "3")],
            DeepEvalMetric.CONTEXTUAL_RECALL: [MockResult(35, "4")],
            DeepEvalMetric.CONTEXTUAL_RELEVANCE: [MockResult(1.5, "5")],
        }

        out = []
        for x in test_cases:
            r = TestResult(
                name=x.name or "",
                success=False,
                metrics_data=copy.deepcopy(output_map[self.metric]),  # type: ignore
                conversational=False,
                input=x.input,
                actual_output=x.actual_output,
                expected_output=x.expected_output,
                context=x.context,
                retrieval_context=x.retrieval_context,
            )
            out.append(r)
        return EvaluationResult(test_results=out, confident_link=None, test_run_id=None)


# A custom metric that doesn't need an LLM, so that it can be measured locally.
# The subclasses below only differ in the test case parameters they require.
class LengthMetric(BaseMetric):
    def __init__(self, max_words: int = 10) -> None:
        self.max_words = max_words
        self.threshold = 0.0

    def measure(self, test_case: LLMTestCase) -> float:
        words = len((test_case.actual_output or "").split())
        self.score = min(1.0, words / self.max_words)
        self.reason = f"The response to '{test_case.input}' contains {words} words"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)


class ResponseLengthMetric(LengthMetric):
    _required_params: ClassVar = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT]

    @property
    def __name__(self) -> str:
        return "Response Length"


class GroundTruthMetric(LengthMetric):
    _required_params: ClassVar = [
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ]


class UndeclaredParamsMetric(LengthMetric):
    """Doesn't declare `_required_params`, so the built-in RAG inputs are expected."""


class TupleParamsMetric(LengthMetric):
    """Declares `_required_params` as a tuple; DeepEval promises no concrete container."""

    _required_params: ClassVar = (SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT)


class BadParamsMetric(LengthMetric):
    """Declares `_required_params` as something we cannot read as params."""

    _required_params: ClassVar = 42


class UnsupportedParamsMetric(LengthMetric):
    _required_params: ClassVar = [SingleTurnParams.INPUT, SingleTurnParams.TOOLS_CALLED]


# Measures the metric locally, mimicking the results returned by `deepeval.evaluate`.
def measure_locally(test_cases, metric) -> EvaluationResult:
    out = []
    for test_case in test_cases:
        score = metric.measure(test_case)
        threshold = getattr(metric, "threshold", 0.0)
        r = TestResult(
            name=test_case.name or "",
            success=False,
            # `threshold` and `success` are required by deepeval < 4 and optional after,
            # so pass them explicitly to keep this double valid across the version floor.
            metrics_data=[
                MetricData(
                    name=metric.__name__,
                    score=score,
                    reason=metric.reason,
                    threshold=threshold,
                    success=score >= threshold,
                )
            ],
            conversational=False,
            input=test_case.input,
            actual_output=test_case.actual_output,
            expected_output=test_case.expected_output,
            context=test_case.context,
            retrieval_context=test_case.retrieval_context,
        )
        out.append(r)
    return EvaluationResult(test_results=out, confident_link=None, test_run_id=None)


def test_evaluator_metric_init_params(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    evaluator = DeepEvalEvaluator(DeepEvalMetric.ANSWER_RELEVANCY, metric_params={"model": "gpt-4o"})
    assert evaluator._backend_metric.evaluation_model == "gpt-4o"

    with pytest.raises(ValueError, match="Invalid init parameters"):
        DeepEvalEvaluator(DeepEvalMetric.FAITHFULNESS, metric_params={"role": "village idiot"})

    with pytest.raises(ValueError, match="expected init parameters"):
        DeepEvalEvaluator(DeepEvalMetric.CONTEXTUAL_RECALL)


def test_evaluator_serde(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    init_params = {
        "metric": DeepEvalMetric.ANSWER_RELEVANCY,
        "metric_params": {"model": "gpt-4o"},
    }
    evaluator = DeepEvalEvaluator(**init_params)
    serde_data = evaluator.to_dict()
    new_eval = DeepEvalEvaluator.from_dict(serde_data)

    assert evaluator.metric == new_eval.metric
    assert evaluator.metric_params == new_eval.metric_params
    assert isinstance(new_eval._backend_metric, type(evaluator._backend_metric))

    with pytest.raises(DeserializationError, match=r"cannot serialize the metric parameters"):
        evaluator.metric_params["model"] = Unserializable("")
        evaluator.to_dict()


@pytest.mark.parametrize(
    "metric, inputs, params",
    [
        (
            DeepEvalMetric.ANSWER_RELEVANCY,
            {"questions": [""], "contexts": [[""]], "responses": [""]},
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.FAITHFULNESS,
            {"questions": [""], "contexts": [[""]], "responses": [""]},
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_PRECISION,
            {"questions": [""], "contexts": [[""]], "responses": [""], "ground_truths": [""]},
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_RECALL,
            {"questions": [""], "contexts": [[""]], "responses": [""], "ground_truths": [""]},
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_RELEVANCE,
            {"questions": [""], "contexts": [[""]], "responses": [""]},
            {"model": "gpt-4o"},
        ),
    ],
)
def test_evaluator_valid_inputs(metric, inputs, params, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    init_params = {
        "metric": metric,
        "metric_params": params,
    }
    evaluator = DeepEvalEvaluator(**init_params)
    InputConverters.validate_input_parameters(evaluator.metric, evaluator.descriptor.input_parameters, inputs)


@pytest.mark.parametrize(
    "metric, inputs, error_string, params",
    [
        (
            DeepEvalMetric.ANSWER_RELEVANCY,
            {"questions": [], "contexts": [], "responses": []},
            "expected init parameters but got none",
            None,
        ),
        (
            DeepEvalMetric.ANSWER_RELEVANCY,
            {"questions": {}, "contexts": [], "responses": []},
            "to be a collection of type 'list'",
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.ANSWER_RELEVANCY,
            {"questions": [], "contexts": [], "responses": []},
            "Invalid init parameters",
            {"role": "chatbot"},
        ),
        (
            DeepEvalMetric.FAITHFULNESS,
            {"questions": [1], "contexts": [2], "responses": [3]},
            "expects inputs to be of type 'str'",
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.FAITHFULNESS,
            {"questions": [], "contexts": [[]], "responses": []},
            "Mismatching counts ",
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_RECALL,
            {"questions": [], "contexts": [], "responses": []},
            "expected input parameter ",
            {"model": "gpt-4o"},
        ),
    ],
)
def test_evaluator_invalid_inputs(metric, inputs, error_string, params, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    with pytest.raises(ValueError, match=error_string):
        init_params = {
            "metric": metric,
            "metric_params": params,
        }
        evaluator = DeepEvalEvaluator(**init_params)
        _ = evaluator.run(**inputs)


# This test validates the expected outputs of the evaluator.
# Each output is parameterized as a list of tuples, where each tuple is
# (name, score, explanation). The name and explanation are optional. If
# the name is None, then the metric name is used.
@pytest.mark.parametrize(
    "metric, inputs, expected_outputs, metric_params",
    [
        (
            DeepEvalMetric.ANSWER_RELEVANCY,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            [[(None, 0.5, "1")]] * 2,
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.FAITHFULNESS,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            [[(None, 0.1, "2")]] * 2,
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_PRECISION,
            {
                "questions": DEFAULT_QUESTIONS,
                "contexts": DEFAULT_CONTEXTS,
                "responses": DEFAULT_RESPONSES,
                "ground_truths": DEFAULT_GROUND_TRUTHS,
            },
            [[(None, 0.2, "3")]] * 2,
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_RECALL,
            {
                "questions": DEFAULT_QUESTIONS,
                "contexts": DEFAULT_CONTEXTS,
                "responses": DEFAULT_RESPONSES,
                "ground_truths": DEFAULT_GROUND_TRUTHS,
            },
            [[(None, 35, "4")]] * 2,
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_RELEVANCE,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            [[(None, 1.5, "5")]] * 2,
            {"model": "gpt-4o"},
        ),
    ],
)
def test_evaluator_outputs(metric, inputs, expected_outputs, metric_params, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    init_params = {
        "metric": metric,
        "metric_params": metric_params,
    }
    evaluator = DeepEvalEvaluator(**init_params)
    evaluator._backend_callable = lambda testcases, metrics: MockBackend(metric).eval(testcases, metrics)
    results = evaluator.run(**inputs)["results"]

    assert isinstance(results, type(expected_outputs))
    assert len(results) == len(expected_outputs)

    for r, o in zip(results, expected_outputs, strict=True):
        assert len(r) == len(o)

        expected = {(name if name is not None else str(metric), score, exp) for name, score, exp in o}
        got = {(x["name"], x["score"], x["explanation"]) for x in r}
        assert got == expected


def test_evaluator_custom_metric_backend():
    metric = ResponseLengthMetric(max_words=5)
    evaluator = DeepEvalEvaluator(metric)

    assert evaluator.metric is metric
    assert evaluator.metric_params is None
    # The user-provided instance is used as-is, including its threshold.
    assert evaluator._backend_metric is metric
    assert evaluator._backend_metric.threshold == 0.0
    assert evaluator._backend_metric.max_words == 5
    assert evaluator.descriptor.backend is ResponseLengthMetric


def test_evaluator_custom_metric_params_not_allowed():
    with pytest.raises(ValueError, match="'metric_params' must not be provided"):
        DeepEvalEvaluator(ResponseLengthMetric(), metric_params={"model": "gpt-4o"})


@pytest.mark.parametrize(
    "metric, expected_inputs",
    [
        (ResponseLengthMetric(), ["questions", "responses"]),
        (UndeclaredParamsMetric(), ["questions", "responses", "contexts"]),
        (TupleParamsMetric(), ["questions", "responses"]),
        (GroundTruthMetric(), ["questions", "responses", "contexts", "ground_truths"]),
    ],
)
def test_evaluator_custom_metric_expected_inputs(metric, expected_inputs):
    evaluator = DeepEvalEvaluator(metric)

    assert list(evaluator.descriptor.input_parameters) == expected_inputs
    assert list(evaluator.__haystack_input__._sockets_dict) == expected_inputs


def test_evaluator_custom_metric_unsupported_params():
    with pytest.raises(ValueError, match="cannot provide: \\['tools_called'\\]"):
        DeepEvalEvaluator(UnsupportedParamsMetric())


def test_evaluator_custom_metric_uninterpretable_params():
    # An unreadable declaration is treated the same as not declaring it at all: fall back
    # to the built-in RAG inputs rather than crashing on a private-attribute quirk.
    evaluator = DeepEvalEvaluator(BadParamsMetric())

    assert list(evaluator.descriptor.input_parameters) == ["questions", "responses", "contexts"]


def test_evaluator_custom_metric_missing_inputs():
    evaluator = DeepEvalEvaluator(ResponseLengthMetric())

    with pytest.raises(ValueError, match="expected input parameter 'responses' for metric 'ResponseLengthMetric'"):
        evaluator.run(questions=DEFAULT_QUESTIONS)


def test_evaluator_custom_metric_invalid_inputs():
    evaluator = DeepEvalEvaluator(ResponseLengthMetric())

    with pytest.raises(ValueError, match="to be a collection of type 'list'"):
        evaluator.run(questions={}, responses=DEFAULT_RESPONSES)

    with pytest.raises(ValueError, match="Mismatching counts "):
        evaluator.run(questions=DEFAULT_QUESTIONS, responses=DEFAULT_RESPONSES[:1])


def test_evaluator_custom_metric_outputs():
    evaluator = DeepEvalEvaluator(ResponseLengthMetric(max_words=10))
    evaluator._backend_callable = measure_locally

    results = evaluator.run(questions=DEFAULT_QUESTIONS, responses=DEFAULT_RESPONSES)["results"]

    assert results == [
        [
            {
                "name": "Response Length",
                "score": 1.0,
                "explanation": "The response to 'Which is the most popular global sport?' contains 12 words",
            }
        ],
        [
            {
                "name": "Response Length",
                "score": 0.8,
                "explanation": "The response to 'Who created the Python language?' contains 8 words",
            }
        ],
    ]


def test_evaluator_custom_metric_test_case_conversion():
    metric = GroundTruthMetric()
    evaluator = DeepEvalEvaluator(metric)
    test_cases = list(
        evaluator.descriptor.input_converter(
            questions=DEFAULT_QUESTIONS,
            contexts=DEFAULT_CONTEXTS,
            responses=DEFAULT_RESPONSES,
            ground_truths=DEFAULT_GROUND_TRUTHS,
        )
    )

    assert len(test_cases) == len(DEFAULT_QUESTIONS)
    for i, test_case in enumerate(test_cases):
        assert test_case.input == DEFAULT_QUESTIONS[i]
        assert test_case.actual_output == DEFAULT_RESPONSES[i]
        assert test_case.retrieval_context == DEFAULT_CONTEXTS[i]
        assert test_case.expected_output == DEFAULT_GROUND_TRUTHS[i]


def test_evaluator_custom_metric_not_serializable():
    evaluator = DeepEvalEvaluator(ResponseLengthMetric())

    with pytest.raises(DeserializationError, match=r"cannot serialize the metric instance 'ResponseLengthMetric'"):
        evaluator.to_dict()


def test_evaluator_builtin_metric_instance(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    metric = FaithfulnessMetric(model="gpt-4o", threshold=0.75)
    evaluator = DeepEvalEvaluator(metric)

    assert list(evaluator.descriptor.input_parameters) == ["questions", "responses", "contexts"]
    assert evaluator._backend_metric is metric
    assert evaluator._backend_metric.threshold == 0.75


# This integration test validates the evaluator by running it against the
# OpenAI API. It is parameterized by the metric, the inputs to the evalutor
# and the metric parameters.
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.integration
@pytest.mark.parametrize(
    "metric, inputs, metric_params",
    [
        (
            DeepEvalMetric.ANSWER_RELEVANCY,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.FAITHFULNESS,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_PRECISION,
            {
                "questions": DEFAULT_QUESTIONS,
                "contexts": DEFAULT_CONTEXTS,
                "responses": DEFAULT_RESPONSES,
                "ground_truths": DEFAULT_GROUND_TRUTHS,
            },
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_RECALL,
            {
                "questions": DEFAULT_QUESTIONS,
                "contexts": DEFAULT_CONTEXTS,
                "responses": DEFAULT_RESPONSES,
                "ground_truths": DEFAULT_GROUND_TRUTHS,
            },
            {"model": "gpt-4o"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_RELEVANCE,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            {"model": "gpt-4o"},
        ),
    ],
)
def test_integration_run(metric, inputs, metric_params):
    init_params = {
        "metric": metric,
        "metric_params": metric_params,
    }
    evaluator = DeepEvalEvaluator(**init_params)
    output = evaluator.run(**inputs)

    assert isinstance(output, dict)
    assert len(output) == 1
    assert "results" in output
    assert len(output["results"]) == len(next(iter(inputs.values())))
