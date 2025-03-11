import copy
import os
from dataclasses import dataclass
from typing import Dict, Optional
from unittest.mock import patch

import pytest
from deepeval.evaluate import BaseMetric, TestResult
from haystack import DeserializationError

from haystack_integrations.components.evaluators.deepeval import DeepEvalEvaluator, DeepEvalMetric

DEFAULT_QUESTIONS = [
    "Which is the most popular global sport?",
    "Who created the Python language?",
]
DEFAULT_CONTEXTS = [
    [
        "The popularity of sports can be measured in various ways, including TV viewership, social media presence, number of participants, and economic impact. Football is undoubtedly the world's most popular sport with major events like the FIFA World Cup and sports personalities like Ronaldo and Messi, drawing a followership of more than 4 billion people."
    ],
    [
        "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming language. Its design philosophy emphasizes code readability, and its language constructs aim to help programmers write clear, logical code for both small and large-scale software projects."
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
    reason: Optional[str] = None
    score_breakdown: Optional[Dict[str, float]] = None


# Only returns results for the passed metrics.
class MockBackend:
    def __init__(self, metric: DeepEvalMetric) -> None:
        self.metric = metric

    def eval(self, test_cases, metric):
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
            r = TestResult(False, [], x.input, x.actual_output, x.expected_output, x.context, x.retrieval_context)
            r.metrics = copy.deepcopy(output_map[self.metric])
            out.append(r)
        return out


def test_evaluator_metric_init_params(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    eval = DeepEvalEvaluator(DeepEvalMetric.ANSWER_RELEVANCY, metric_params={"model": "gpt-4-32k"})
    assert eval._backend_metric.evaluation_model == "gpt-4-32k"

    with pytest.raises(ValueError, match="Invalid init parameters"):
        DeepEvalEvaluator(DeepEvalMetric.FAITHFULNESS, metric_params={"role": "village idiot"})

    with pytest.raises(ValueError, match="expected init parameters"):
        DeepEvalEvaluator(DeepEvalMetric.CONTEXTUAL_RECALL)


def test_evaluator_serde(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    init_params = {
        "metric": DeepEvalMetric.ANSWER_RELEVANCY,
        "metric_params": {"model": "gpt-4-32k"},
    }
    eval = DeepEvalEvaluator(**init_params)
    serde_data = eval.to_dict()
    new_eval = DeepEvalEvaluator.from_dict(serde_data)

    assert eval.metric == new_eval.metric
    assert eval.metric_params == new_eval.metric_params
    assert type(new_eval._backend_metric) == type(eval._backend_metric)

    with pytest.raises(DeserializationError, match=r"cannot serialize the metric parameters"):
        eval.metric_params["model"] = Unserializable("")
        eval.to_dict()


@pytest.mark.parametrize(
    "metric, inputs, params",
    [
        (
            DeepEvalMetric.ANSWER_RELEVANCY,
            {"questions": [], "contexts": [], "responses": []},
            {"model": "gpt-4"},
        ),
        (
            DeepEvalMetric.FAITHFULNESS,
            {"questions": [], "contexts": [], "responses": []},
            {"model": "gpt-4"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_PRECISION,
            {"questions": [], "contexts": [], "responses": [], "ground_truths": []},
            {"model": "gpt-4"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_RECALL,
            {"questions": [], "contexts": [], "responses": [], "ground_truths": []},
            {"model": "gpt-4"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_RELEVANCE,
            {"questions": [], "contexts": [], "responses": []},
            {"model": "gpt-4"},
        ),
    ],
)
def test_evaluator_valid_inputs(metric, inputs, params, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    init_params = {
        "metric": metric,
        "metric_params": params,
    }
    eval = DeepEvalEvaluator(**init_params)
    output = eval.run(**inputs)


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
            {"model": "gpt-4"},
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
            {"model": "gpt-4"},
        ),
        (
            DeepEvalMetric.FAITHFULNESS,
            {"questions": [], "contexts": [[]], "responses": []},
            "Mismatching counts ",
            {"model": "gpt-4"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_RECALL,
            {"questions": [], "contexts": [], "responses": []},
            "expected input parameter ",
            {"model": "gpt-4"},
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
        eval = DeepEvalEvaluator(**init_params)
        output = eval.run(**inputs)


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
            {"model": "gpt-4"},
        ),
        (
            DeepEvalMetric.FAITHFULNESS,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            [[(None, 0.1, "2")]] * 2,
            {"model": "gpt-4"},
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
            {"model": "gpt-4"},
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
            {"model": "gpt-4"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_RELEVANCE,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            [[(None, 1.5, "5")]] * 2,
            {"model": "gpt-4"},
        ),
    ],
)
def test_evaluator_outputs(metric, inputs, expected_outputs, metric_params, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    init_params = {
        "metric": metric,
        "metric_params": metric_params,
    }
    eval = DeepEvalEvaluator(**init_params)
    eval._backend_callable = lambda testcases, metrics: MockBackend(metric).eval(testcases, metrics)
    results = eval.run(**inputs)["results"]

    assert type(results) == type(expected_outputs)
    assert len(results) == len(expected_outputs)

    for r, o in zip(results, expected_outputs):
        assert len(r) == len(o)

        expected = {(name if name is not None else str(metric), score, exp) for name, score, exp in o}
        got = {(x["name"], x["score"], x["explanation"]) for x in r}
        assert got == expected


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
            {"model": "gpt-4"},
        ),
        (
            DeepEvalMetric.FAITHFULNESS,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            {"model": "gpt-4"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_PRECISION,
            {
                "questions": DEFAULT_QUESTIONS,
                "contexts": DEFAULT_CONTEXTS,
                "responses": DEFAULT_RESPONSES,
                "ground_truths": DEFAULT_GROUND_TRUTHS,
            },
            {"model": "gpt-4"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_RECALL,
            {
                "questions": DEFAULT_QUESTIONS,
                "contexts": DEFAULT_CONTEXTS,
                "responses": DEFAULT_RESPONSES,
                "ground_truths": DEFAULT_GROUND_TRUTHS,
            },
            {"model": "gpt-4"},
        ),
        (
            DeepEvalMetric.CONTEXTUAL_RELEVANCE,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            {"model": "gpt-4"},
        ),
    ],
)
def test_integration_run(metric, inputs, metric_params):
    init_params = {
        "metric": metric,
        "metric_params": metric_params,
    }
    eval = DeepEvalEvaluator(**init_params)
    output = eval.run(**inputs)

    assert type(output) == dict
    assert len(output) == 1
    assert "results" in output
    assert len(output["results"]) == len(next(iter(inputs.values())))
