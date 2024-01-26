import copy
import os
from dataclasses import dataclass
from typing import List
from unittest.mock import patch

import pytest
from haystack import DeserializationError

from haystack_integrations.components.evaluators import UpTrainEvaluator, UpTrainMetric

DEFAULT_QUESTIONS = [
    "Which is the most popular global sport?",
    "Who created the Python language?",
]
DEFAULT_CONTEXTS = [
    "The popularity of sports can be measured in various ways, including TV viewership, social media presence, number of participants, and economic impact. Football is undoubtedly the world's most popular sport with major events like the FIFA World Cup and sports personalities like Ronaldo and Messi, drawing a followership of more than 4 billion people.",
    "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming language. Its design philosophy emphasizes code readability, and its language constructs aim to help programmers write clear, logical code for both small and large-scale software projects.",
]
DEFAULT_RESPONSES = [
    "Football is the most popular sport with around 4 billion followers worldwide",
    "Python language was created by Guido van Rossum.",
]


@dataclass(frozen=True)
class Unserializable:
    something: str


# Only returns results for the passed metrics.
class MockBackend:
    def __init__(self, metric_outputs: List[UpTrainMetric]) -> None:
        self.metrics = metric_outputs
        if not self.metrics:
            self.metrics = [e for e in UpTrainMetric]

    def log_and_evaluate(self, data, checks, **kwargs):
        output_map = {
            UpTrainMetric.CONTEXT_RELEVANCE: {
                "score_context_relevance": 0.5,
                "explanation_context_relevance": "1",
            },
            UpTrainMetric.FACTUAL_ACCURACY: {
                "score_factual_accuracy": 1.0,
                "explanation_factual_accuracy": "2",
            },
            UpTrainMetric.RESPONSE_RELEVANCE: {
                "score_response_relevance": 1.0,
                "explanation_response_relevance": "3",
            },
            UpTrainMetric.RESPONSE_COMPLETENESS: {
                "score_response_completeness": 0.5,
                "explanation_response_completeness": "4",
            },
            UpTrainMetric.RESPONSE_COMPLETENESS_WRT_CONTEXT: {
                "score_response_completeness_wrt_context": 1.0,
                "explanation_response_completeness_wrt_context": "5",
            },
            UpTrainMetric.RESPONSE_CONSISTENCY: {
                "score_response_consistency": 0.9,
                "explanation_response_consistency": "6",
            },
            UpTrainMetric.RESPONSE_CONCISENESS: {
                "score_response_conciseness": 1.0,
                "explanation_response_conciseness": "7",
            },
            UpTrainMetric.CRITIQUE_LANGUAGE: {
                "score_fluency": 1.0,
                "score_coherence": 1.0,
                "score_grammar": 1.0,
                "score_politeness": 1.0,
                "explanation_fluency": "8",
                "explanation_coherence": "9",
                "explanation_grammar": "10",
                "explanation_politeness": "11",
            },
            UpTrainMetric.CRITIQUE_TONE: {
                "score_tone": 0.4,
                "explanation_tone": "12",
            },
            UpTrainMetric.GUIDELINE_ADHERENCE: {
                "score_guideline_adherence": 1.0,
                "explanation_guideline_adherence": "13",
            },
            UpTrainMetric.RESPONSE_MATCHING: {
                "response_match_precision": 1.0,
                "response_match_recall": 0.6666666666666666,
                "score_response_match": 0.7272727272727273,
            },
        }

        data = copy.deepcopy(data)
        for x in data:
            for m in self.metrics:
                x.update(output_map[m])
        return data


@patch("os.environ.get")
def test_evaluator_api(os_environ_get):
    api_key_var = "test-api-key"
    os_environ_get.return_value = api_key_var

    eval = UpTrainEvaluator(UpTrainMetric.RESPONSE_COMPLETENESS)
    assert eval.api == "openai"
    assert eval.api_key_env_var == "OPENAI_API_KEY"

    eval = UpTrainEvaluator(UpTrainMetric.RESPONSE_COMPLETENESS, api="uptrain", api_key_env_var="UPTRAIN_API_KEY")
    assert eval.api == "uptrain"
    assert eval.api_key_env_var == "UPTRAIN_API_KEY"

    with pytest.raises(ValueError, match="Unsupported API"):
        UpTrainEvaluator(UpTrainMetric.CONTEXT_RELEVANCE, api="cohere")

    os_environ_get.return_value = None
    with pytest.raises(ValueError, match="Missing API key"):
        UpTrainEvaluator(UpTrainMetric.CONTEXT_RELEVANCE, api="uptrain")


@patch("os.environ.get")
def test_evaluator_metric_init_params(os_environ_get):
    api_key = "test-api-key"
    os_environ_get.return_value = api_key

    eval = UpTrainEvaluator(UpTrainMetric.CRITIQUE_TONE, metric_params={"llm_persona": "village idiot"})
    assert eval._backend_metric.llm_persona == "village idiot"

    with pytest.raises(ValueError, match="Invalid init parameters"):
        UpTrainEvaluator(UpTrainMetric.CRITIQUE_TONE, metric_params={"role": "village idiot"})

    with pytest.raises(ValueError, match="unexpected init parameters"):
        UpTrainEvaluator(UpTrainMetric.FACTUAL_ACCURACY, metric_params={"check_numbers": True})

    with pytest.raises(ValueError, match="expected init parameters"):
        UpTrainEvaluator(UpTrainMetric.RESPONSE_MATCHING)


@patch("os.environ.get")
def test_evaluator_serde(os_environ_get):
    os_environ_get.return_value = "abacab"

    init_params = {
        "metric": UpTrainMetric.RESPONSE_MATCHING,
        "metric_params": {"method": "rouge"},
        "api": "uptrain",
        "api_key_env_var": "abacab",
        "api_params": {"eval_name": "test"},
    }
    eval = UpTrainEvaluator(**init_params)
    serde_data = eval.to_dict()
    new_eval = UpTrainEvaluator.from_dict(serde_data)

    assert eval.metric == new_eval.metric
    assert eval.api == new_eval.api
    assert eval.api_key_env_var == new_eval.api_key_env_var
    assert eval.metric_params == new_eval.metric_params
    assert eval.api_params == new_eval.api_params
    assert type(new_eval._backend_client) == type(eval._backend_client)
    assert type(new_eval._backend_metric) == type(eval._backend_metric)

    with pytest.raises(DeserializationError, match=r"cannot serialize the API/metric parameters"):
        init_params3 = copy.deepcopy(init_params)
        init_params3["api_params"] = {"arg": Unserializable("")}
        eval = UpTrainEvaluator(**init_params3)
        eval.to_dict()


@pytest.mark.parametrize(
    "metric, inputs, params",
    [
        (UpTrainMetric.CONTEXT_RELEVANCE, {"questions": [], "contexts": []}, None),
        (UpTrainMetric.FACTUAL_ACCURACY, {"questions": [], "contexts": [], "responses": []}, None),
        (UpTrainMetric.RESPONSE_RELEVANCE, {"questions": [], "responses": []}, None),
        (UpTrainMetric.RESPONSE_COMPLETENESS, {"questions": [], "responses": []}, None),
        (UpTrainMetric.RESPONSE_COMPLETENESS_WRT_CONTEXT, {"questions": [], "contexts": [], "responses": []}, None),
        (UpTrainMetric.RESPONSE_CONSISTENCY, {"questions": [], "contexts": [], "responses": []}, None),
        (UpTrainMetric.RESPONSE_CONCISENESS, {"questions": [], "responses": []}, None),
        (UpTrainMetric.CRITIQUE_LANGUAGE, {"responses": []}, None),
        (UpTrainMetric.CRITIQUE_TONE, {"responses": []}, {"llm_persona": "idiot"}),
        (
            UpTrainMetric.GUIDELINE_ADHERENCE,
            {"questions": [], "responses": []},
            {"guideline": "Do nothing", "guideline_name": "somename", "response_schema": None},
        ),
        (UpTrainMetric.RESPONSE_MATCHING, {"ground_truths": [], "responses": []}, {"method": "llm"}),
    ],
)
@patch("os.environ.get")
def test_evaluator_valid_inputs(os_environ_get, metric, inputs, params):
    os_environ_get.return_value = "abacab"
    init_params = {
        "metric": metric,
        "metric_params": params,
        "api": "uptrain",
        "api_key_env_var": "abacab",
        "api_params": None,
    }
    eval = UpTrainEvaluator(**init_params)
    eval._backend_client = MockBackend([metric])
    output = eval.run(**inputs)


@pytest.mark.parametrize(
    "metric, inputs, error_string, params",
    [
        (UpTrainMetric.CONTEXT_RELEVANCE, {"questions": {}, "contexts": []}, "to be a collection of type 'list'", None),
        (
            UpTrainMetric.FACTUAL_ACCURACY,
            {"questions": [1], "contexts": [2], "responses": [3]},
            "expects inputs to be of type 'str'",
            None,
        ),
        (UpTrainMetric.RESPONSE_RELEVANCE, {"questions": [""], "responses": []}, "Mismatching counts ", None),
        (UpTrainMetric.RESPONSE_RELEVANCE, {"responses": []}, "expected input parameter ", None),
    ],
)
@patch("os.environ.get")
def test_evaluator_invalid_inputs(os_environ_get, metric, inputs, error_string, params):
    os_environ_get.return_value = "abacab"
    with pytest.raises(ValueError, match=error_string):
        init_params = {
            "metric": metric,
            "metric_params": params,
            "api": "uptrain",
            "api_key_env_var": "abacab",
            "api_params": None,
        }
        eval = UpTrainEvaluator(**init_params)
        eval._backend_client = MockBackend([metric])
        output = eval.run(**inputs)


# This test validates the expected outputs of the evaluator.
# Each output is parameterized as a list of tuples, where each tuple is
# (name, score, explanation). The name and explanation are optional. If
# the name is None, then the metric name is used.
@pytest.mark.parametrize(
    "metric, inputs, expected_outputs, metric_params",
    [
        (UpTrainMetric.CONTEXT_RELEVANCE, {"questions": ["q1"], "contexts": ["c1"]}, [[(None, 0.5, "1")]], None),
        (
            UpTrainMetric.FACTUAL_ACCURACY,
            {"questions": ["q2"], "contexts": ["c2"], "responses": ["r2"]},
            [[(None, 1.0, "2")]],
            None,
        ),
        (UpTrainMetric.RESPONSE_RELEVANCE, {"questions": ["q3"], "responses": ["r3"]}, [[(None, 1.0, "3")]], None),
        (UpTrainMetric.RESPONSE_COMPLETENESS, {"questions": ["q4"], "responses": ["r4"]}, [[(None, 0.5, "4")]], None),
        (
            UpTrainMetric.RESPONSE_COMPLETENESS_WRT_CONTEXT,
            {"questions": ["q5"], "contexts": ["c5"], "responses": ["r5"]},
            [[(None, 1.0, "5")]],
            None,
        ),
        (
            UpTrainMetric.RESPONSE_CONSISTENCY,
            {"questions": ["q6"], "contexts": ["c6"], "responses": ["r6"]},
            [[(None, 0.9, "6")]],
            None,
        ),
        (UpTrainMetric.RESPONSE_CONCISENESS, {"questions": ["q7"], "responses": ["r7"]}, [[(None, 1.0, "7")]], None),
        (
            UpTrainMetric.CRITIQUE_LANGUAGE,
            {"responses": ["r8"]},
            [
                [
                    ("fluency", 1.0, "8"),
                    ("coherence", 1.0, "9"),
                    ("grammar", 1.0, "10"),
                    ("politeness", 1.0, "11"),
                ]
            ],
            None,
        ),
        (UpTrainMetric.CRITIQUE_TONE, {"responses": ["r9"]}, [[("tone", 0.4, "12")]], {"llm_persona": "idiot"}),
        (
            UpTrainMetric.GUIDELINE_ADHERENCE,
            {"questions": ["q10"], "responses": ["r10"]},
            [[(None, 1.0, "13")]],
            {"guideline": "Do nothing", "guideline_name": "guideline", "response_schema": None},
        ),
        (
            UpTrainMetric.RESPONSE_MATCHING,
            {"ground_truths": ["g11"], "responses": ["r11"]},
            [
                [
                    ("response_match_precision", 1.0, None),
                    ("response_match_recall", 0.6666666666666666, None),
                    ("response_match", 0.7272727272727273, None),
                ]
            ],
            {"method": "llm"},
        ),
    ],
)
@patch("os.environ.get")
def test_evaluator_outputs(os_environ_get, metric, inputs, expected_outputs, metric_params):
    os_environ_get.return_value = "abacab"
    init_params = {
        "metric": metric,
        "metric_params": metric_params,
        "api": "uptrain",
        "api_key_env_var": "abacab",
        "api_params": None,
    }
    eval = UpTrainEvaluator(**init_params)
    eval._backend_client = MockBackend([metric])
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
@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY not set")
@pytest.mark.parametrize(
    "metric, inputs, metric_params",
    [
        (UpTrainMetric.CONTEXT_RELEVANCE, {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS}, None),
        (
            UpTrainMetric.FACTUAL_ACCURACY,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            None,
        ),
        (UpTrainMetric.RESPONSE_RELEVANCE, {"questions": DEFAULT_QUESTIONS, "responses": DEFAULT_RESPONSES}, None),
        (UpTrainMetric.RESPONSE_COMPLETENESS, {"questions": DEFAULT_QUESTIONS, "responses": DEFAULT_RESPONSES}, None),
        (
            UpTrainMetric.RESPONSE_COMPLETENESS_WRT_CONTEXT,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            None,
        ),
        (
            UpTrainMetric.RESPONSE_CONSISTENCY,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            None,
        ),
        (UpTrainMetric.RESPONSE_CONCISENESS, {"questions": DEFAULT_QUESTIONS, "responses": DEFAULT_RESPONSES}, None),
        (UpTrainMetric.CRITIQUE_LANGUAGE, {"responses": DEFAULT_RESPONSES}, None),
        (UpTrainMetric.CRITIQUE_TONE, {"responses": DEFAULT_RESPONSES}, {"llm_persona": "idiot"}),
        (
            UpTrainMetric.GUIDELINE_ADHERENCE,
            {"questions": DEFAULT_QUESTIONS, "responses": DEFAULT_RESPONSES},
            {"guideline": "Do nothing", "guideline_name": "somename", "response_schema": None},
        ),
        (
            UpTrainMetric.RESPONSE_MATCHING,
            {
                "ground_truths": [
                    "Consumerism is the most popular sport in the world",
                    "Python language was created by some dude.",
                ],
                "responses": DEFAULT_RESPONSES,
            },
            {"method": "llm"},
        ),
    ],
)
def test_integration_run(metric, inputs, metric_params):
    init_params = {
        "metric": metric,
        "metric_params": metric_params,
        "api": "openai",
    }
    eval = UpTrainEvaluator(**init_params)
    output = eval.run(**inputs)

    assert type(output) == dict
    assert len(output) == 1
    assert "results" in output
    assert len(output["results"]) == len(next(iter(inputs.values())))
