import copy
import os
from dataclasses import dataclass

import pytest
from datasets import Dataset
from haystack import DeserializationError
from haystack_integrations.components.evaluators.ragas import RagasEvaluator, RagasMetric
from ragas.evaluation import Result
from ragas.metrics.base import Metric

DEFAULT_QUESTIONS = [
    "Which is the most popular global sport?",
    "Who created the Python language?",
]
DEFAULT_CONTEXTS = [
    [
        "The popularity of sports can be measured in various ways, including TV viewership, social media presence, number of participants, and economic impact.",
        "Football is undoubtedly the world's most popular sport with major events like the FIFA World Cup and sports personalities like Ronaldo and Messi, drawing a followership of more than 4 billion people.",
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
    "Football (Soccer) is the most popular sport in the world with almost 4 billion fans around the world.",
    "Guido van Rossum is the creator of the Python programming language.",
]


@dataclass(frozen=True)
class Unserializable:
    something: str


# Only returns results for the passed metrics.
class MockBackend:
    def __init__(self, metric: RagasMetric) -> None:
        self.metric = metric

    def evaluate(self, _, metric: Metric, **kwargs):
        output_map = {
            RagasMetric.ANSWER_CORRECTNESS: Result(scores=Dataset.from_list([{"answer_correctness": 0.5}])),
            RagasMetric.FAITHFULNESS: Result(scores=Dataset.from_list([{"faithfulness": 1.0}])),
            RagasMetric.ANSWER_SIMILARITY: Result(scores=Dataset.from_list([{"answer_similarity": 1.0}])),
            RagasMetric.CONTEXT_PRECISION: Result(scores=Dataset.from_list([{"context_precision": 0.5}])),
            RagasMetric.CONTEXT_UTILIZATION: Result(scores=Dataset.from_list([{"context_utilization": 1.0}])),
            RagasMetric.CONTEXT_RECALL: Result(scores=Dataset.from_list([{"context_recall": 0.9}])),
            RagasMetric.ASPECT_CRITIQUE: Result(scores=Dataset.from_list([{"harmfulness": 1.0}])),
            RagasMetric.ANSWER_RELEVANCY: Result(scores=Dataset.from_list([{"answer_relevancy": 0.4}])),
        }
        assert isinstance(metric, Metric)
        return output_map[self.metric]


@pytest.mark.parametrize(
    "metric, init_params",
    [
        (RagasMetric.ANSWER_CORRECTNESS, {"weights": [0.5, 0.5]}),
        (RagasMetric.FAITHFULNESS, None),
        (RagasMetric.ANSWER_SIMILARITY, {"threshold": 0.5}),
        (RagasMetric.CONTEXT_PRECISION, None),
        (RagasMetric.CONTEXT_UTILIZATION, None),
        (RagasMetric.CONTEXT_RECALL, None),
        (
            RagasMetric.ASPECT_CRITIQUE,
            {
                "name": "harmfulness",
                "definition": "Does the submission causes or has the potential to "
                "cause harm to individuals, groups, or society at "
                "large?",
            },
        ),
        (RagasMetric.ANSWER_RELEVANCY, {"strictness": 2}),
    ],
)
def test_evaluator_valid_metric_init_params(metric, init_params):
    eval = RagasEvaluator(
        metric,
        metric_params=init_params,
    )
    assert eval.metric_params == init_params

    msg = f"Invalid init parameters for Ragas metric '{metric}'. "
    with pytest.raises(ValueError, match=msg):
        RagasEvaluator(
            metric,
            metric_params={"invalid_param": "invalid_value"},
        )


@pytest.mark.parametrize(
    "metric",
    [
        RagasMetric.ANSWER_CORRECTNESS,
        RagasMetric.ANSWER_SIMILARITY,
        RagasMetric.ASPECT_CRITIQUE,
        RagasMetric.ANSWER_RELEVANCY,
    ],
)
def test_evaluator_fails_with_no_metric_init_params(metric):
    msg = f"Ragas metric '{metric}' expected init parameters but got none"
    with pytest.raises(ValueError, match=msg):
        RagasEvaluator(
            metric,
            metric_params=None,
        )


def test_evaluator_serde():
    init_params = {
        "metric": RagasMetric.ASPECT_CRITIQUE,
        "metric_params": {
            "name": "harmfulness",
            "definition": "Does the submission causes or has the potential to "
            "cause harm to individuals, groups, or society at "
            "large?",
        },
    }
    eval = RagasEvaluator(**init_params)
    serde_data = eval.to_dict()
    new_eval = RagasEvaluator.from_dict(serde_data)

    assert eval.metric == new_eval.metric
    assert eval.metric_params == new_eval.metric_params

    with pytest.raises(DeserializationError, match=r"cannot serialize the metric parameters"):
        init_params3 = copy.deepcopy(init_params)
        init_params3["metric_params"]["name"] = Unserializable("")
        eval = RagasEvaluator(**init_params3)
        eval.to_dict()


@pytest.mark.parametrize(
    "current_metric, inputs, params",
    [
        (
            RagasMetric.ANSWER_CORRECTNESS,
            {"questions": [], "responses": [], "ground_truths": []},
            {"weights": [0.5, 0.5]},
        ),
        (RagasMetric.FAITHFULNESS, {"questions": [], "contexts": [], "responses": []}, None),
        (RagasMetric.ANSWER_SIMILARITY, {"responses": [], "ground_truths": []}, {"threshold": 0.5}),
        (RagasMetric.CONTEXT_PRECISION, {"questions": [], "contexts": [], "ground_truths": []}, None),
        (RagasMetric.CONTEXT_UTILIZATION, {"questions": [], "contexts": [], "responses": []}, None),
        (RagasMetric.CONTEXT_RECALL, {"questions": [], "contexts": [], "ground_truths": []}, None),
        (
            RagasMetric.ASPECT_CRITIQUE,
            {"questions": [], "contexts": [], "responses": []},
            {
                "name": "harmfulness",
                "definition": "Does the submission causes or has the potential to "
                "cause harm to individuals, groups, or society at "
                "large?",
            },
        ),
        (RagasMetric.ANSWER_RELEVANCY, {"questions": [], "contexts": [], "responses": []}, {"strictness": 2}),
    ],
)
def test_evaluator_valid_inputs(current_metric, inputs, params):
    init_params = {
        "metric": current_metric,
        "metric_params": params,
    }
    eval = RagasEvaluator(**init_params)
    eval._backend_callable = lambda dataset, metric: MockBackend(current_metric).evaluate(dataset, metric)
    output = eval.run(**inputs)


@pytest.mark.parametrize(
    "current_metric, inputs, error_string, params",
    [
        (
            RagasMetric.FAITHFULNESS,
            {"questions": [1], "contexts": [2], "responses": [3]},
            "expects inputs to be of type 'str'",
            None,
        ),
        (
            RagasMetric.ANSWER_RELEVANCY,
            {"questions": [""], "responses": [], "contexts": []},
            "Mismatching counts ",
            {"strictness": 2},
        ),
        (RagasMetric.ANSWER_RELEVANCY, {"responses": []}, "expected input parameter ", {"strictness": 2}),
    ],
)
def test_evaluator_invalid_inputs(current_metric, inputs, error_string, params):
    with pytest.raises(ValueError, match=error_string):
        init_params = {
            "metric": current_metric,
            "metric_params": params,
        }
        eval = RagasEvaluator(**init_params)
        eval._backend_callable = lambda dataset, metric: MockBackend(current_metric).evaluate(dataset, metric)
        output = eval.run(**inputs)


# This test validates the expected outputs of the evaluator.
# Each output is parameterized as a list of tuples, where each tuple is (name, score).
@pytest.mark.parametrize(
    "current_metric, inputs, expected_outputs, metric_params",
    [
        (
            RagasMetric.ANSWER_CORRECTNESS,
            {"questions": ["q1"], "responses": ["r1"], "ground_truths": ["gt1"]},
            [[(None, 0.5)]],
            {"weights": [0.5, 0.5]},
        ),
        (
            RagasMetric.FAITHFULNESS,
            {"questions": ["q2"], "contexts": [["c2"]], "responses": ["r2"]},
            [[(None, 1.0)]],
            None,
        ),
        (
            RagasMetric.ANSWER_SIMILARITY,
            {"responses": ["r3"], "ground_truths": ["gt3"]},
            [[(None, 1.0)]],
            {"threshold": 0.5},
        ),
        (
            RagasMetric.CONTEXT_PRECISION,
            {"questions": ["q4"], "contexts": [["c4"]], "ground_truths": ["gt44"]},
            [[(None, 0.5)]],
            None,
        ),
        (
            RagasMetric.CONTEXT_UTILIZATION,
            {"questions": ["q5"], "contexts": [["c5"]], "responses": ["r5"]},
            [[(None, 1.0)]],
            None,
        ),
        (
            RagasMetric.CONTEXT_RECALL,
            {"questions": ["q6"], "contexts": [["c6"]], "ground_truths": ["gt6"]},
            [[(None, 0.9)]],
            None,
        ),
        (
            RagasMetric.ASPECT_CRITIQUE,
            {"questions": ["q7"], "contexts": [["c7"]], "responses": ["r7"]},
            [[("harmfulness", 1.0)]],
            {
                "name": "harmfulness",
                "definition": "Does the submission causes or has the potential to "
                "cause harm to individuals, groups, or society at "
                "large?",
            },
        ),
        (
            RagasMetric.ANSWER_RELEVANCY,
            {"questions": ["q9"], "contexts": [["c9"]], "responses": ["r9"]},
            [[(None, 0.4)]],
            {"strictness": 2},
        ),
    ],
)
def test_evaluator_outputs(current_metric, inputs, expected_outputs, metric_params):
    init_params = {
        "metric": current_metric,
        "metric_params": metric_params,
    }
    eval = RagasEvaluator(**init_params)
    eval._backend_callable = lambda dataset, metric: MockBackend(current_metric).evaluate(dataset, metric)
    results = eval.run(**inputs)["results"]

    assert type(results) == type(expected_outputs)
    assert len(results) == len(expected_outputs)

    for r, o in zip(results, expected_outputs):
        assert len(r) == len(o)

        expected = {(name if name is not None else str(current_metric), score) for name, score in o}
        got = {(x["name"], x["score"]) for x in r}
        assert got == expected


# This integration test validates the evaluator by running it against the
# OpenAI API. It is parameterized by the metric, the inputs to the evaluator
# and the metric parameters.
@pytest.mark.asyncio
@pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY not set")
@pytest.mark.parametrize(
    "metric, inputs, metric_params",
    [
        (
            RagasMetric.ANSWER_CORRECTNESS,
            {"questions": DEFAULT_QUESTIONS, "responses": DEFAULT_RESPONSES, "ground_truths": DEFAULT_GROUND_TRUTHS},
            {"weights": [0.5, 0.5]},
        ),
        (
            RagasMetric.FAITHFULNESS,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            None,
        ),
        (
            RagasMetric.ANSWER_SIMILARITY,
            {"responses": DEFAULT_QUESTIONS, "ground_truths": DEFAULT_GROUND_TRUTHS},
            {"threshold": 0.5},
        ),
        (
            RagasMetric.CONTEXT_PRECISION,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "ground_truths": DEFAULT_GROUND_TRUTHS},
            None,
        ),
        (
            RagasMetric.CONTEXT_UTILIZATION,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            None,
        ),
        (
            RagasMetric.CONTEXT_RECALL,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "ground_truths": DEFAULT_GROUND_TRUTHS},
            None,
        ),
        (
            RagasMetric.ASPECT_CRITIQUE,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            {
                "name": "harmfulness",
                "definition": "Does the submission causes or has the potential to "
                "cause harm to individuals, groups, or society at "
                "large?",
            },
        ),
        (
            RagasMetric.ANSWER_RELEVANCY,
            {"questions": DEFAULT_QUESTIONS, "contexts": DEFAULT_CONTEXTS, "responses": DEFAULT_RESPONSES},
            {"strictness": 2},
        ),
    ],
)
def test_integration_run(metric, inputs, metric_params):
    init_params = {
        "metric": metric,
        "metric_params": metric_params,
    }
    eval = RagasEvaluator(**init_params)
    output = eval.run(**inputs)

    assert isinstance(output, dict)
    assert len(output) == 1
    assert "results" in output
    assert len(output["results"]) == len(next(iter(inputs.values())))
