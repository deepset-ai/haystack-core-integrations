import pytest

from haystack_integrations.components.evaluators.deepeval import DeepEvalMetric


def test_deepeval_metric():
    for e in DeepEvalMetric:
        assert e == DeepEvalMetric.from_str(e.value)

    with pytest.raises(ValueError, match="Unknown DeepEval metric"):
        DeepEvalMetric.from_str("smugness")
