import pytest

from haystack_integrations.components.evaluators.ragas import RagasMetric


def test_ragas_metric():
    for e in RagasMetric:
        assert e == RagasMetric.from_str(e.value)

    with pytest.raises(ValueError, match="Unknown Ragas metric"):
        RagasMetric.from_str("smugness")
