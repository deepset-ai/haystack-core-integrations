import pytest

from uptrain_haystack import UpTrainMetric


def test_uptrain_metric():
    for e in UpTrainMetric:
        assert e == UpTrainMetric.from_str(e.value)

    with pytest.raises(ValueError, match="Unknown UpTrain metric"):
        UpTrainMetric.from_str("smugness")
