import pytest
from haystack.core.errors import DeserializationError
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.base import SimpleBaseMetric
from ragas.metrics.collections import Faithfulness
from ragas.metrics.result import MetricResult

from haystack_integrations.components.evaluators.ragas.utils import _deserialize_metric, _serialize_metric


class ConcreteMetric(SimpleBaseMetric):
    """Minimal concrete SimpleBaseMetric for serialization tests."""

    def __init__(self, name: str = "concrete_metric", llm=None, embeddings=None):
        self.name = name
        self.llm = llm
        self.embeddings = embeddings

    async def ascore(self, user_input: str, response: str) -> MetricResult:
        return MetricResult(value=1.0, reason="test")

    def score(self, **kwargs) -> MetricResult:
        return MetricResult(value=1.0, reason="test")


def test_serialization(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    result = _serialize_metric(
        ConcreteMetric(
            llm=llm_factory("gpt-4o-mini", client=AsyncOpenAI()),
            embeddings=embedding_factory("openai", model="text-embedding-3-small", client=AsyncOpenAI()),
        )
    )
    assert result == {
        "type": "tests.test_utils.ConcreteMetric",
        "name": "concrete_metric",
        "llm": {"model": "gpt-4o-mini", "provider": "openai"},
        "embeddings": {"model": "text-embedding-3-small", "provider": "openai"},
    }


class TestDeserializeMetric:
    def test_deserialization(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        data = {
            "type": "tests.test_utils.ConcreteMetric",
            "name": "concrete_metric",
            "llm": {"model": "gpt-4o-mini", "provider": "openai"},
            "embeddings": {"model": "text-embedding-3-small", "provider": "openai"},
        }
        result = _deserialize_metric(data)
        assert isinstance(result, ConcreteMetric)
        assert result.name == "concrete_metric"
        assert result.llm.model == "gpt-4o-mini"
        assert result.embeddings.model == "text-embedding-3-small"

    def test_raises_for_unsupported_llm_provider(self):
        data = {
            "type": "tests.test_utils.ConcreteMetric",
            "name": "concrete_metric",
            "llm": {"model": "gemini-pro", "provider": "google"},
        }

        with pytest.raises(ValueError, match="only supports the 'openai' provider"):
            _deserialize_metric(data)

    def test_raises_for_unsupported_embeddings_provider(self):
        data = {
            "type": "tests.test_utils.ConcreteMetric",
            "name": "concrete_metric",
            "embeddings": {"model": "embedding-001", "provider": "google"},
        }

        with pytest.raises(ValueError, match="only supports the 'openai' provider"):
            _deserialize_metric(data)

    def test_raises_for_untrusted_metric_module(self):
        data = {"type": "some.untrusted.Metric", "name": "some_metric"}

        # haystack-ai 2.x fails to import the unknown module; haystack-ai >= 3.0 refuses it upfront
        # because it is not on the trusted-module allowlist
        with pytest.raises((ImportError, DeserializationError)):
            _deserialize_metric(data)

    def test_ragas_own_metrics_need_no_allowlisting(self, monkeypatch):
        """Metrics shipped by ragas deserialize without the caller extending the allowlist."""
        monkeypatch.delenv("HAYSTACK_DESERIALIZATION_ALLOWLIST", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        metric = Faithfulness(llm=llm_factory("gpt-4o-mini", client=AsyncOpenAI()))

        result = _deserialize_metric(_serialize_metric(metric))

        assert isinstance(result, Faithfulness)

    def test_round_trip(self):
        metric = ConcreteMetric(name="round_trip")
        result = _deserialize_metric(_serialize_metric(metric))

        assert isinstance(result, ConcreteMetric)
        assert result.name == "round_trip"
