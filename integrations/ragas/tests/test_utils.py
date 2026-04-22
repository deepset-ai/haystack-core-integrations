from unittest.mock import MagicMock, patch

import pytest
from ragas.metrics.base import SimpleBaseMetric
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


def make_llm_mock(model: str = "gpt-4o-mini", provider: str = "openai") -> MagicMock:
    llm = MagicMock()
    llm.model = model
    llm.provider = provider
    return llm


def make_emb_mock(model: str = "text-embedding-3-small", provider: str = "openai") -> MagicMock:
    emb = MagicMock()
    emb.model = model
    emb.PROVIDER_NAME = provider
    return emb


class TestSerializeMetric:
    def test_stores_type_path(self):
        result = _serialize_metric(ConcreteMetric())
        assert "type" in result
        assert result["type"].endswith(".ConcreteMetric")

    def test_stores_name(self):
        result = _serialize_metric(ConcreteMetric(name="my_metric"))
        assert result["name"] == "my_metric"

    def test_stores_llm(self):
        metric = ConcreteMetric(llm=make_llm_mock("gpt-4o-mini", "openai"))
        result = _serialize_metric(metric)
        assert result["llm"] == {"model": "gpt-4o-mini", "provider": "openai"}

    def test_stores_embeddings(self):
        metric = ConcreteMetric(embeddings=make_emb_mock("text-embedding-3-small", "openai"))
        result = _serialize_metric(metric)
        assert result["embeddings"] == {"model": "text-embedding-3-small", "provider": "openai"}

    def test_omits_llm_when_none(self):
        assert "llm" not in _serialize_metric(ConcreteMetric())

    def test_omits_embeddings_when_none(self):
        assert "embeddings" not in _serialize_metric(ConcreteMetric())


class TestDeserializeMetric:
    def test_reconstructs_instance(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        fake_llm = make_llm_mock()
        data = _serialize_metric(ConcreteMetric(name="concrete_metric", llm=fake_llm))

        with patch("haystack_integrations.components.evaluators.ragas.utils.llm_factory", return_value=fake_llm):
            result = _deserialize_metric(data)

        assert isinstance(result, ConcreteMetric)
        assert result.name == "concrete_metric"
        assert result.llm is fake_llm

    def test_reconstructs_embeddings(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        fake_emb = make_emb_mock()
        data = _serialize_metric(ConcreteMetric(name="concrete_metric", embeddings=fake_emb))

        with patch("haystack_integrations.components.evaluators.ragas.utils.embedding_factory", return_value=fake_emb):
            result = _deserialize_metric(data)

        assert result.embeddings is fake_emb

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

    def test_round_trip(self):
        metric = ConcreteMetric(name="round_trip")
        result = _deserialize_metric(_serialize_metric(metric))

        assert isinstance(result, ConcreteMetric)
        assert result.name == "round_trip"
