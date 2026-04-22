import pytest
from unittest.mock import MagicMock
from ragas.metrics.base import SimpleBaseMetric
from ragas.metrics.result import MetricResult
from haystack import Document
from haystack_integrations.components.evaluators.ragas import RagasEvaluator


def _make_metric(name: str, score: float = 0.8, reason: str = "test reason") -> MagicMock:
    """Create a mock SimpleBaseMetric with a concrete ascore signature for inspect.signature."""
    metric = MagicMock(spec=SimpleBaseMetric)
    metric.name = name
    metric.score.return_value = MetricResult(value=score, reason=reason)

    async def ascore(user_input: str, response: str, retrieved_contexts: list) -> MetricResult:
        return MetricResult(value=score, reason=reason)

    metric.ascore = ascore
    return metric


class TestInitialization:
    def test_successful_initialization(self):
        metric = _make_metric("faithfulness")
        evaluator = RagasEvaluator(ragas_metrics=[metric])
        assert evaluator.metrics == [metric]

    def test_initialization_with_multiple_metrics(self):
        metrics = [_make_metric("faithfulness"), _make_metric("answer_relevancy")]
        evaluator = RagasEvaluator(ragas_metrics=metrics)
        assert len(evaluator.metrics) == 2

    def test_invalid_metrics_raises_type_error(self):
        with pytest.raises(TypeError, match="All items in ragas_metrics must be instances of SimpleBaseMetric."):
            RagasEvaluator(ragas_metrics=["not_a_metric"])

    def test_invalid_metrics_mixed_raises_type_error(self):
        valid = _make_metric("faithfulness")
        with pytest.raises(TypeError):
            RagasEvaluator(ragas_metrics=[valid, "not_a_metric"])


class TestRunResultStructure:
    def test_run_returns_metric_results_keyed_by_name(self):
        metric = _make_metric("faithfulness", score=0.9)
        evaluator = RagasEvaluator(ragas_metrics=[metric])

        output = evaluator.run(
            query="Which is the most popular global sport?",
            response="Football is the most popular sport.",
            documents=["Football is undoubtedly the world's most popular sport."],
        )

        assert "result" in output
        assert "faithfulness" in output["result"]
        result = output["result"]["faithfulness"]
        assert isinstance(result, MetricResult)
        assert result.value == 0.9

    def test_run_scores_all_metrics(self):
        metrics = [_make_metric("faithfulness", 0.9), _make_metric("answer_relevancy", 0.7)]
        evaluator = RagasEvaluator(ragas_metrics=metrics)

        output = evaluator.run(query="test?", response="answer", documents=["doc"])

        assert set(output["result"].keys()) == {"faithfulness", "answer_relevancy"}
        assert output["result"]["faithfulness"].value == 0.9
        assert output["result"]["answer_relevancy"].value == 0.7

    def test_run_calls_score_on_each_metric(self):
        metric_a = _make_metric("faithfulness")
        metric_b = _make_metric("answer_relevancy")
        evaluator = RagasEvaluator(ragas_metrics=[metric_a, metric_b])

        evaluator.run(query="test?", response="answer", documents=["doc"])

        metric_a.score.assert_called_once()
        metric_b.score.assert_called_once()


class TestRunParameterFiltering:
    def test_score_metric_passes_only_matching_params(self):
        """Metric that only needs user_input + response should not receive retrieved_contexts."""
        metric = MagicMock(spec=SimpleBaseMetric)
        metric.name = "selective_metric"
        metric.score.return_value = MetricResult(value=0.5, reason="ok")

        async def ascore(user_input: str, response: str) -> MetricResult:
            return MetricResult(value=0.5, reason="ok")

        metric.ascore = ascore

        evaluator = RagasEvaluator(ragas_metrics=[metric])
        evaluator.run(query="test?", response="answer", documents=["doc"], reference="ref")

        metric.score.assert_called_once_with(user_input="test?", response="answer")

    def test_score_metric_omits_none_fields(self):
        metric = _make_metric("faithfulness")
        evaluator = RagasEvaluator(ragas_metrics=[metric])

        evaluator.run(query="test?", response="answer")  # no documents → retrieved_contexts=None

        _, kwargs = metric.score.call_args
        assert "retrieved_contexts" not in kwargs


class TestRunInputProcessing:
    def test_run_accepts_document_objects(self):
        metric = _make_metric("faithfulness")
        evaluator = RagasEvaluator(ragas_metrics=[metric])

        evaluator.run(
            query="test?",
            response="answer",
            documents=[Document(content="some content"), Document(content="more content")],
        )

        _, kwargs = metric.score.call_args
        assert kwargs["retrieved_contexts"] == ["some content", "more content"]

    def test_run_accepts_string_documents(self):
        metric = _make_metric("faithfulness")
        evaluator = RagasEvaluator(ragas_metrics=[metric])

        evaluator.run(query="test?", response="answer", documents=["doc one", "doc two"])

        _, kwargs = metric.score.call_args
        assert kwargs["retrieved_contexts"] == ["doc one", "doc two"]


class TestRunInputValidation:
    @pytest.mark.parametrize(
        "invalid_input,field_name,error_message",
        [
            (["Invalid query type"], "query", "'query' field expected"),
            ([123, ["Invalid document"]], "documents", "'documents' must be a list"),
            (["score_1"], "rubrics", "'rubrics' field expected"),
        ],
    )
    def test_run_raises_on_invalid_input_types(self, invalid_input, field_name, error_message):
        evaluator = RagasEvaluator(ragas_metrics=[_make_metric("faithfulness")])
        query = "Which is the most popular global sport?"
        documents = ["Football is the most popular sport."]
        response = "Football is the most popular sport in the world"

        with pytest.raises(ValueError) as exc_info:
            if field_name == "query":
                evaluator.run(query=invalid_input, documents=documents, response=response)
            elif field_name == "documents":
                evaluator.run(query=query, documents=invalid_input, response=response)
            elif field_name == "rubrics":
                evaluator.run(query=query, rubrics=invalid_input, documents=documents, response=response)

        assert error_message in str(exc_info.value)
