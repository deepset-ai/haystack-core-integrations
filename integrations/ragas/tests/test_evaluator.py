import pytest
from unittest import mock
from unittest.mock import MagicMock, patch
from ragas.metrics import Metric, Faithfulness
from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings
from ragas.dataset_schema import EvaluationResult
from haystack import Document
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.evaluators.ragas import RagasEvaluator


# Fixture to mock the 'run' method of RagasEvaluator
@pytest.fixture
def mock_run():
    with mock.patch.object(RagasEvaluator, 'run') as mock_method:
        yield mock_method


def test_successful_initialization():
    """Test RagasEvaluator initializes correctly with valid inputs."""
    valid_metrics = [MagicMock(spec=Metric) for _ in range(3)]
    valid_llm = MagicMock(spec=BaseRagasLLM)
    valid_embedding = MagicMock(spec=BaseRagasEmbeddings)

    evaluator = RagasEvaluator(
        ragas_metrics=valid_metrics,
        evaluator_llm=valid_llm,
        evaluator_embedding=valid_embedding,
    )
    assert evaluator.metrics == valid_metrics
    assert evaluator.llm == valid_llm
    assert evaluator.embedding == valid_embedding


def test_invalid_metrics():
    """Test RagasEvaluator raises TypeError for invalid metrics."""
    invalid_metric = "not_a_metric"

    with pytest.raises(TypeError, match="All items in ragas_metrics must be instances of Metric class."):
        RagasEvaluator(ragas_metrics=[invalid_metric])


def test_invalid_llm():
    """Test RagasEvaluator raises TypeError for invalid evaluator_llm."""
    valid_metric = MagicMock(spec=Metric)
    invalid_llm = "not_a_llm"

    with pytest.raises(TypeError, match="Expected evaluator_llm to be BaseRagasLLM or LangchainLLM"):
        RagasEvaluator(ragas_metrics=[valid_metric], evaluator_llm=invalid_llm)


def test_invalid_embedding():
    """Test RagasEvaluator raises TypeError for invalid evaluator_embedding."""
    valid_metric = MagicMock(spec=Metric)
    invalid_embedding = "not_an_embedding"

    with pytest.raises(
        TypeError, match="Expected evaluator_embedding to be BaseRagasEmbeddings or LangchainEmbeddings"
    ):
        RagasEvaluator(ragas_metrics=[valid_metric], evaluator_embedding=invalid_embedding)


def test_optional_llm_and_embeddings():
    """Test RagasEvaluator initializes correctly with None for optional parameters."""
    valid_metric = MagicMock(spec=Metric)

    evaluator = RagasEvaluator(
        ragas_metrics=[valid_metric],
        evaluator_llm=None,
        evaluator_embedding=None,
    )
    assert evaluator.metrics == [valid_metric]
    assert evaluator.llm is None
    assert evaluator.embedding is None


def test_missing_columns_in_dataset():
    """Test if RagasEvaluator raises a ValueError when required columns are missing for a specific metric."""
    evaluator = RagasEvaluator(ragas_metrics=[Faithfulness()])
    query = "Which is the most popular global sport?"
    reference = "Football is the most popular sport with around 4 billion followers worldwide"
    response = "Football is the most popular sport in the world"

    with pytest.raises(ValueError) as exc_info:
        evaluator.run(query=query, reference=reference, response=response)

    expected_error_message = "The metric [faithfulness] that is used requires the following additional columns ['documents'] to be present in the dataset."
    assert expected_error_message == str(exc_info.value)


def test_run_invalid_query_type():
    """Test RagasEvaluator raises ValueError for invalid query type."""
    evaluator = RagasEvaluator(ragas_metrics=[Faithfulness()])
    query = ["Invalid query type"]  # Should be str
    documents = ["Football is the most popular sport."]
    reference = ChatMessage(_content="Football is the most popular sport.", _role="human")
    response = "Football is the most popular sport in the world"

    with pytest.raises(ValueError, match="The 'query' field expected .* but got 'list'"):
        evaluator.run(query=query, documents=documents, reference=reference, response=response)


def test_run_invalid_rubrics_type():
    """Test RagasEvaluator raises ValueError for invalid rubrics type."""
    evaluator = RagasEvaluator(ragas_metrics=[Faithfulness()])
    query = "Which is the most popular global sport?"
    response = "Football is the most popular sport in the world"
    documents = ["Football is the most popular sport."]
    rubrics = ["score_1"]  # Should be dict

    with pytest.raises(ValueError, match="The 'rubrics' field expected 'one of Dict, NoneType', but got 'list'."):
        evaluator.run(query=query, rubrics=rubrics, response=response, documents=documents)


def test_run_invalid_documents_type():
    """Test RagasEvaluator raises ValueError for invalid document types."""
    evaluator = RagasEvaluator(ragas_metrics=[Faithfulness()])
    query = "Which is the most popular global sport?"
    documents = [123, ["Invalid document"]]  # Invalid types

    with pytest.raises(ValueError, match="Unsupported type in documents list."):
        evaluator.run(query=query, documents=documents)


@patch.object(RagasEvaluator, 'run')
def test_run_valid_input(mock_run):
    """Test RagasEvaluator runs successfully with valid input."""
    mock_run.return_value = {"result": {"score": MagicMock(), "details": MagicMock(spec=EvaluationResult)}}
    evaluator = RagasEvaluator(ragas_metrics=[MagicMock(Metric)])

    query = "Which is the most popular global sport?"
    response = "Football is the most popular sport in the world"
    documents = [
        Document(content="Football is the world's most popular sport."),
        Document(content="Football has over 4 billion followers."),
    ]
    reference_contexts = ["Football is a globally popular sport."]
    multi_responses = ["Football is considered the most popular sport."]
    reference = "Football is the most popular sport with around 4 billion followers worldwide"
    rubrics = {"accuracy": "high", "relevance": "high"}

    output = evaluator.run(
        query=query,
        response=response,
        documents=documents,
        reference_contexts=reference_contexts,
        multi_responses=multi_responses,
        reference=reference,
        rubrics=rubrics,
    )

    assert "result" in output
    assert isinstance(output["result"], dict)
    assert "score" in output["result"]
