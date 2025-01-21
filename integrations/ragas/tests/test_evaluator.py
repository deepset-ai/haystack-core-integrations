import pytest
from unittest.mock import MagicMock, patch
from ragas.metrics import ContextPrecision
from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings
from haystack.dataclasses import ChatMessage
from haystack import Document
from ragas.dataset_schema import EvaluationResult
from haystack_integrations.components.evaluators.ragas.evaluator import RagasEvaluator


@pytest.fixture
def evaluator():
    """Fixture to create a RagasEvaluator instance for testing."""
    llm = MagicMock(spec=BaseRagasLLM)
    metrics = [ContextPrecision()]
    return RagasEvaluator(ragas_metrics=metrics, evaluator_llm=llm)





def test_run_valid_input(evaluator):
    """Test the run method with valid input."""
    documents = [Document(content="Football is the most popular sport.")]
    response = [ChatMessage(_content=["Football is the most popular sport."], _role="assistant")]

    result = evaluator.run(
        query="Which is the most popular global sport?",
        response=response,
        documents=documents,
        reference="Football is the most popular sport.",
    )

    assert "result" in result
    assert isinstance(result["result"], EvaluationResult)


def test_run_invalid_documents_type(evaluator):
    """Test the run method with invalid document types."""
    with pytest.raises(ValueError, match="All elements in the documents list must be of type Document."):
        evaluator.run(
            query="Which is the most popular global sport?",
            documents=["Not a Document"],
        )


def test_run_empty_documents(evaluator):
    """Test the run method with empty documents."""
    with pytest.raises(ValueError, match="Documents must be provided for evaluation."):
        evaluator.run(
            query="Which is the most popular global sport?",
            documents=[],
        )


def test_run_invalid_response_type(evaluator):
    """Test the run method with invalid response type."""
    documents = [Document(content="Football is the most popular sport.")]

    with pytest.raises(ValueError, match="All elements in the documents list must be strings."):
        evaluator.run(
            query="Which is the most popular global sport?",
            response=["Not a ChatMessage"],
            documents=documents,
        )


def test_run_validation_error(evaluator):
    """Test the run method raises validation error."""
    documents = [Document(content="Football is the most popular sport.")]
    response = [ChatMessage(_content=["Football is the most popular sport."], _role="assistant")]

    # Mocking the SingleTurnSample to raise ValidationError
    with patch('ragas.dataset_schema.SingleTurnSample') as mock_sample:
        mock_sample.side_effect = Exception("Validation error")

        with pytest.raises(ValueError, match="Validation error in RagasEvaluator Component:"):
            evaluator.run(
                query="Which is the most popular global sport?",
                response=response,
                documents=documents,
            )
