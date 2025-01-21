import re
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


# Test fixture to mock the 'run' method of RagasEvaluator
@pytest.fixture
def mock_run():
    with mock.patch.object(RagasEvaluator, 'run') as mock_method:
        yield mock_method


def test_successful_initialization():
    """Test if the RagasEvaluator initializes correctly with valid inputs."""
    valid_metric_1 = MagicMock(spec=Metric)
    valid_metric_2 = MagicMock(spec=Metric)
    valid_metric_3 = MagicMock(spec=Metric)
    valid_llm = MagicMock(spec=BaseRagasLLM)
    valid_embedding = MagicMock(spec=BaseRagasEmbeddings)

    evaluator = RagasEvaluator(
        ragas_metrics=[valid_metric_1, valid_metric_2, valid_metric_3],
        evaluator_llm=valid_llm,
        evaluator_embedding=valid_embedding,
    )
    assert evaluator.metrics == [valid_metric_1, valid_metric_2, valid_metric_3]
    assert evaluator.llm == valid_llm
    assert evaluator.embedding == valid_embedding


def test_invalid_metrics():
    """Test if RagasEvaluator raises TypeError for invalid metrics."""
    invalid_metric = "not_a_metric"

    with pytest.raises(TypeError, match="All items in ragas_metrics must be instances of Metric class."):
        RagasEvaluator(ragas_metrics=[invalid_metric])


def test_invalid_llm():
    """Test if RagasEvaluator raises TypeError for invalid evaluator_llm."""
    valid_metric = MagicMock(spec=Metric)
    invalid_llm = "not_a_llm"

    with pytest.raises(TypeError, match="Expected evaluator_llm to be BaseRagasLLM or LangchainLLM"):
        RagasEvaluator(
            ragas_metrics=[valid_metric],
            evaluator_llm=invalid_llm,
        )


def test_invalid_embedding():
    """Test if RagasEvaluator raises TypeError for invalid evaluator_embedding."""
    valid_metric = MagicMock(spec=Metric)
    invalid_embedding = "not_an_embedding"

    with pytest.raises(
        TypeError, match="Expected evaluator_embedding to be BaseRagasEmbeddings or LangchainEmbeddings"
    ):
        RagasEvaluator(
            ragas_metrics=[valid_metric],
            evaluator_embedding=invalid_embedding,
        )


def test_optional_llm_and_embeddings():
    """Test if RagasEvaluator initializes correctly with None for optional parameters."""
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
    """
    Test if RagasEvaluator raises a ValueError when required columns
    are missing for a specific metric.
    """
    faithfulness_metric = Faithfulness()

    # Initialize RagasEvaluator with the mocked metric
    evaluator = RagasEvaluator(ragas_metrics=[faithfulness_metric])

    # Prepare inputs with missing required columns
    query = "Which is the most popular global sport?"
    reference = "Football is the most popular sport with around 4 billion followers worldwide"
    response = "Football is the most popular sport in the world"

    # Run the evaluator and expect a ValueError
    with pytest.raises(ValueError) as exc_info:
        evaluator.run(query=query, reference=reference, response=response)
    # Check if the error message contains the expected text
    expected_error_message = "The metric [faithfulness] that is used requires the following additional columns ['documents'] to be present in the dataset."
    actual_error_message = str(exc_info.value)

    # Sort lists and compare, if order of 'query' and 'documents' is not guaranteed
    assert expected_error_message == actual_error_message


def test_run_invalid_query_type():
    evaluator = RagasEvaluator(ragas_metrics=[Faithfulness()])

    query = ["Which is the most popular global sport?"]  # Invalid type: list instead of str
    documents = [
        "Football is undoubtedly the world's most popular sport with major events like the FIFA World Cup and sports personalities like Ronaldo and Messi, drawing a followership of more than 4 billion people."
    ]
    reference = ChatMessage(
        _content="Football is the most popular sport with around 4 billion followers worldwide", _role="human"
    )
    response = "Football is the most popular sport in the world"

    with pytest.raises(ValueError) as exc_info:
        evaluator.run(
            query=query,  # This will raise a validation error
            documents=documents,
            reference=reference,
            response=response,
        )

    # Check that the error message contains the expected validation hint
    assert "The 'query' field expected 'one of str, NoneType', but got 'list'" in str(exc_info.value)


def test_run_invalid_rubrics_type():
    evaluator = RagasEvaluator(ragas_metrics=[Faithfulness()])

    query = "Which is the most popular global sport?"
    # Pass ChatMessage object instead of a dictionary for rubrics
    rubrics = ChatMessage(
        _content="Football is the most popular sport with around 4 billion followers worldwide", _role="human"
    )

    with pytest.raises(ValueError) as exc_info:
        evaluator.run(
            query=query,
            rubrics=rubrics,  # This will raise a validation error
        )

    # Check that the error message contains the expected validation hint
    assert "The 'rubrics' field expected 'one of Dict, NoneType', but got 'ChatMessage'" in str(exc_info.value)


def test_run_invalid_documents_type():
    evaluator = RagasEvaluator(ragas_metrics=[Faithfulness()])

    query = "Which is the most popular global sport?"
    # Invalid types: passing an integer and a list instead of valid documents
    documents = [123, ["invalid_doc"]]  # Invalid types

    with pytest.raises(ValueError) as exc_info:
        evaluator.run(
            query=query,
            documents=documents,  # This will raise a validation error
        )

    assert "Unsupported type in documents list." in str(exc_info.value)


# Test case to check valid input for RagasEvaluator's run method
def test_run_valid_input():
    # Set up the mock return value

    mock_run.return_value = {"result": {MagicMock(spec=EvaluationResult)}}

    # Create the evaluator instance
    evaluator = RagasEvaluator(ragas_metrics=[MagicMock(Metric)])

    # Define a valid input
    query = "Which is the most popular global sport?"
    response = "Football is the most popular sport in the world"

    documents = [
        Document(
            content="Football is undoubtedly the world's most popular sport with major events like the FIFA World Cup."
        ),
        Document(content="Football's popularity continues to grow globally, with over 4 billion followers."),
    ]

    reference_contexts = ["Football is a globally popular sport with a huge fan base worldwide."]
    multi_responses = ["Football is considered the most popular sport in the world."]
    reference = "Football is the most popular sport with around 4 billion followers worldwide"
    rubrics = {"accuracy": "high", "relevance": "high"}

    # Run the evaluator with valid input
    output = evaluator.run(
        query=query,
        response=response,
        documents=documents,
        reference_contexts=reference_contexts,
        multi_responses=multi_responses,
        reference=reference,
        rubrics=rubrics,
    )

    # Verify that the output contains a 'result' key
    assert "result" in output

    # Optionally verify specific details in the result, based on the expected behavior
    # For example, you could check the type of the result or assert some properties of the result
    assert isinstance(output["result"], dict)  # Assuming the result is a dictionary
    assert "score" in output["result"]  # Assuming the result has a "score" key
