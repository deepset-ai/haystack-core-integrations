import re
import pytest
from unittest.mock import MagicMock
from ragas.metrics import Metric, Faithfulness
from ragas.llms import BaseRagasLLM
from ragas.embeddings import BaseRagasEmbeddings
from langchain_core.language_models import BaseLanguageModel as LangchainLLM
from langchain_core.embeddings import Embeddings as LangchainEmbeddings
from haystack_integrations.components.evaluators.ragas import RagasEvaluator


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

    with pytest.raises(TypeError, match="Expected evaluator_embedding to be BaseRagasEmbeddings or LangchainEmbeddings"):
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
    evaluator = RagasEvaluator(
        ragas_metrics=[faithfulness_metric]
    )

    # Prepare inputs with missing required columns
    reference = "Football is the most popular sport with around 4 billion followers worldwide"
    response = "Football is the most popular sport in the world"

    # Run the evaluator and expect a ValueError
    with pytest.raises(ValueError, match=re.escape("The metric [faithfulness] that is used requires the following additional columns ['query', 'documents'] to be present in the dataset.")):
        evaluator.run(
            reference=reference,
            response=response
        )
