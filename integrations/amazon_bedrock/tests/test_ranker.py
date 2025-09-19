from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockInferenceError,
)
from haystack_integrations.components.rankers.amazon_bedrock import AmazonBedrockRanker


@pytest.fixture
def mock_aws_session():
    with patch("haystack_integrations.components.rankers.amazon_bedrock.ranker.get_aws_session") as mock_session:
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client
        yield mock_client


def test_amazon_bedrock_ranker_initialization(mock_aws_session):
    ranker = AmazonBedrockRanker(
        model="cohere.rerank-v3-5:0",
        top_k=2,
        aws_access_key_id=Secret.from_token("test_access_key"),
        aws_secret_access_key=Secret.from_token("test_secret_key"),
        aws_region_name=Secret.from_token("us-west-2"),
    )
    assert ranker.model_name == "cohere.rerank-v3-5:0"
    assert ranker.top_k == 2


def test_bedrock_ranker_run(mock_aws_session):
    ranker = AmazonBedrockRanker(
        model="cohere.rerank-v3-5:0",
        top_k=2,
        aws_access_key_id=Secret.from_token("test_access_key"),
        aws_secret_access_key=Secret.from_token("test_secret_key"),
        aws_region_name=Secret.from_token("us-west-2"),
    )

    mock_response = {"results": [{"index": 0, "relevanceScore": 0.9}, {"index": 1, "relevanceScore": 0.7}]}

    mock_aws_session.rerank.return_value = mock_response

    docs = [Document(content="Test document 1"), Document(content="Test document 2")]
    result = ranker.run(query="test query", documents=docs)

    assert len(result["documents"]) == 2
    assert result["documents"][0].score == 0.9
    assert result["documents"][1].score == 0.7


# In the CI, those tests are skipped if AWS Authentication fails
@pytest.mark.integration
def test_amazon_bedrock_ranker_live_run():
    ranker = AmazonBedrockRanker(
        model="cohere.rerank-v3-5:0",
        top_k=2,
        aws_region_name=Secret.from_token("eu-central-1"),
    )

    docs = [Document(content="Test document 1"), Document(content="Test document 2")]
    result = ranker.run(query="test query", documents=docs)
    assert len(result["documents"]) == 2
    assert isinstance(result["documents"][0].score, float)


def test_amazon_bedrock_ranker_run_inference_error(mock_aws_session):
    ranker = AmazonBedrockRanker(
        model="cohere.rerank-v3-5:0",
        top_k=2,
        aws_access_key_id=Secret.from_token("test_access_key"),
        aws_secret_access_key=Secret.from_token("test_secret_key"),
        aws_region_name=Secret.from_token("us-west-2"),
    )

    mock_aws_session.rerank.side_effect = Exception("Inference error")

    docs = [Document(content="Test document 1"), Document(content="Test document 2")]
    with pytest.raises(AmazonBedrockInferenceError):
        ranker.run(query="test query", documents=docs)


def test_amazon_bedrock_ranker_serialization(mock_aws_session):
    ranker = AmazonBedrockRanker(model="cohere.rerank-v3-5:0", top_k=2)

    serialized = ranker.to_dict()
    assert serialized["init_parameters"]["model"] == "cohere.rerank-v3-5:0"
    assert serialized["init_parameters"]["top_k"] == 2

    deserialized = AmazonBedrockRanker.from_dict(serialized)
    assert isinstance(deserialized, AmazonBedrockRanker)
    assert deserialized.model_name == "cohere.rerank-v3-5:0"
    assert deserialized.top_k == 2


def test_amazon_bedrock_ranker_empty_documents(mock_aws_session):
    ranker = AmazonBedrockRanker(
        model="cohere.rerank-v3-5:0",
        top_k=2,
        aws_access_key_id=Secret.from_token("test_access_key"),
        aws_secret_access_key=Secret.from_token("test_secret_key"),
        aws_region_name=Secret.from_token("us-west-2"),
    )

    docs = []
    result = ranker.run(query="test query", documents=docs)

    # Check that we get back an empty list of documents
    assert len(result["documents"]) == 0
