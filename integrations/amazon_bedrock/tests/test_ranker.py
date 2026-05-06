from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
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

    mock_response = {
        "results": [
            {"index": 0, "relevanceScore": 0.9},
            {"index": 1, "relevanceScore": 0.7},
        ]
    }

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


def test_amazon_bedrock_ranker_empty_model():
    with pytest.raises(ValueError, match="cannot be None or empty string"):
        AmazonBedrockRanker(model="")


def test_amazon_bedrock_ranker_connection_error():
    with patch(
        "haystack_integrations.components.rankers.amazon_bedrock.ranker.get_aws_session",
        side_effect=Exception("boom"),
    ):
        with pytest.raises(AmazonBedrockConfigurationError):
            AmazonBedrockRanker(aws_region_name=Secret.from_token("us-west-2"))


def test_amazon_bedrock_ranker_invalid_top_k(mock_aws_session):
    ranker = AmazonBedrockRanker(aws_region_name=Secret.from_token("us-west-2"))
    with pytest.raises(ValueError, match="top_k must be > 0"):
        ranker.run(query="q", documents=[Document(content="x")], top_k=-1)


def test_amazon_bedrock_ranker_truncates_large_input(mock_aws_session, caplog):
    ranker = AmazonBedrockRanker(aws_region_name=Secret.from_token("us-west-2"))
    mock_aws_session.rerank.return_value = {"results": []}

    docs = [Document(content=f"doc {i}") for i in range(1005)]
    ranker.run(query="q", documents=docs)

    sent_sources = mock_aws_session.rerank.call_args.kwargs["sources"]
    assert len(sent_sources) == 1000
    assert any("truncated" in record.message for record in caplog.records)


def test_amazon_bedrock_ranker_client_error(mock_aws_session):
    ranker = AmazonBedrockRanker(aws_region_name=Secret.from_token("us-west-2"))
    mock_aws_session.rerank.side_effect = ClientError(
        error_response={"Error": {"Code": "x", "Message": "y"}}, operation_name="rerank"
    )
    with pytest.raises(AmazonBedrockInferenceError, match="Could not perform inference"):
        ranker.run(query="q", documents=[Document(content="x")])


def test_amazon_bedrock_ranker_unexpected_response(mock_aws_session):
    ranker = AmazonBedrockRanker(aws_region_name=Secret.from_token("us-west-2"))
    mock_aws_session.rerank.return_value = {"unexpected_key": []}

    with pytest.raises(AmazonBedrockInferenceError, match="Unexpected response format"):
        ranker.run(query="q", documents=[Document(content="x")])


def test_amazon_bedrock_ranker_meta_fields_to_embed(mock_aws_session):
    ranker = AmazonBedrockRanker(
        aws_region_name=Secret.from_token("us-west-2"),
        meta_fields_to_embed=["title"],
        meta_data_separator=" | ",
    )
    mock_aws_session.rerank.return_value = {"results": [{"index": 0, "relevanceScore": 0.5}]}

    docs = [Document(content="body", meta={"title": "T"})]
    ranker.run(query="q", documents=docs)

    sent_text = mock_aws_session.rerank.call_args.kwargs["sources"][0]["inlineDocumentSource"]["textDocument"]["text"]
    assert sent_text == "T | body"
