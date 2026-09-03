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


def test_amazon_bedrock_ranker_initialization():
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


def test_amazon_bedrock_ranker_serialization():
    ranker = AmazonBedrockRanker(model="cohere.rerank-v3-5:0", top_k=2)

    serialized = ranker.to_dict()
    assert serialized["init_parameters"]["model"] == "cohere.rerank-v3-5:0"
    assert serialized["init_parameters"]["top_k"] == 2

    deserialized = AmazonBedrockRanker.from_dict(serialized)
    assert isinstance(deserialized, AmazonBedrockRanker)
    assert deserialized.model_name == "cohere.rerank-v3-5:0"
    assert deserialized.top_k == 2


def test_from_dict_aws_region_name():
    """
    Test that aws_region_name as str value is correctly parsed
    """
    ranker = AmazonBedrockRanker.from_dict(
        {
            "type": "haystack_integrations.components.rankers.amazon_bedrock.ranker.AmazonBedrockRanker",
            "init_parameters": {
                "aws_region_name": "my-fake-region",
                "model": "cohere.rerank-v3-5:0",
            },
        }
    )
    assert ranker.model_name == "cohere.rerank-v3-5:0"
    assert ranker.aws_region_name == "my-fake-region"

    serialized = ranker.to_dict()
    assert serialized["init_parameters"]["aws_region_name"] == "my-fake-region"


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


def test_amazon_bedrock_ranker_invalid_top_k(mock_aws_session):
    ranker = AmazonBedrockRanker(aws_region_name=Secret.from_token("us-west-2"))
    with pytest.raises(ValueError, match="top_k must be > 0"):
        ranker.run(query="q", documents=[Document(content="x")], top_k=-1)


@pytest.mark.parametrize("top_k", [0, -1])
def test_amazon_bedrock_ranker_init_invalid_top_k(top_k):
    with pytest.raises(ValueError, match=rf"top_k must be > 0, but got {top_k}"):
        AmazonBedrockRanker(aws_region_name=Secret.from_token("us-west-2"), top_k=top_k)


def test_amazon_bedrock_ranker_run_zero_top_k(mock_aws_session):
    ranker = AmazonBedrockRanker(aws_region_name=Secret.from_token("us-west-2"))
    with pytest.raises(ValueError, match="top_k must be > 0, but got 0"):
        ranker.run(query="q", documents=[Document(content="x")], top_k=0)


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


class TestComponentLifecycle:
    def test_client_is_none_after_init(self):
        ranker = AmazonBedrockRanker()
        assert ranker._bedrock_client is None

    def test_warm_up_uses_resolved_credentials(self, mock_boto3_session, set_env_variables):
        ranker = AmazonBedrockRanker()
        ranker.warm_up()
        mock_boto3_session.assert_called_once_with(
            aws_access_key_id="some_fake_id",
            aws_secret_access_key="some_fake_key",
            aws_session_token="some_fake_token",
            region_name="fake_region",
            profile_name="some_fake_profile",
        )

    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("MISSING_AWS_ACCESS_KEY", raising=False)
        ranker = AmazonBedrockRanker(
            aws_access_key_id=Secret.from_env_var("MISSING_AWS_ACCESS_KEY"),
        )
        with pytest.raises(AmazonBedrockConfigurationError):
            ranker.warm_up()

    def test_warm_up_connection_error(self, mock_boto3_session):
        mock_boto3_session.side_effect = Exception("connection error")
        ranker = AmazonBedrockRanker()
        with pytest.raises(AmazonBedrockConfigurationError):
            ranker.warm_up()

    def test_sync_lifecycle(self, mock_boto3_session):
        ranker = AmazonBedrockRanker()
        client = mock_boto3_session.return_value.client.return_value
        ranker.warm_up()
        assert ranker._bedrock_client is client
        ranker.close()
        client.close.assert_called_once_with()
        assert ranker._bedrock_client is None
        ranker.warm_up()
        assert mock_boto3_session.call_count == 2

    def test_warm_up_is_idempotent(self, mock_boto3_session):
        ranker = AmazonBedrockRanker()
        ranker.warm_up()
        ranker.warm_up()
        mock_boto3_session.assert_called_once()

    def test_close_is_safe_without_warm_up(self):
        ranker = AmazonBedrockRanker()
        ranker.close()
        assert ranker._bedrock_client is None
