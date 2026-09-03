# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.components.retrievers.amazon_bedrock.knowledge_base_retriever import (
    AmazonBedrockKnowledgeBaseRetriever,
)


@pytest.fixture
def mock_aws_session():
    with patch(
        "haystack_integrations.components.retrievers.amazon_bedrock.knowledge_base_retriever.get_aws_session"
    ) as mock_session:
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client
        yield mock_client


class TestAmazonBedrockKnowledgeBaseRetriever:
    def test_init_defaults(self):
        retriever = AmazonBedrockKnowledgeBaseRetriever(knowledge_base_id="TEST123456")
        assert retriever.knowledge_base_id == "TEST123456"
        assert retriever.number_of_results == 5
        assert retriever.knowledge_base_type == "MANAGED"
        assert retriever._client is None

    @patch.dict("os.environ", {"AWS_KNOWLEDGE_BASE_ID": "ENV_KB", "AWS_DEFAULT_REGION": "eu-west-1"})
    def test_init_from_env(self):
        retriever = AmazonBedrockKnowledgeBaseRetriever()
        assert retriever.knowledge_base_id == "ENV_KB"
        assert retriever.aws_region_name.resolve_value() == "eu-west-1"

    def test_run_managed(self, mock_aws_session):
        mock_aws_session.retrieve.return_value = {
            "retrievalResults": [
                {
                    "content": {"text": "Managed KB handles everything automatically."},
                    "location": {"s3Location": {"uri": "s3://bucket/doc.pdf"}},
                    "score": 0.95,
                },
                {
                    "content": {"text": "No vector store needed."},
                    "location": {"s3Location": {"uri": "s3://bucket/doc2.pdf"}},
                    "score": 0.87,
                },
            ]
        }

        retriever = AmazonBedrockKnowledgeBaseRetriever(knowledge_base_id="TEST123456")
        result = retriever.run(query="What is managed KB?")

        # Verify correct API call
        mock_aws_session.retrieve.assert_called_once()
        call_kwargs = mock_aws_session.retrieve.call_args.kwargs
        assert call_kwargs["knowledgeBaseId"] == "TEST123456"
        assert "managedSearchConfiguration" in call_kwargs["retrievalConfiguration"]

        # Verify documents returned
        assert "documents" in result
        docs = result["documents"]
        assert len(docs) == 2
        assert isinstance(docs[0], Document)
        assert docs[0].content == "Managed KB handles everything automatically."
        assert docs[0].meta["source"] == "s3://bucket/doc.pdf"
        assert docs[0].score == 0.95

    def test_run_top_k_override(self, mock_aws_session):
        mock_aws_session.retrieve.return_value = {"retrievalResults": []}

        retriever = AmazonBedrockKnowledgeBaseRetriever(knowledge_base_id="TEST123456", number_of_results=5)
        retriever.run(query="test", top_k=10)

        call_kwargs = mock_aws_session.retrieve.call_args.kwargs
        assert call_kwargs["retrievalConfiguration"]["managedSearchConfiguration"]["numberOfResults"] == 10

    def test_run_empty_results(self, mock_aws_session):
        mock_aws_session.retrieve.return_value = {"retrievalResults": []}

        retriever = AmazonBedrockKnowledgeBaseRetriever(knowledge_base_id="TEST123456")
        result = retriever.run(query="no match")

        assert result["documents"] == []

    def test_run_error_handling(self, mock_aws_session):
        mock_aws_session.retrieve.side_effect = Exception("Service unavailable")

        retriever = AmazonBedrockKnowledgeBaseRetriever(knowledge_base_id="TEST123456")

        with pytest.raises(AmazonBedrockInferenceError):
            retriever.run(query="test")

    def test_to_dict(self):
        retriever = AmazonBedrockKnowledgeBaseRetriever(
            knowledge_base_id="TEST123456",
            number_of_results=10,
        )
        serialized = retriever.to_dict()

        assert serialized == {
            "type": (
                "haystack_integrations.components.retrievers.amazon_bedrock."
                "knowledge_base_retriever.AmazonBedrockKnowledgeBaseRetriever"
            ),
            "init_parameters": {
                "knowledge_base_id": "TEST123456",
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {
                    "type": "env_var",
                    "env_vars": ["AWS_SECRET_ACCESS_KEY"],
                    "strict": False,
                },
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "number_of_results": 10,
                "use_agentic_retrieval": True,
            },
        }

    def test_from_dict(self):
        data = {
            "type": (
                "haystack_integrations.components.retrievers.amazon_bedrock."
                "knowledge_base_retriever.AmazonBedrockKnowledgeBaseRetriever"
            ),
            "init_parameters": {
                "knowledge_base_id": "TEST123456",
                "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
                "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
                "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
                "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
                "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
                "number_of_results": 10,
                "use_agentic_retrieval": False,
            },
        }
        retriever = AmazonBedrockKnowledgeBaseRetriever.from_dict(data)
        assert retriever.knowledge_base_id == "TEST123456"
        assert retriever.number_of_results == 10
        assert retriever.use_agentic_retrieval is False


class TestComponentLifecycle:
    def test_warm_up_uses_resolved_credentials(self, mock_boto3_session, set_env_variables):
        retriever = AmazonBedrockKnowledgeBaseRetriever(knowledge_base_id="kb")
        retriever.warm_up()
        mock_boto3_session.assert_called_once_with(
            aws_access_key_id="some_fake_id",
            aws_secret_access_key="some_fake_key",
            aws_session_token="some_fake_token",
            region_name="fake_region",
            profile_name="some_fake_profile",
        )

    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("MISSING_AWS_ACCESS_KEY", raising=False)
        retriever = AmazonBedrockKnowledgeBaseRetriever(
            knowledge_base_id="kb",
            aws_access_key_id=Secret.from_env_var("MISSING_AWS_ACCESS_KEY"),
        )
        with pytest.raises(AmazonBedrockConfigurationError):
            retriever.warm_up()

    def test_sync_lifecycle(self, mock_boto3_session):
        retriever = AmazonBedrockKnowledgeBaseRetriever(knowledge_base_id="kb")
        client = mock_boto3_session.return_value.client.return_value
        retriever.warm_up()
        assert retriever._client is client
        retriever.close()
        client.close.assert_called_once_with()
        assert retriever._client is None
        retriever.warm_up()
        assert mock_boto3_session.call_count == 2

    def test_warm_up_is_idempotent(self, mock_boto3_session):
        retriever = AmazonBedrockKnowledgeBaseRetriever(knowledge_base_id="kb")
        retriever.warm_up()
        retriever.warm_up()
        mock_boto3_session.assert_called_once()

    def test_close_is_safe_without_warm_up(self):
        retriever = AmazonBedrockKnowledgeBaseRetriever(knowledge_base_id="kb")
        retriever.close()
        assert retriever._client is None


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("AWS_KNOWLEDGE_BASE_ID"),
    reason="Set AWS_KNOWLEDGE_BASE_ID env var to run integration tests",
)
class TestAmazonBedrockKnowledgeBaseRetrieverIntegration:
    """Integration tests for AmazonBedrockKnowledgeBaseRetriever against a live managed KB."""

    def test_standard_retrieve(self):
        """Test standard Retrieve API on a managed knowledge base."""
        retriever = AmazonBedrockKnowledgeBaseRetriever(
            knowledge_base_id=os.environ["AWS_KNOWLEDGE_BASE_ID"],
            aws_region_name=os.environ["AWS_REGION"],
            use_agentic_retrieval=False,
            number_of_results=3,
        )

        result = retriever.run(query="What is Amazon Bedrock?")
        documents = result["documents"]

        assert len(documents) > 0
        assert all(doc.content for doc in documents)
        assert all(doc.score is not None for doc in documents)

    def test_agentic_retrieve(self):
        """Test AgenticRetrieveStream on a managed knowledge base."""
        retriever = AmazonBedrockKnowledgeBaseRetriever(
            knowledge_base_id=os.environ["AWS_KNOWLEDGE_BASE_ID"],
            aws_region_name=os.environ["AWS_REGION"],
            use_agentic_retrieval=True,
            number_of_results=5,
        )

        result = retriever.run(query="What is Amazon Bedrock managed knowledge base?")
        documents = result["documents"]

        assert len(documents) > 0
        assert all(doc.content for doc in documents)

    def test_user_agent(self):
        """Verify user-agent header is set correctly."""
        retriever = AmazonBedrockKnowledgeBaseRetriever(
            knowledge_base_id=os.environ["AWS_KNOWLEDGE_BASE_ID"],
            aws_region_name=os.environ["AWS_REGION"],
        )
        retriever.warm_up()

        ua = retriever._client._client_config.user_agent_extra
        assert "x-client-framework:haystack" in ua
