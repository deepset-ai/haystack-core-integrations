# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from haystack_integrations.components.retrievers.amazon_bedrock.knowledge_base_retriever import (
    BedrockKnowledgeBaseRetriever,
)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("KNOWLEDGE_BASE_ID"),
    reason="Set KNOWLEDGE_BASE_ID env var to run integration tests",
)
class TestBedrockKnowledgeBaseRetrieverIntegration:
    """Integration tests for BedrockKnowledgeBaseRetriever against a live managed KB."""

    def test_standard_retrieve(self):
        """Test standard Retrieve API on a managed knowledge base."""
        retriever = BedrockKnowledgeBaseRetriever(
            knowledge_base_id=os.environ["KNOWLEDGE_BASE_ID"],
            region_name=os.environ.get("AWS_REGION", "us-west-2"),
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
        retriever = BedrockKnowledgeBaseRetriever(
            knowledge_base_id=os.environ["KNOWLEDGE_BASE_ID"],
            region_name=os.environ.get("AWS_REGION", "us-west-2"),
            use_agentic_retrieval=True,
            number_of_results=5,
        )

        result = retriever.run(query="What is Amazon Bedrock managed knowledge base?")
        documents = result["documents"]

        assert len(documents) > 0
        assert all(doc.content for doc in documents)

    def test_user_agent(self):
        """Verify user-agent header is set correctly."""
        retriever = BedrockKnowledgeBaseRetriever(
            knowledge_base_id=os.environ["KNOWLEDGE_BASE_ID"],
            region_name=os.environ.get("AWS_REGION", "us-west-2"),
        )

        # Trigger client creation
        retriever.run(query="test")

        ua = retriever._client._client_config.user_agent_extra
        assert "haystack/bedrock-kb" in ua
