# Bedrock Managed Knowledge Base Support

## Overview
Adds a Haystack retriever component that queries Amazon Bedrock Knowledge Bases for managed semantic search.

## Usage
```python
from haystack import Pipeline
from haystack_integrations.components.retrievers.amazon_bedrock import BedrockKnowledgeBaseRetriever

retriever = BedrockKnowledgeBaseRetriever(knowledge_base_id="YOUR_KB_ID")
pipeline = Pipeline()
pipeline.add_component("retriever", retriever)
result = pipeline.run({"retriever": {"query": "What is our SLA policy?"}})
docs = result["retriever"]["documents"]
```

## Configuration
| Variable | Description | Default |
|---|---|---|
| KNOWLEDGE_BASE_ID | Bedrock Knowledge Base ID | None |
| AWS_ACCESS_KEY_ID | AWS access key ID | None |
| AWS_SECRET_ACCESS_KEY | AWS secret access key | None |
| AWS_SESSION_TOKEN | AWS session token | None |
| AWS_DEFAULT_REGION | AWS region for the KB | None |
| AWS_PROFILE | AWS credentials profile | None |
| USE_AGENTIC_RETRIEVAL | Enable agentic retrieval | true |

## Features
- Managed search (no vector store needed)
- Agentic retrieval with query decomposition + reranking
- Automatic fallback to plain Retrieve if agentic fails
- Multi-source support (S3, Web, Confluence, SharePoint)
- Returns Haystack Document objects with metadata and scores

## SDK Requirements
- boto3 >= 1.43
- haystack-ai >= 2.0

## Required IAM Permissions
```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:Retrieve",
    "bedrock:AgenticRetrieveStream"
  ],
  "Resource": "arn:aws:bedrock:<region>:<account-id>:knowledge-base/<kb-id>"
}
```

## References
- [Build a Managed Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-build-managed.html)
- [Retrieve API](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-retrieve.html)
- [Agentic Retrieval](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-agentic.html)
