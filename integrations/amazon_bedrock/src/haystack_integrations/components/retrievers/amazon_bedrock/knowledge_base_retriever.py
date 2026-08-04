# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

from botocore.config import Config
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

logger = logging.getLogger(__name__)


def _get_source_uri(result: dict) -> str:
    """Extract source URI from a retrieval result, handling all location types."""
    location = result.get("location", {})
    loc_type = location.get("type", "")
    if loc_type == "S3" or "s3Location" in location:
        return location.get("s3Location", {}).get("uri", "")
    elif loc_type == "WEB" or "webLocation" in location:
        return location.get("webLocation", {}).get("url", "")
    elif "confluenceLocation" in location:
        return location.get("confluenceLocation", {}).get("url", "")
    elif "salesforceLocation" in location:
        return location.get("salesforceLocation", {}).get("url", "")
    elif "sharePointLocation" in location:
        return location.get("sharePointLocation", {}).get("url", "")
    elif "customDocumentLocation" in location:
        return location.get("customDocumentLocation", {}).get("id", "")
    # Fallback to metadata._source_uri (for agentic results)
    return result.get("metadata", {}).get("_source_uri", "")


@component
class BedrockKnowledgeBaseRetriever:
    """
    Retrieves documents from an Amazon Bedrock Managed Knowledge Base.

    Uses AgenticRetrieveStream when available, falling back to the standard Retrieve API otherwise.

    Usage example:
    ```python
    from haystack.utils import Secret
    from haystack_integrations.components.retrievers.amazon_bedrock import BedrockKnowledgeBaseRetriever

    retriever = BedrockKnowledgeBaseRetriever(
        knowledge_base_id="ABCDEFGHIJ",
        aws_region_name=Secret.from_token("eu-central-1"),
    )

    result = retriever.run(query="What are the benefits of managed knowledge bases?")
    for doc in result["documents"]:
        print(doc.content)
        print(doc.meta["source"])
        print(doc.score)
    ```

    BedrockKnowledgeBaseRetriever uses AWS for authentication. You can use the AWS CLI to authenticate through
    your IAM. For more information on setting up an IAM identity-based policy, see [Amazon Bedrock documentation]
    (https://docs.aws.amazon.com/bedrock/latest/userguide/security_iam_id-based-policy-examples.html).

    If the AWS environment is configured correctly, the AWS credentials are not required as they're loaded
    automatically from the environment or the AWS configuration file.
    If the AWS environment is not configured, set `aws_access_key_id`, `aws_secret_access_key`,
    and `aws_region_name` as environment variables or pass them as
    [Secret](https://docs.haystack.deepset.ai/docs/secret-management) arguments.
    """

    def __init__(
        self,
        knowledge_base_id: str | None = None,
        aws_access_key_id: Secret | None = Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False),  # noqa: B008
        aws_secret_access_key: Secret | None = Secret.from_env_var(  # noqa: B008
            "AWS_SECRET_ACCESS_KEY", strict=False
        ),
        aws_session_token: Secret | None = Secret.from_env_var("AWS_SESSION_TOKEN", strict=False),  # noqa: B008
        aws_region_name: Secret | str | None = Secret.from_env_var("AWS_DEFAULT_REGION", strict=False),  # noqa: B008
        aws_profile_name: Secret | None = Secret.from_env_var("AWS_PROFILE", strict=False),  # noqa: B008
        number_of_results: int = 5,
        use_agentic_retrieval: bool | None = None,
    ) -> None:
        """
        Create the BedrockKnowledgeBaseRetriever component.

        :param knowledge_base_id: The ID of the Bedrock Knowledge Base. Falls back to the KNOWLEDGE_BASE_ID
            environment variable.
        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param number_of_results: Maximum number of results to return.
        :param use_agentic_retrieval: If True, try AgenticRetrieveStream before plain Retrieve.
            Defaults to the USE_AGENTIC_RETRIEVAL environment variable, or True.
        """
        self.knowledge_base_id = knowledge_base_id or os.environ.get("KNOWLEDGE_BASE_ID", "")
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region_name = aws_region_name
        self.aws_profile_name = aws_profile_name
        self.number_of_results = number_of_results
        self.knowledge_base_type = "MANAGED"
        self.use_agentic_retrieval = (
            use_agentic_retrieval
            if use_agentic_retrieval is not None
            else os.environ.get("USE_AGENTIC_RETRIEVAL", "true").lower() != "false"
        )

        def resolve_secret(secret: Secret | str | None) -> str | None:
            return secret.resolve_value() if isinstance(secret, Secret) else secret

        try:
            session = get_aws_session(
                aws_access_key_id=resolve_secret(aws_access_key_id),
                aws_secret_access_key=resolve_secret(aws_secret_access_key),
                aws_session_token=resolve_secret(aws_session_token),
                aws_region_name=resolve_secret(aws_region_name),
                aws_profile_name=resolve_secret(aws_profile_name),
            )
            self._client = session.client(
                "bedrock-agent-runtime",
                config=Config(user_agent_extra="x-client-framework:haystack"),
            )
        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

    @component.output_types(documents=list[Document])
    def run(self, query: str, top_k: int | None = None) -> dict[str, list[Document]]:
        """
        Retrieve documents from the Bedrock Knowledge Base.

        :param query: The search query.
        :param top_k: Maximum number of results. Overrides number_of_results if provided.
        :returns: A dictionary with a "documents" key containing the retrieved Documents.
        :raises AmazonBedrockInferenceError: If the retrieval call fails.
        """
        k = top_k or self.number_of_results

        # Try agentic retrieval first
        if self.use_agentic_retrieval:
            try:
                response = self._client.agentic_retrieve_stream(
                    messages=[{"content": {"text": query}, "role": "user"}],
                    generateResponse=False,
                    retrievers=[
                        {
                            "configuration": {
                                "knowledgeBase": {
                                    "knowledgeBaseId": self.knowledge_base_id,
                                    "retrievalOverrides": {"maxNumberOfResults": k},
                                }
                            }
                        }
                    ],
                    agenticRetrieveConfiguration={
                        "foundationModelType": "MANAGED",
                        "rerankingModelType": "MANAGED",
                    },
                )
                documents = []
                for event in response.get("stream", []):
                    if "result" in event and "results" in event["result"]:
                        for result in event["result"]["results"]:
                            content = result.get("content", {}).get("text", "")
                            source = _get_source_uri(result)
                            score = result.get("score")
                            doc = Document(
                                content=content,
                                meta={
                                    "source": source,
                                    "knowledge_base_id": self.knowledge_base_id,
                                    "knowledge_base_type": self.knowledge_base_type,
                                },
                                score=score,
                            )
                            documents.append(doc)
                if documents:
                    return {"documents": documents}
            except Exception:
                logger.debug("Agentic retrieval unavailable, falling back to plain retrieve")

        retrieval_config: dict[str, Any] = {"managedSearchConfiguration": {"numberOfResults": k}}

        try:
            response = self._client.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration=retrieval_config,
            )
        except Exception as exception:
            msg = (
                f"Could not retrieve documents from Amazon Bedrock Knowledge Base "
                f"'{self.knowledge_base_id}' due to:\n{exception}"
            )
            raise AmazonBedrockInferenceError(msg) from exception

        documents = []
        for result in response.get("retrievalResults", []):
            content = result.get("content", {}).get("text", "")
            source = _get_source_uri(result)
            score = result.get("score", 0.0)

            doc = Document(
                content=content,
                meta={
                    "source": source,
                    "knowledge_base_id": self.knowledge_base_id,
                    "knowledge_base_type": self.knowledge_base_type,
                },
                score=score,
            )
            documents.append(doc)

        return {"documents": documents}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            knowledge_base_id=self.knowledge_base_id,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            aws_region_name=self.aws_region_name,
            aws_profile_name=self.aws_profile_name,
            number_of_results=self.number_of_results,
            use_agentic_retrieval=self.use_agentic_retrieval,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BedrockKnowledgeBaseRetriever":
        """
        Deserializes the component from a dictionary.

        :param data: The dictionary to deserialize from.
        :returns: The deserialized component.
        """
        return default_from_dict(cls, data)
