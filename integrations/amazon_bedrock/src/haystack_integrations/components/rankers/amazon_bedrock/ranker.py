from typing import Any, Dict, List, Optional

from botocore.exceptions import ClientError
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.common.amazon_bedrock.errors import (
    AmazonBedrockConfigurationError,
    AmazonBedrockInferenceError,
)
from haystack_integrations.common.amazon_bedrock.utils import get_aws_session

logger = logging.getLogger(__name__)

MAX_NUM_DOCS_FOR_BEDROCK_RANKER = 1000


@component
class AmazonBedrockRanker:
    """
    Ranks Documents based on their similarity to the query using Amazon Bedrock's Cohere Rerank model.

    Documents are indexed from most to least semantically relevant to the query.

    Supported Amazon Bedrock models:
    - cohere.rerank-v3-5:0
    - amazon.rerank-v1:0

    Usage example:
    ```python
    from haystack import Document
    from haystack.utils import Secret
    from haystack_integrations.components.rankers.amazon_bedrock import AmazonBedrockRanker

    ranker = AmazonBedrockRanker(
        model="cohere.rerank-v3-5:0",
        top_k=2,
        aws_region_name=Secret.from_token("eu-central-1")
    )

    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "What is the capital of germany?"
    output = ranker.run(query=query, documents=docs)
    docs = output["documents"]
    ```

    AmazonBedrockRanker uses AWS for authentication. You can use the AWS CLI to authenticate through your IAM.
    For more information on setting up an IAM identity-based policy, see [Amazon Bedrock documentation]
    (https://docs.aws.amazon.com/bedrock/latest/userguide/security_iam_id-based-policy-examples.html).

    If the AWS environment is configured correctly, the AWS credentials are not required as they're loaded
    automatically from the environment or the AWS configuration file.
    If the AWS environment is not configured, set `aws_access_key_id`, `aws_secret_access_key`,
    and `aws_region_name` as environment variables or pass them as
    [Secret](https://docs.haystack.deepset.ai/v2.0/docs/secret-management) arguments. Make sure the region you set
    supports Amazon Bedrock.
    """

    def __init__(
        self,
        model: str = "cohere.rerank-v3-5:0",
        top_k: int = 10,
        aws_access_key_id: Optional[Secret] = Secret.from_env_var(["AWS_ACCESS_KEY_ID"], strict=False),  # noqa: B008
        aws_secret_access_key: Optional[Secret] = Secret.from_env_var(  # noqa: B008
            ["AWS_SECRET_ACCESS_KEY"], strict=False
        ),
        aws_session_token: Optional[Secret] = Secret.from_env_var(["AWS_SESSION_TOKEN"], strict=False),  # noqa: B008
        aws_region_name: Optional[Secret] = Secret.from_env_var(["AWS_DEFAULT_REGION"], strict=False),  # noqa: B008
        aws_profile_name: Optional[Secret] = Secret.from_env_var(["AWS_PROFILE"], strict=False),  # noqa: B008
        max_chunks_per_doc: Optional[int] = None,
        meta_fields_to_embed: Optional[List[str]] = None,
        meta_data_separator: str = "\n",
    ) -> None:
        if not model:
            msg = "'model' cannot be None or empty string"
            raise ValueError(msg)
        """
        Creates an instance of the 'AmazonBedrockRanker'.

        :param model: Amazon Bedrock model name for Cohere Rerank. Default is "cohere.rerank-v3-5:0".
        :param top_k: The maximum number of documents to return.
        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param max_chunks_per_doc: If your document exceeds 512 tokens, this determines the maximum number of
            chunks a document can be split into. If `None`, the default of 10 is used.
            Note: This parameter is not currently used in the implementation but is included for future compatibility.
        :param meta_fields_to_embed: List of meta fields that should be concatenated
            with the document content for reranking.
        :param meta_data_separator: Separator used to concatenate the meta fields
            to the Document content.
        """
        self.model_name = model
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region_name = aws_region_name
        self.aws_profile_name = aws_profile_name
        self.top_k = top_k
        self.max_chunks_per_doc = max_chunks_per_doc
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.meta_data_separator = meta_data_separator

        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None

        try:
            session = get_aws_session(
                aws_access_key_id=resolve_secret(aws_access_key_id),
                aws_secret_access_key=resolve_secret(aws_secret_access_key),
                aws_session_token=resolve_secret(aws_session_token),
                aws_region_name=resolve_secret(aws_region_name),
                aws_profile_name=resolve_secret(aws_profile_name),
            )
            self._bedrock_client = session.client("bedrock-agent-runtime")
        except Exception as exception:
            msg = (
                "Could not connect to Amazon Bedrock. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AmazonBedrockConfigurationError(msg) from exception

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model_name,
            aws_access_key_id=self.aws_access_key_id.to_dict() if self.aws_access_key_id else None,
            aws_secret_access_key=self.aws_secret_access_key.to_dict() if self.aws_secret_access_key else None,
            aws_session_token=self.aws_session_token.to_dict() if self.aws_session_token else None,
            aws_region_name=self.aws_region_name.to_dict() if self.aws_region_name else None,
            aws_profile_name=self.aws_profile_name.to_dict() if self.aws_profile_name else None,
            top_k=self.top_k,
            max_chunks_per_doc=self.max_chunks_per_doc,
            meta_fields_to_embed=self.meta_fields_to_embed,
            meta_data_separator=self.meta_data_separator,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmazonBedrockRanker":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_secrets_inplace(
            data["init_parameters"],
            ["aws_access_key_id", "aws_secret_access_key", "aws_session_token", "aws_region_name", "aws_profile_name"],
        )
        return default_from_dict(cls, data)

    def _prepare_bedrock_input_docs(self, documents: List[Document]) -> List[str]:
        """
        Prepare the input by concatenating the document text with the metadata fields specified.
        :param documents: The list of Document objects.

        :return: A list of strings to be given as input to Bedrock model.
        """
        concatenated_input_list = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta.get(key)
            ]
            concatenated_input = self.meta_data_separator.join([*meta_values_to_embed, doc.content or ""])
            concatenated_input_list.append(concatenated_input)

        return concatenated_input_list

    @component.output_types(documents=List[Document])
    def run(self, query: str, documents: List[Document], top_k: Optional[int] = None) -> Dict[str, List[Document]]:
        """
        Use the Amazon Bedrock Reranker to re-rank the list of documents based on the query.

        :param query:
            Query string.
        :param documents:
            List of Documents.
        :param top_k:
            The maximum number of Documents you want the Ranker to return.
        :returns:
            A dictionary with the following keys:
            - `documents`: List of Documents most similar to the given query in descending order of similarity.

        :raises ValueError: If `top_k` is not > 0.
        """
        top_k = top_k or self.top_k
        if top_k <= 0:
            msg = f"top_k must be > 0, but got {top_k}"
            raise ValueError(msg)

        if not documents:
            return {"documents": []}

        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None

        region = resolve_secret(self.aws_region_name)

        bedrock_input_docs = self._prepare_bedrock_input_docs(documents)
        if len(bedrock_input_docs) > MAX_NUM_DOCS_FOR_BEDROCK_RANKER:
            logger.warning(
                f"The Amazon Bedrock reranking endpoint only supports {MAX_NUM_DOCS_FOR_BEDROCK_RANKER} documents.\
                The number of documents has been truncated to {MAX_NUM_DOCS_FOR_BEDROCK_RANKER} \
                from {len(bedrock_input_docs)}."
            )
            bedrock_input_docs = bedrock_input_docs[:MAX_NUM_DOCS_FOR_BEDROCK_RANKER]

        try:
            # Make the API call to Amazon Bedrock
            response = self._bedrock_client.rerank(
                queries=[
                    {"textQuery": {"text": query}, "type": "TEXT"},
                ],
                rerankingConfiguration={
                    "bedrockRerankingConfiguration": {
                        "modelConfiguration": {
                            "modelArn": f"arn:aws:bedrock:{region}::foundation-model/{self.model_name}"
                        },
                        "numberOfResults": top_k,
                    },
                    "type": "BEDROCK_RERANKING_MODEL",
                },
                sources=[
                    {
                        "inlineDocumentSource": {
                            "textDocument": {"text": doc},
                            "type": "TEXT",
                        },
                        "type": "INLINE",
                    }
                    for doc in bedrock_input_docs
                ],
            )

            # Parse the response
            results = response["results"]

            # Sort documents based on the reranking results
            sorted_docs = []
            for result in results:
                idx = result["index"]
                score = result["relevanceScore"]
                doc = documents[idx]
                doc.score = score
                sorted_docs.append(doc)

            return {"documents": sorted_docs}
        except ClientError as client_error:
            msg = f"Could not perform inference for Amazon Bedrock model {self.model_name} due to:\n{client_error}"
            raise AmazonBedrockInferenceError(msg) from client_error
        except KeyError as key_error:
            msg = f"Unexpected response format from Amazon Bedrock: {key_error}"
            raise AmazonBedrockInferenceError(msg) from key_error
        except Exception as exception:
            msg = f"Error during Amazon Bedrock API call: {exception}"
            raise AmazonBedrockInferenceError(msg) from exception
