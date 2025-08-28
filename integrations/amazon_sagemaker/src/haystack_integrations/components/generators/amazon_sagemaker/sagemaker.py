import json
from typing import Any, ClassVar, Dict, List, Optional, Union

import boto3
import requests
from botocore.exceptions import BotoCoreError
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.generators.amazon_sagemaker.errors import (
    AWSConfigurationError,
    SagemakerInferenceError,
    SagemakerNotReadyError,
)


logger = logging.getLogger(__name__)


MODEL_NOT_READY_STATUS_CODE = 429


@component
class SagemakerGenerator:
    """
    Enables text generation using Amazon Sagemaker.

    SagemakerGenerator supports Large Language Models (LLMs) hosted and deployed on a SageMaker Inference Endpoint.
    For guidance on how to deploy a model to SageMaker, refer to the
    [SageMaker JumpStart foundation models documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-use.html).

    Usage example:
    ```python
    # Make sure your AWS credentials are set up correctly. You can use environment variables or a shared credentials
    # file. Then you can use the generator as follows:
    from haystack_integrations.components.generators.amazon_sagemaker import SagemakerGenerator

    generator = SagemakerGenerator(model="jumpstart-dft-hf-llm-falcon-7b-bf16")
    response = generator.run("What's Natural Language Processing? Be brief.")
    print(response)
    >>> {'replies': ['Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on
    >>> the interaction between computers and human language. It involves enabling computers to understand, interpret,
    >>> and respond to natural human language in a way that is both meaningful and useful.'], 'meta': [{}]}
    ```
    """

    model_generation_keys: ClassVar = ["generated_text", "generation"]

    def __init__(
        self,
        model: str,
        aws_access_key_id: Optional[Secret] = Secret.from_env_var(["AWS_ACCESS_KEY_ID"], strict=False),  # noqa: B008
        aws_secret_access_key: Optional[Secret] = Secret.from_env_var(  # noqa: B008
            ["AWS_SECRET_ACCESS_KEY"], strict=False
        ),
        aws_session_token: Optional[Secret] = Secret.from_env_var(["AWS_SESSION_TOKEN"], strict=False),  # noqa: B008
        aws_region_name: Optional[Secret] = Secret.from_env_var(["AWS_DEFAULT_REGION"], strict=False),  # noqa: B008
        aws_profile_name: Optional[Secret] = Secret.from_env_var(["AWS_PROFILE"], strict=False),  # noqa: B008
        aws_custom_attributes: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Instantiates the session with SageMaker.

        :param aws_access_key_id: The `Secret` for AWS access key ID.
        :param aws_secret_access_key: The `Secret` for AWS secret access key.
        :param aws_session_token: The `Secret` for AWS session token.
        :param aws_region_name: The `Secret` for AWS region name. If not provided, the default region will be used.
        :param aws_profile_name: The `Secret` for AWS profile name. If not provided, the default profile will be used.
        :param model: The name for SageMaker Model Endpoint.
        :param aws_custom_attributes: Custom attributes to be passed to SageMaker, for example `{"accept_eula": True}`
            in case of Llama-2 models.
        :param generation_kwargs: Additional keyword arguments for text generation. For a list of supported parameters
            see your model's documentation page, for example here for HuggingFace models:
            https://huggingface.co/blog/sagemaker-huggingface-llm#4-run-inference-and-chat-with-our-model

            Specifically, Llama-2 models support the following inference payload parameters:

            - `max_new_tokens`: Model generates text until the output length (excluding the input context length)
                reaches `max_new_tokens`. If specified, it must be a positive integer.
            - `temperature`: Controls the randomness in the output. Higher temperature results in output sequence with
                low-probability words and lower temperature results in output sequence with high-probability words.
                If `temperature=0`, it results in greedy decoding. If specified, it must be a positive float.
            - `top_p`: In each step of text generation, sample from the smallest possible set of words with cumulative
                probability `top_p`. If specified, it must be a float between 0 and 1.
            - `return_full_text`: If `True`, input text will be part of the output generated text. If specified, it must
                be boolean. The default value for it is `False`.
        """
        self.model = model
        self.aws_custom_attributes = aws_custom_attributes or {}
        self.generation_kwargs = generation_kwargs or {"max_new_tokens": 1024}
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token
        self.aws_region_name = aws_region_name
        self.aws_profile_name = aws_profile_name

        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None

        try:
            session = self._get_aws_session(
                aws_access_key_id=resolve_secret(aws_access_key_id),
                aws_secret_access_key=resolve_secret(aws_secret_access_key),
                aws_session_token=resolve_secret(aws_session_token),
                aws_region_name=resolve_secret(aws_region_name),
                aws_profile_name=resolve_secret(aws_profile_name),
            )
            self.client = session.client("runtime.sagemaker")
        except Exception as e:
            msg = (
                f"Could not connect to SageMaker Inference Endpoint '{self.model}'."
                f"Make sure the Endpoint exists and AWS environment is configured."
            )
            raise AWSConfigurationError(msg) from e

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Returns data that is sent to Posthog for usage analytics.
        :returns: A dictionary with the following keys:
            - `model`: The name of the model.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model,
            aws_access_key_id=self.aws_access_key_id.to_dict() if self.aws_access_key_id else None,
            aws_secret_access_key=self.aws_secret_access_key.to_dict() if self.aws_secret_access_key else None,
            aws_session_token=self.aws_session_token.to_dict() if self.aws_session_token else None,
            aws_region_name=self.aws_region_name.to_dict() if self.aws_region_name else None,
            aws_profile_name=self.aws_profile_name.to_dict() if self.aws_profile_name else None,
            aws_custom_attributes=self.aws_custom_attributes,
            generation_kwargs=self.generation_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SagemakerGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(
            data["init_parameters"],
            ["aws_access_key_id", "aws_secret_access_key", "aws_session_token", "aws_region_name", "aws_profile_name"],
        )
        return default_from_dict(cls, data)

    @staticmethod
    def _get_aws_session(
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
    ) -> boto3.Session:
        """
        Creates an AWS Session with the given parameters.

        Checks if the provided AWS credentials are valid and can be used to connect to AWS.

        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.

        :raises AWSConfigurationError: If the provided AWS credentials are invalid.
        :returns: The created AWS session.
        """
        try:
            return boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=aws_region_name,
                profile_name=aws_profile_name,
            )
        except BotoCoreError as e:
            msg = f"Failed to initialize the session with provided AWS credentials: {e}."
            raise AWSConfigurationError(msg) from e

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(
        self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Union[List[str], List[Dict[str, Any]]]]:
        """
        Invoke the text generation inference based on the provided prompt and generation parameters.

        :param prompt: The string prompt to use for text generation.
        :param generation_kwargs: Additional keyword arguments for text generation. These parameters will
            potentially override the parameters passed in the `__init__` method.
        :raises ValueError: If the model response type is not a list of dictionaries or a single dictionary.
        :raises SagemakerNotReadyError: If the SageMaker model is not ready to accept requests.
        :raises SagemakerInferenceError: If the SageMaker Inference returns an error.
        :returns: A dictionary with the following keys:
            - `replies`: A list of strings containing the generated responses
            - `meta`: A list of dictionaries containing the metadata for each response.
        """
        generation_kwargs = generation_kwargs or self.generation_kwargs
        custom_attributes = ";".join(
            f"{k}={str(v).lower() if isinstance(v, bool) else str(v)}" for k, v in self.aws_custom_attributes.items()
        )
        try:
            body = json.dumps({"inputs": prompt, "parameters": generation_kwargs})
            response = self.client.invoke_endpoint(
                EndpointName=self.model,
                Body=body,
                ContentType="application/json",
                Accept="application/json",
                CustomAttributes=custom_attributes,
            )
            response_json = response.get("Body").read().decode("utf-8")
            output: Dict[str, Dict[str, Any]] = json.loads(response_json)

            # The output might be either a list of dictionaries or a single dictionary
            list_output: List[Dict[str, Any]]
            if output and isinstance(output, dict):
                list_output = [output]
            elif isinstance(output, list) and all(isinstance(o, dict) for o in output):
                list_output = output
            else:
                msg = f"Unexpected model response type: {type(output)}"
                raise ValueError(msg)

            # The key where the replies are stored changes from model to model, so we need to look for it.
            # All other keys in the response are added to the metadata.
            # Unfortunately every model returns different metadata, most of them return none at all,
            # so we can't replicate the metadata structure of other generators.
            for key in self.model_generation_keys:
                if key in list_output[0]:
                    break
            replies = [o.pop(key, None) for o in list_output]

            return {"replies": replies, "meta": list_output * len(replies)}

        except requests.HTTPError as err:
            res = err.response
            if res.status_code == MODEL_NOT_READY_STATUS_CODE:
                msg = f"Sagemaker model not ready: {res.text}"
                raise SagemakerNotReadyError(msg) from err

            msg = f"SageMaker Inference returned an error. Status code: {res.status_code}. Response body: {res.text}"
            raise SagemakerInferenceError(msg) from err
