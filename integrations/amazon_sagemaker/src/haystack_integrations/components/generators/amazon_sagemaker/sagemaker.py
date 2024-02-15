import json
import logging
from typing import Any, ClassVar, Dict, List, Optional

import boto3
import requests
from botocore.exceptions import BotoCoreError
from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.generators.amazon_sagemaker.errors import (
    AWSConfigurationError,
    SagemakerInferenceError,
    SagemakerNotReadyError,
)


logger = logging.getLogger(__name__)

AWS_CONFIGURATION_KEYS = [
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_profile_name",
]


MODEL_NOT_READY_STATUS_CODE = 429


@component
class SagemakerGenerator:
    """
    Enables text generation using Sagemaker. It supports Large Language Models (LLMs) hosted and deployed on a SageMaker
    Inference Endpoint. For guidance on how to deploy a model to SageMaker, refer to the
    [SageMaker JumpStart foundation models documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-use.html).

    **Example:**

    First export your AWS credentials as environment variables:
    ```bash
    export AWS_ACCESS_KEY_ID=<your_access_key_id>
    export AWS_SECRET_ACCESS_KEY=<your_secret_access_key>
    ```
    (Note: you may also need to set the session token and region name, depending on your AWS configuration)

    Then you can use the generator as follows:
    ```python
    from haystack_integrations.components.generators.amazon_sagemaker import SagemakerGenerator
    generator = SagemakerGenerator(model="jumpstart-dft-hf-llm-falcon-7b-instruct-bf16")
    response = generator.run("What's Natural Language Processing? Be brief.")
    print(response)
    ```
    ```
    >> {'replies': ['Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on
    >> the interaction between computers and human language. It involves enabling computers to understand, interpret,
    >> and respond to natural human language in a way that is both meaningful and useful.'], 'meta': [{}]}
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

        :param model: The name for SageMaker Model Endpoint.

        :param aws_access_key_id: The name of the env var where the AWS access key ID is stored.
        :param aws_secret_access_key: The name of the env var where the AWS secret access key is stored.
        :param aws_session_token: The name of the env var where the AWS session token is stored.
        :param aws_region_name: The name of the env var where the AWS region name is stored.
        :param aws_profile_name: The name of the env var where the AWS profile name is stored.

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
            session = self.get_aws_session(
                aws_access_key_id=resolve_secret(aws_access_key_id),
                aws_secret_access_key=resolve_secret(aws_secret_access_key),
                aws_session_token=resolve_secret(aws_session_token),
                aws_region_name=resolve_secret(aws_region_name),
                aws_profile_name=resolve_secret(aws_profile_name),
            )
            self.client = session.client("sagemaker-runtime")
        except Exception as e:
            msg = (
                f"Could not connect to SageMaker Inference Endpoint '{self.model}'."
                f"Make sure the Endpoint exists and AWS environment is configured."
            )
            raise AWSConfigurationError(msg) from e

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the object to a dictionary.
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
    def from_dict(cls, data) -> "SagemakerGenerator":
        """
        Deserialize the dictionary into an instance of SagemakerGenerator.
        """
        deserialize_secrets_inplace(
            data["init_parameters"],
            ["aws_access_key_id", "aws_secret_access_key", "aws_session_token", "aws_region_name", "aws_profile_name"],
        )
        return default_from_dict(cls, data)

    @classmethod
    def get_aws_session(
        cls,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Creates an AWS Session with the given parameters.
        Checks if the provided AWS credentials are valid and can be used to connect to AWS.

        :param aws_access_key_id: AWS access key ID.
        :param aws_secret_access_key: AWS secret access key.
        :param aws_session_token: AWS session token.
        :param aws_region_name: AWS region name.
        :param aws_profile_name: AWS profile name.
        :param kwargs: The kwargs passed down to the service client. Supported kwargs depend on the model chosen.
            See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html.
        :raises AWSConfigurationError: If the provided AWS credentials are invalid.
        :return: The created AWS session.
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
            provided_aws_config = {k: v for k, v in kwargs.items() if k in AWS_CONFIGURATION_KEYS}
            msg = f"Failed to initialize the session with provided AWS credentials {provided_aws_config}"
            raise AWSConfigurationError(msg) from e

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param prompt: The string prompt to use for text generation.
        :param generation_kwargs: Additional keyword arguments for text generation. These parameters will
        potentially override the parameters passed in the `__init__` method.

        :return: A list of strings containing the generated responses and a list of dictionaries containing the metadata
        for each response.
        """
        if self.client is None:
            msg = "SageMaker Inference client is not initialized. Please call warm_up() first."
            raise ValueError(msg)

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

            msg = f"SageMaker Inference returned an error. Status code: {res.status_code} Response body: {res.text}"
            raise SagemakerInferenceError(msg, status_code=res.status_code) from err
