import json
import logging
import os
from typing import Any, ClassVar, Dict, List, Optional

import requests
from haystack import component, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport
from haystack_integrations.components.generators.amazon_sagemaker.errors import (
    AWSConfigurationError,
    SagemakerInferenceError,
    SagemakerNotReadyError,
)

with LazyImport(message="Run 'pip install boto3'") as boto3_import:
    import boto3  # type: ignore
    from botocore.client import BaseClient  # type: ignore


logger = logging.getLogger(__name__)


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
    from haystack.components.generators.sagemaker import SagemakerGenerator
    generator = SagemakerGenerator(model="jumpstart-dft-hf-llm-falcon-7b-instruct-bf16")
    generator.warm_up()
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
        aws_access_key_id_var: str = "AWS_ACCESS_KEY_ID",
        aws_secret_access_key_var: str = "AWS_SECRET_ACCESS_KEY",
        aws_session_token_var: str = "AWS_SESSION_TOKEN",
        aws_region_name_var: str = "AWS_REGION",
        aws_profile_name_var: str = "AWS_PROFILE",
        aws_custom_attributes: Optional[Dict[str, Any]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Instantiates the session with SageMaker.

        :param model: The name for SageMaker Model Endpoint.
        :param aws_access_key_id_var: The name of the env var where the AWS access key ID is stored.
        :param aws_secret_access_key_var: The name of the env var where the AWS secret access key is stored.
        :param aws_session_token_var: The name of the env var where the AWS session token is stored.
        :param aws_region_name_var: The name of the env var where the AWS region name is stored.
        :param aws_profile_name_var: The name of the env var where the AWS profile name is stored.
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
        self.aws_access_key_id_var = aws_access_key_id_var
        self.aws_secret_access_key_var = aws_secret_access_key_var
        self.aws_session_token_var = aws_session_token_var
        self.aws_region_name_var = aws_region_name_var
        self.aws_profile_name_var = aws_profile_name_var
        self.aws_custom_attributes = aws_custom_attributes or {}
        self.generation_kwargs = generation_kwargs or {"max_new_tokens": 1024}
        self.client: Optional[BaseClient] = None

        if not os.getenv(self.aws_access_key_id_var) or not os.getenv(self.aws_secret_access_key_var):
            msg = (
                f"Please provide AWS credentials via environment variables '{self.aws_access_key_id_var}' and "
                f"'{self.aws_secret_access_key_var}'."
            )
            raise AWSConfigurationError(msg)

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
            aws_access_key_id_var=self.aws_access_key_id_var,
            aws_secret_access_key_var=self.aws_secret_access_key_var,
            aws_session_token_var=self.aws_session_token_var,
            aws_region_name_var=self.aws_region_name_var,
            aws_profile_name_var=self.aws_profile_name_var,
            aws_custom_attributes=self.aws_custom_attributes,
            generation_kwargs=self.generation_kwargs,
        )

    @classmethod
    def from_dict(cls, data) -> "SagemakerGenerator":
        """
        Deserialize the dictionary into an instance of SagemakerGenerator.
        """
        return default_from_dict(cls, data)

    def warm_up(self):
        """
        Initializes the SageMaker Inference client.
        """
        boto3_import.check()
        try:
            session = boto3.Session(
                aws_access_key_id=os.getenv(self.aws_access_key_id_var),
                aws_secret_access_key=os.getenv(self.aws_secret_access_key_var),
                aws_session_token=os.getenv(self.aws_session_token_var),
                region_name=os.getenv(self.aws_region_name_var),
                profile_name=os.getenv(self.aws_profile_name_var),
            )
            self.client = session.client("sagemaker-runtime")
        except Exception as e:
            msg = (
                f"Could not connect to SageMaker Inference Endpoint '{self.model}'."
                f"Make sure the Endpoint exists and AWS environment is configured."
            )
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
