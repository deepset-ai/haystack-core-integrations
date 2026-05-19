import os
import re
from unittest.mock import Mock

import pytest
import requests
from botocore.exceptions import BotoCoreError
from haystack.utils.auth import EnvVarSecret

from haystack_integrations.components.generators.amazon_sagemaker import SagemakerGenerator
from haystack_integrations.components.generators.amazon_sagemaker.errors import (
    AWSConfigurationError,
    SagemakerInferenceError,
    SagemakerNotReadyError,
)


def test_to_dict(set_env_variables, mock_boto3_session):  # noqa: ARG001
    """
    Test that the to_dict method returns the correct dictionary without aws credentials
    """

    mocked_dict = {
        "type": "haystack_integrations.components.generators.amazon_sagemaker.sagemaker.SagemakerGenerator",
        "init_parameters": {
            "model": "model",
            "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
            "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
            "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
            "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
            "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
            "aws_custom_attributes": {"accept_eula": True},
            "generation_kwargs": {"max_new_tokens": 10},
        },
    }

    generator = SagemakerGenerator(
        model="model",
        generation_kwargs={"max_new_tokens": 10},
        aws_custom_attributes={"accept_eula": True},
    )
    assert generator.to_dict() == mocked_dict


def test_from_dict(set_env_variables, mock_boto3_session):  # noqa: ARG001
    """
    Test that the from_dict method returns the correct object
    """

    mocked_dict = {
        "type": "haystack_integrations.components.generators.amazon_sagemaker.sagemaker.SagemakerGenerator",
        "init_parameters": {
            "model": "model",
            "aws_access_key_id": {"type": "env_var", "env_vars": ["AWS_ACCESS_KEY_ID"], "strict": False},
            "aws_secret_access_key": {"type": "env_var", "env_vars": ["AWS_SECRET_ACCESS_KEY"], "strict": False},
            "aws_session_token": {"type": "env_var", "env_vars": ["AWS_SESSION_TOKEN"], "strict": False},
            "aws_region_name": {"type": "env_var", "env_vars": ["AWS_DEFAULT_REGION"], "strict": False},
            "aws_profile_name": {"type": "env_var", "env_vars": ["AWS_PROFILE"], "strict": False},
            "aws_custom_attributes": {"accept_eula": True},
            "generation_kwargs": {"max_new_tokens": 10},
        },
    }

    generator = SagemakerGenerator.from_dict(mocked_dict)
    assert generator.model == "model"
    assert isinstance(generator.aws_access_key_id, EnvVarSecret)


def test_default_constructor(set_env_variables, mock_boto3_session):  # noqa: ARG001
    """
    Test that the default constructor sets the correct values
    """

    generator = SagemakerGenerator(model="test-model")
    assert generator.generation_kwargs == {"max_new_tokens": 1024}
    assert generator.model == "test-model"

    # assert mocked boto3 client called exactly once
    mock_boto3_session.assert_called_once()
    assert generator.client is not None

    # assert mocked boto3 client was called with the correct parameters
    mock_boto3_session.assert_called_with(
        aws_access_key_id="some_fake_id",
        aws_secret_access_key="some_fake_key",
        aws_session_token="some_fake_token",
        profile_name="some_fake_profile",
        region_name="fake_region",
    )


def test_init_raises_boto_error(set_env_variables, mock_boto3_session):  # noqa: ARG001
    mock_boto3_session.side_effect = BotoCoreError()
    with pytest.raises(
        AWSConfigurationError,
        match=re.escape(
            "Could not connect to SageMaker Inference Endpoint 'test-model'."
            "Make sure the Endpoint exists and AWS environment is configured."
        ),
    ):
        SagemakerGenerator(model="test-model")


def test_run_with_list_of_dictionaries(set_env_variables, mock_boto3_session):  # noqa: ARG001
    client_mock = Mock()
    client_mock.invoke_endpoint.return_value = {
        "Body": Mock(read=lambda: b'[{"generated_text": "test-reply", "other": "metadata"}]')
    }
    component = SagemakerGenerator(model="test-model")
    component.client = client_mock
    response = component.run("What's Natural Language Processing?")

    # check that the component returns the correct ChatMessage response
    assert isinstance(response, dict)
    assert "replies" in response
    assert isinstance(response["replies"], list)
    assert len(response["replies"]) == 1
    assert [isinstance(reply, str) for reply in response["replies"]]
    assert "test-reply" in response["replies"][0]

    assert "meta" in response
    assert isinstance(response["meta"], list)
    assert len(response["meta"]) == 1
    assert [isinstance(reply, dict) for reply in response["meta"]]
    assert response["meta"][0]["other"] == "metadata"


def test_run_with_single_dictionary(set_env_variables, mock_boto3_session):  # noqa: ARG001
    client_mock = Mock()
    client_mock.invoke_endpoint.return_value = {
        "Body": Mock(read=lambda: b'{"generation": "test-reply", "other": "metadata"}')
    }

    component = SagemakerGenerator(model="test-model")
    component.client = client_mock
    response = component.run("What's Natural Language Processing?")

    # check that the component returns the correct ChatMessage response
    assert isinstance(response, dict)
    assert "replies" in response
    assert isinstance(response["replies"], list)
    assert len(response["replies"]) == 1
    assert [isinstance(reply, str) for reply in response["replies"]]
    assert "test-reply" in response["replies"][0]

    assert "meta" in response
    assert isinstance(response["meta"], list)
    assert len(response["meta"]) == 1
    assert [isinstance(reply, dict) for reply in response["meta"]]
    assert response["meta"][0]["other"] == "metadata"


def test_run_raises_on_unexpected_response_type(set_env_variables, mock_boto3_session):  # noqa: ARG001
    client_mock = Mock()
    client_mock.invoke_endpoint.return_value = {"Body": Mock(read=lambda: b'"just-a-string"')}
    component = SagemakerGenerator(model="test-model")
    component.client = client_mock
    with pytest.raises(ValueError, match="Unexpected model response type"):
        component.run("What's Natural Language Processing?")


def test_run_with_unknown_generation_key_returns_none_reply(set_env_variables, mock_boto3_session):  # noqa: ARG001
    client_mock = Mock()
    client_mock.invoke_endpoint.return_value = {"Body": Mock(read=lambda: b'{"unknown_key": "value"}')}
    component = SagemakerGenerator(model="test-model")
    component.client = client_mock
    response = component.run("What's Natural Language Processing?")
    assert response["replies"] == [None]
    assert response["meta"] == [{"unknown_key": "value"}]


def test_run_raises_not_ready_error_on_429(set_env_variables, mock_boto3_session):  # noqa: ARG001
    client_mock = Mock()
    http_response = Mock(status_code=429, text="model is loading")
    client_mock.invoke_endpoint.side_effect = requests.HTTPError(response=http_response)
    component = SagemakerGenerator(model="test-model")
    component.client = client_mock
    with pytest.raises(SagemakerNotReadyError, match="Sagemaker model not ready: model is loading"):
        component.run("What's Natural Language Processing?")


def test_run_raises_inference_error_on_other_http_error(set_env_variables, mock_boto3_session):  # noqa: ARG001
    client_mock = Mock()
    http_response = Mock(status_code=500, text="internal server error")
    client_mock.invoke_endpoint.side_effect = requests.HTTPError(response=http_response)
    component = SagemakerGenerator(model="test-model")
    component.client = client_mock
    with pytest.raises(
        SagemakerInferenceError,
        match=re.escape(
            "SageMaker Inference returned an error. Status code: 500. Response body: internal server error"
        ),
    ):
        component.run("What's Natural Language Processing?")


def test_run_passes_custom_attributes_and_generation_kwargs(set_env_variables, mock_boto3_session):  # noqa: ARG001
    client_mock = Mock()
    client_mock.invoke_endpoint.return_value = {"Body": Mock(read=lambda: b'{"generated_text": "ok"}')}
    component = SagemakerGenerator(
        model="test-model",
        aws_custom_attributes={"accept_eula": True, "max_retries": 3},
    )
    component.client = client_mock
    component.run("prompt", generation_kwargs={"max_new_tokens": 5})

    call_kwargs = client_mock.invoke_endpoint.call_args.kwargs
    assert call_kwargs["EndpointName"] == "test-model"
    assert call_kwargs["CustomAttributes"] == "accept_eula=true;max_retries=3"
    assert '"max_new_tokens": 5' in call_kwargs["Body"]


@pytest.mark.skipif(
    (not os.environ.get("AWS_ACCESS_KEY_ID", None) or not os.environ.get("AWS_SECRET_ACCESS_KEY", None)),
    reason="Export two env vars called AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to run this test.",
)
@pytest.mark.integration
@pytest.mark.parametrize(
    "model",
    [
        "jumpstart-dft-hf-llm-falcon-7b-instruct-bf16",
        "jumpstart-dft-meta-textgenerationneuron-llama-2-7b",
        "jumpstart-dft-hf-textgeneration-bloomz-1b1",
    ],
)
def test_run(model: str):
    component = SagemakerGenerator(model=model, generation_kwargs={"max_new_tokens": 10})
    response = component.run("What's Natural Language Processing?")

    # check that the component returns the correct ChatMessage response
    assert isinstance(response, dict)
    assert "replies" in response
    assert isinstance(response["replies"], list)
    assert len(response["replies"]) == 1
    assert [isinstance(reply, str) for reply in response["replies"]]

    # Coarse check: assuming no more than 4 chars per token. In any case it
    # will fail if the `max_new_tokens` parameter is not respected, as the
    # default is either 256 or 1024
    assert all(len(reply) <= 40 for reply in response["replies"])

    assert "meta" in response
    assert isinstance(response["meta"], list)
    assert len(response["meta"]) == 1
    assert [isinstance(reply, dict) for reply in response["meta"]]
