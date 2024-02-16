import pytest

from haystack_integrations.components.generators.amazon_sagemaker import SagemakerGenerator
from haystack.utils.auth import EnvVarSecret

from botocore.exceptions import BotoCoreError

from haystack_integrations.components.generators.amazon_sagemaker.errors import AWSConfigurationError

from unittest.mock import Mock, patch

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


@pytest.mark.unit
@pytest.mark.usefixtures("set_env_variables", "mock_boto3_session")
def test_to_dict():
    """
    Test that the to_dict method returns the correct dictionary without aws credentials
    """

    generator = SagemakerGenerator(
        model="model",
        generation_kwargs={"max_new_tokens": 10},
        aws_custom_attributes={"accept_eula": True},
    )
    assert generator.to_dict() == mocked_dict


@pytest.mark.unit
@pytest.mark.usefixtures("set_env_variables", "mock_boto3_session")
def test_from_dict():
    """
    Test that the from_dict method returns the correct object
    """

    generator = SagemakerGenerator.from_dict(mocked_dict)
    assert generator.model == "model"
    assert isinstance(generator.aws_access_key_id, EnvVarSecret)


@pytest.mark.unit
@pytest.mark.usefixtures("set_env_variables", "mock_boto3_session")
def test_default_constructor(mock_boto3_session):
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


@pytest.mark.unit
@pytest.mark.usefixtures("set_env_variables", "mock_boto3_session")
def test_init_raises_boto_error():
    with patch("boto3.Session") as mock_boto3_session:
        mock_boto3_session.side_effect = BotoCoreError()
        with pytest.raises(
            AWSConfigurationError,
            match="Could not connect to SageMaker Inference Endpoint 'test-model'."
            "Make sure the Endpoint exists and AWS environment is configured.",
        ):
            SagemakerGenerator(model="test-model")


@pytest.mark.unit
@pytest.mark.usefixtures("set_env_variables", "mock_boto3_session")
def test_run_with_list_of_dictionaries():
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


@pytest.mark.unit
@pytest.mark.usefixtures("set_env_variables", "mock_boto3_session")
def test_run_with_single_dictionary():
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


# @pytest.mark.skipif(
#     (not os.environ.get("AWS_ACCESS_KEY_ID", None) or not os.environ.get("AWS_SECRET_ACCESS_KEY", None)),
#     reason="Export two env vars called AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to run this test.",
# )
# @pytest.mark.integration
# def test_run_falcon(self):
#     component = SagemakerGenerator(
#         model="jumpstart-dft-hf-llm-falcon-7b-instruct-bf16", generation_kwargs={"max_new_tokens": 10}
#     )
#     # component.warm_up()
#     response = component.run("What's Natural Language Processing?")
#
#     # check that the component returns the correct ChatMessage response
#     assert isinstance(response, dict)
#     assert "replies" in response
#     assert isinstance(response["replies"], list)
#     assert len(response["replies"]) == 1
#     assert [isinstance(reply, str) for reply in response["replies"]]
#
#     # Coarse check: assuming no more than 4 chars per token. In any case it
#     # will fail if the `max_new_tokens` parameter is not respected, as the
#     # default is either 256 or 1024
#     assert all(len(reply) <= 40 for reply in response["replies"])
#
#     assert "meta" in response
#     assert isinstance(response["meta"], list)
#     assert len(response["meta"]) == 1
#     assert [isinstance(reply, dict) for reply in response["meta"]]
#
# @pytest.mark.skipif(
#     (not os.environ.get("AWS_ACCESS_KEY_ID", None) or not os.environ.get("AWS_SECRET_ACCESS_KEY", None)),
#     reason="Export two env vars called AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to run this test.",
# )
# @pytest.mark.integration
# def test_run_llama2(self):
#     component = SagemakerGenerator(
#         model="jumpstart-dft-meta-textgenerationneuron-llama-2-7b",
#         generation_kwargs={"max_new_tokens": 10},
#         aws_custom_attributes={"accept_eula": True},
#     )
#     component.warm_up()
#     response = component.run("What's Natural Language Processing?")
#
#     # check that the component returns the correct ChatMessage response
#     assert isinstance(response, dict)
#     assert "replies" in response
#     assert isinstance(response["replies"], list)
#     assert len(response["replies"]) == 1
#     assert [isinstance(reply, str) for reply in response["replies"]]
#
#     # Coarse check: assuming no more than 4 chars per token. In any case it
#     # will fail if the `max_new_tokens` parameter is not respected, as the
#     # default is either 256 or 1024
#     assert all(len(reply) <= 40 for reply in response["replies"])
#
#     assert "meta" in response
#     assert isinstance(response["meta"], list)
#     assert len(response["meta"]) == 1
#     assert [isinstance(reply, dict) for reply in response["meta"]]
#
# @pytest.mark.skipif(
#     (not os.environ.get("AWS_ACCESS_KEY_ID", None) or not os.environ.get("AWS_SECRET_ACCESS_KEY", None)),
#     reason="Export two env vars called AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to run this test.",
# )
# @pytest.mark.integration
# def test_run_bloomz(self):
#     component = SagemakerGenerator(
#         model="jumpstart-dft-hf-textgeneration-bloomz-1b1", generation_kwargs={"max_new_tokens": 10}
#     )
#     component.warm_up()
#     response = component.run("What's Natural Language Processing?")
#
#     # check that the component returns the correct ChatMessage response
#     assert isinstance(response, dict)
#     assert "replies" in response
#     assert isinstance(response["replies"], list)
#     assert len(response["replies"]) == 1
#     assert [isinstance(reply, str) for reply in response["replies"]]
#
#     # Coarse check: assuming no more than 4 chars per token. In any case it
#     # will fail if the `max_new_tokens` parameter is not respected, as the
#     # default is either 256 or 1024
#     assert all(len(reply) <= 40 for reply in response["replies"])
#
#     assert "meta" in response
#     assert isinstance(response["meta"], list)
#     assert len(response["meta"]) == 1
#     assert [isinstance(reply, dict) for reply in response["meta"]]
