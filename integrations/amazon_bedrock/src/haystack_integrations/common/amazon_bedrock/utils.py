# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

import aioboto3
import boto3
from botocore.exceptions import BotoCoreError

from haystack_integrations.common.amazon_bedrock.errors import AWSConfigurationError

AWS_CONFIGURATION_KEYS = [
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_profile_name",
]


def get_aws_session(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    aws_region_name: Optional[str] = None,
    aws_profile_name: Optional[str] = None,
    async_mode: bool = False,
    **kwargs: Any,
) -> Union[boto3.Session, aioboto3.Session]:
    """
    Creates an AWS Session with the given parameters.
    Checks if the provided AWS credentials are valid and can be used to connect to AWS.

    :param aws_access_key_id: AWS access key ID.
    :param aws_secret_access_key: AWS secret access key.
    :param aws_session_token: AWS session token.
    :param aws_region_name: AWS region name.
    :param aws_profile_name: AWS profile name.
    :param async_mode: If True, returns an async AWS session.
    :param kwargs: The kwargs passed down to the service client. Supported kwargs depend on the model chosen.
        See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html.
    :raises AWSConfigurationError: If the provided AWS credentials are invalid.
    :returns: The created AWS session.
    """
    try:
        if async_mode:
            return aioboto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                region_name=aws_region_name,
                profile_name=aws_profile_name,
            )

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


def aws_configured(**kwargs: Any) -> bool:
    """
    Checks whether AWS configuration is provided.
    :param kwargs: The kwargs passed down to the generator.
    :returns: True if AWS configuration is provided, False otherwise.
    """
    aws_config_provided = any(key in kwargs for key in AWS_CONFIGURATION_KEYS)
    return aws_config_provided
