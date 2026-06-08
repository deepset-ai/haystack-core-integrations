from unittest.mock import patch

import aiobotocore.session
import pytest
from botocore.exceptions import BotoCoreError

from haystack_integrations.common.amazon_bedrock.errors import AWSConfigurationError
from haystack_integrations.common.amazon_bedrock.utils import aws_configured, get_aws_session


class TestGetAwsSession:
    def test_async_session_no_profile(self):
        session = get_aws_session(async_mode=True)
        assert isinstance(session, aiobotocore.session.AioSession)

    def test_async_session_with_profile(self):
        session = get_aws_session(async_mode=True, aws_profile_name="my-profile")

        assert isinstance(session, aiobotocore.session.AioSession)
        assert session.get_config_variable("profile") == "my-profile"

    def test_async_session_without_profile_does_not_set_config(self):
        # When no profile is given, profile config variable should be unset (None)
        session = get_aws_session(async_mode=True)
        assert session.get_config_variable("profile") is None

    def test_sync_session_raises_on_botocoreerror(self):
        with patch("boto3.Session", side_effect=BotoCoreError()):
            with pytest.raises(AWSConfigurationError):
                get_aws_session(aws_access_key_id="bad")


class TestAwsConfigured:
    def test_returns_true_when_key_present(self):
        assert aws_configured(aws_access_key_id="key") is True

    def test_returns_true_when_secret_present(self):
        assert aws_configured(aws_secret_access_key="secret") is True

    def test_returns_true_when_region_present(self):
        assert aws_configured(aws_region_name="us-east-1") is True

    def test_returns_false_when_no_aws_keys(self):
        assert aws_configured(some_other_param="value") is False

    def test_returns_false_when_empty(self):
        assert aws_configured() is False
