from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def set_env_variables(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "some_fake_id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "some_fake_key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "some_fake_token")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "fake_region")
    monkeypatch.setenv("AWS_PROFILE", "some_fake_profile")


# create a fixture with mocked boto3 client and session
@pytest.fixture
def mock_boto3_session():
    with patch("boto3.Session") as mock_client:
        yield mock_client


# create a fixture with mocked aiobotocore session
@pytest.fixture
def mock_aioboto3_session():
    with patch("aiobotocore.session.AioSession") as mock_client:
        yield mock_client


@pytest.fixture()
def test_files_path():
    return Path(__file__).parent / "test_files"


@pytest.fixture(autouse=True)
def allow_deserialization_of_test_modules(monkeypatch):
    """
    haystack-ai >= 3.0 refuses to deserialize classes and callables from modules outside its
    trusted-module allowlist. Tools and callbacks defined in the test modules live outside that
    allowlist, so trust them explicitly; haystack-ai 2.x ignores this environment variable.
    """
    monkeypatch.setenv("HAYSTACK_DESERIALIZATION_ALLOWLIST", "tests,test_*")
