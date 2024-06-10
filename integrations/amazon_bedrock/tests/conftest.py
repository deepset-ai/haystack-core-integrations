from unittest.mock import patch

import os
import pytest

def pytest_runtest_setup(item):
    if os.getenv("IS_FORK") == "true" and "integration" in item.keywords:
        pytest.skip("Skipping integration test since it's running in a forked repository")


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


@pytest.fixture
def mock_prompt_handler():
    with patch(
        "haystack_integrations.components.generators.amazon_bedrock.handlers.DefaultPromptHandler"
    ) as mock_prompt_handler:
        yield mock_prompt_handler
