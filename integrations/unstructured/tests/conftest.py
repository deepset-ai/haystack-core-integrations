from pathlib import Path

import pytest

from haystack_integrations.components.converters.unstructured import UnstructuredFileConverter

LOCAL_API_URL = "http://localhost:8000/general/v0/general"


@pytest.fixture
def set_env_variables(monkeypatch):
    monkeypatch.setenv("UNSTRUCTURED_API_KEY", "test-api-key")


@pytest.fixture
def samples_path():
    return Path(__file__).parent / "samples"


@pytest.fixture
def local_api_url() -> str:
    """URL of an Unstructured API running locally, which needs no API key."""
    return LOCAL_API_URL


@pytest.fixture
def local_converter(local_api_url) -> UnstructuredFileConverter:
    """A converter pointing at a local API, so no API key is required."""
    return UnstructuredFileConverter(api_url=local_api_url)
