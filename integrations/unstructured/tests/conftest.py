from pathlib import Path

import pytest


@pytest.fixture
def set_env_variables(monkeypatch):
    monkeypatch.setenv("UNSTRUCTURED_API_KEY", "test-api-key")


@pytest.fixture
def samples_path():
    return Path(__file__).parent / "samples"
