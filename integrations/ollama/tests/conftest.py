from pathlib import Path

import pytest


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
