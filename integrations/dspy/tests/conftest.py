import pytest


@pytest.fixture(autouse=True)
def allow_deserialization_of_test_modules(monkeypatch):
    """
    Trust the modules that hold the signature classes used in these tests.

    haystack-ai >= 3.0 refuses to deserialize classes from modules outside its trusted-module
    allowlist. Signature classes live in the test modules or in `dspy` itself, both outside that
    allowlist, so trust them explicitly; haystack-ai 2.x ignores this environment variable.
    """
    monkeypatch.setenv("HAYSTACK_DESERIALIZATION_ALLOWLIST", "tests,test_*,dspy")
