# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack import tracing
from haystack.tracing import disable_tracing


@pytest.fixture(autouse=True)
def allow_deserialization_of_test_modules(monkeypatch):
    """
    haystack-ai >= 3.0 refuses to deserialize classes and callables from modules outside its
    trusted-module allowlist. Tools and callbacks defined in the test modules live outside that
    allowlist, so trust them explicitly; haystack-ai 2.x ignores this environment variable.
    """
    monkeypatch.setenv("HAYSTACK_DESERIALIZATION_ALLOWLIST", "tests,test_*")


@pytest.fixture(autouse=True)
def content_tracing_enabled(monkeypatch):
    """
    Turn Haystack content tracing on for every test, by patching the flag rather than the
    environment variable.

    ``HAYSTACK_CONTENT_TRACING_ENABLED`` is read exactly once, when ``haystack.tracing.tracer``
    is first imported, so setting it at test-module scope only works when that module happens to
    be collected before anything else imports Haystack. That makes the whole suite depend on
    collection order. Patching the resolved flag is order-independent, and tests that need it off
    patch it back themselves.
    """
    monkeypatch.setattr(tracing.tracer, "is_content_tracing_enabled", True)


@pytest.fixture(autouse=True)
def reset_global_tracer():
    """
    Uninstall any tracer a test enabled globally.

    ``RhesisConnector.__init__`` calls ``tracing.enable_tracing`` as its last statement, so a test
    that constructs one leaves that tracer installed for the rest of the session.
    """
    yield
    disable_tracing()
