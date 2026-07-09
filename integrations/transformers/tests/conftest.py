# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from haystack.document_stores.in_memory import InMemoryDocumentStore


@pytest.fixture()
def del_hf_env_vars_if_empty(monkeypatch):
    """
    Delete Hugging Face environment variables for tests if empty.

    Prevents passing empty tokens to Hugging Face, which would cause API calls to fail.
    This is particularly relevant for PRs opened from forks, where secrets are not available
    and empty environment variables might be set instead of being removed.

    See https://github.com/deepset-ai/haystack/issues/8811 for more details.
    """
    for var in ("HF_API_TOKEN", "HF_TOKEN"):
        if not os.environ.get(var, "").strip():
            monkeypatch.delenv(var, raising=False)


@pytest.fixture()
def in_memory_doc_store():
    return InMemoryDocumentStore()


@pytest.fixture(autouse=True)
def allow_deserialization_of_test_modules(monkeypatch):
    """
    haystack-ai >= 3.0 refuses to deserialize classes and callables from modules outside its
    trusted-module allowlist. Tools and callbacks defined in the test modules live outside that
    allowlist, so trust them explicitly; haystack-ai 2.x ignores this environment variable.
    """
    monkeypatch.setenv("HAYSTACK_DESERIALIZATION_ALLOWLIST", "tests,test_*")
