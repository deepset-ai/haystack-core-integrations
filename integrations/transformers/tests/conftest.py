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
