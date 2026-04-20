# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from haystack_integrations.components.retrievers.opensearch.utils import _resolve_document_store
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


def test_resolve_document_store_returns_default_when_runtime_is_none():
    default = Mock(spec=OpenSearchDocumentStore)
    assert _resolve_document_store(None, default) is default


def test_resolve_document_store_returns_runtime_when_valid():
    default = Mock(spec=OpenSearchDocumentStore)
    runtime = Mock(spec=OpenSearchDocumentStore)
    assert _resolve_document_store(runtime, default) is runtime


def test_resolve_document_store_raises_on_invalid_runtime():
    default = Mock(spec=OpenSearchDocumentStore)
    with pytest.raises(ValueError, match="document_store must be an instance of OpenSearchDocumentStore"):
        _resolve_document_store("not a document store", default)
