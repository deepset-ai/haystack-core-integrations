# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.document_stores.errors import DocumentStoreError
from haystack.errors import FilterError


class VespaDocumentStoreError(DocumentStoreError):
    """Base exception for Vespa document store errors."""


class VespaDocumentStoreFilterError(FilterError, VespaDocumentStoreError):
    """Raised when Haystack filters cannot be translated to Vespa YQL."""


class VespaDocumentStoreConfigError(VespaDocumentStoreError):
    """Raised when the Vespa document store configuration is invalid."""
