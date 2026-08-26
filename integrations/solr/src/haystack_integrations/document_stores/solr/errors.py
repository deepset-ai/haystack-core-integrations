# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.document_stores.errors import DocumentStoreError


class SolrDocumentStoreError(DocumentStoreError):
    """Raised when a Solr request fails or returns an unexpected payload."""


class SolrDocumentStoreConfigError(SolrDocumentStoreError):
    """Raised when the Solr server or core is not configured in a way the document store can use."""
