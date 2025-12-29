# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.document_stores.errors import DocumentStoreError
from haystack.errors import FilterError


class AstraDocumentStoreError(DocumentStoreError):
    """Parent class for all AstraDocumentStore errors."""

    pass


class AstraDocumentStoreFilterError(FilterError):
    """Raised when an invalid filter is passed to AstraDocumentStore."""

    pass


class AstraDocumentStoreConfigError(AstraDocumentStoreError):
    """Raised when an invalid configuration is passed to AstraDocumentStore."""

    pass
