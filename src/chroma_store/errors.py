# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.preview.document_stores.errors import StoreError, FilterError


class ChromaDocumentStoreFilterError(FilterError):
    pass


class ChromaDocumentStoreConfigError(StoreError):
    pass
