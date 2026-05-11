# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from .document_store import SupabasePgvectorDocumentStore
from .groonga_document_store import SupabaseGroongaDocumentStore

__all__ = [
    "SupabaseGroongaDocumentStore",
    "SupabasePgvectorDocumentStore",
]
