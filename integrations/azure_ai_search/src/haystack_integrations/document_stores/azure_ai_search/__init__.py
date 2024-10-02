# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from .document_store import DEFAULT_VECTOR_SEARCH, AzureAISearchDocumentStore

__all__ = ["AzureAISearchDocumentStore", "DEFAULT_VECTOR_SEARCH"]
