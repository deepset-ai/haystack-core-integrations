# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


def _resolve_document_store(
    runtime_document_store: OpenSearchDocumentStore | None,
    default_document_store: OpenSearchDocumentStore,
) -> OpenSearchDocumentStore:
    """
    Return the runtime document store if provided and valid, otherwise the default one.

    :raises ValueError: If `runtime_document_store` is not None and not an OpenSearchDocumentStore.
    """
    if runtime_document_store is None:
        return default_document_store
    if not isinstance(runtime_document_store, OpenSearchDocumentStore):
        msg = "document_store must be an instance of OpenSearchDocumentStore"
        raise ValueError(msg)
    return runtime_document_store
