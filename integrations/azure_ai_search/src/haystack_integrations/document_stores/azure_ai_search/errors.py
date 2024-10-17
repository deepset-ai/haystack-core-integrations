from haystack.document_stores.errors import DocumentStoreError


class AzureAISearchDocumentStoreError(DocumentStoreError):
    """Parent class for all AzureAISearchDocumentStore exceptions."""

    pass


class AzureAISearchDocumentStoreConfigError(AzureAISearchDocumentStoreError):
    """Raised when a configuration is not valid for a AzureAISearchDocumentStore."""

    pass


class AzureAISearchDocumentStoreFilterError(DocumentStoreError):
    """Raised when filter is not valid for AzureAISearchDocumentStore."""

    pass
