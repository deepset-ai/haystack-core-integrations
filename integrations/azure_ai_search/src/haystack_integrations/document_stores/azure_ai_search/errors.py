from haystack.document_stores.errors import DocumentStoreError
from haystack.errors import FilterError


class AzureAISearchDocumentStoreError(DocumentStoreError):
    """Parent class for all AzureAISearchDocumentStore exceptions."""

    pass


class AzureAISearchDocumentStoreConfigError(AzureAISearchDocumentStoreError):
    """Raised when a configuration is not valid for a AzureAISearchDocumentStore."""

    pass


class AzureAISearchDocumentStoreFilterError(FilterError):
    """Raised when filter is not valid for AzureAISearchDocumentStore."""

    pass
