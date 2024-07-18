from haystack.document_stores.errors import DocumentStoreError


class AWSConfigurationError(DocumentStoreError):
    """Exception raised when AWS is not configured correctly"""
