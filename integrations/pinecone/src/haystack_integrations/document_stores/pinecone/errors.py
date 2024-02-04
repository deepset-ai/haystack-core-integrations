from haystack.document_stores.errors import DocumentStoreError
from haystack.errors import FilterError


class PineconeDocumentStoreError(DocumentStoreError):
    pass


class PineconeDocumentStoreFilterError(FilterError):
    pass
