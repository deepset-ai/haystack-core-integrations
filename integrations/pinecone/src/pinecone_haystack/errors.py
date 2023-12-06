from haystack.preview.document_stores.errors import DocumentStoreError
from haystack.preview.errors import FilterError


class PineconeDocumentStoreError(DocumentStoreError):
    pass


class PineconeDocumentStoreFilterError(FilterError):
    pass
