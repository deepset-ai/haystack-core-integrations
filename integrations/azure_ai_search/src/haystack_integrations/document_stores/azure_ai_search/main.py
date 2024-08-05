from .document_store import AzureAISearchDocumentStore

from haystack import Document

#document_store = AzureAISearchDocumentStore(index_name= "novel", metadata_fields={"name": str, "age": int})
document_store = AzureAISearchDocumentStore(index_name="novel", metadata_fields={"name": str, "age": int})
document_store.write_documents([
    Document(content="This is first" ),
    Document(content="This is second", meta= {"name": "Ron", "age": 30})
    ])
print(document_store.count_documents())