from .document_store import AzureAISearchDocumentStore

from haystack import Document

#document_store = AzureAISearchDocumentStore(index_name= "novel", metadata_fields={"name": str, "age": int})
document_store = AzureAISearchDocumentStore(index_name="novel")
document_store.write_documents([
    Document(content="This is first", embedding=120),
    Document(content="This is second", embedding=8, meta= {"name": "Ron", "age": 30})
    ])
print(document_store.count_documents())