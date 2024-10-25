from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.azure_ai_search import AzureAISearchDocumentStore

document_store = AzureAISearchDocumentStore(
    metadata_fields={"version": float, "label": str},
    index_name="document-store-example",
)

documents = [
    Document(
        content="Use pip to install a basic version of Haystack's latest release: pip install farm-haystack.",
        meta={"version": 1.15, "label": "first"},
    ),
    Document(
        content="Use pip to install a Haystack's latest release: pip install farm-haystack[inference].",
        meta={"version": 1.22, "label": "second"},
    ),
    Document(
        content="Use pip to install only the Haystack 2.0 code: pip install haystack-ai.",
        meta={"version": 2.0, "label": "third"},
    ),
]
document_store.write_documents(documents, policy=DuplicatePolicy.SKIP)

filters = {
    "operator": "AND",
    "conditions": [
        {"field": "meta.version", "operator": ">", "value": 1.21},
        {"field": "meta.label", "operator": "in", "value": ["first", "third"]},
    ],
}

results = document_store.filter_documents(filters)
for doc in results:
    print(doc)
