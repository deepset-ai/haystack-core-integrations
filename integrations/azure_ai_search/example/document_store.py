from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.azure_ai_search import AzureAISearchDocumentStore

"""
This example demonstrates how to use the AzureAISearchDocumentStore to write and filter documents.
To run this example, you'll need an Azure Search service endpoint and API key, which can either be
set as environment variables (AZURE_SEARCH_SERVICE_ENDPOINT and AZURE_SEARCH_API_KEY) or
provided directly to AzureAISearchDocumentStore(as params "api_key", "azure_endpoint").
Otherwise you can use DefaultAzureCredential to authenticate with Azure services.
See more details at https://learn.microsoft.com/en-us/azure/search/keyless-connections?tabs=python%2Cazure-cli
"""
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
