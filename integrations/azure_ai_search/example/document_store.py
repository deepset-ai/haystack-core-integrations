from haystack import Document

from haystack_integrations.document_stores.azure_ai_search import AzureAISearchDocumentStore

"""
This example demonstrates how to use the AzureAISearchDocumentStore to write and filter documents.
To run this example, you'll need an Azure Search service endpoint and API key, which can either be
set as environment variables (AZURE_AI_SEARCH_ENDPOINT and AZURE_AI_SEARCH_API_KEY) or
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
        content="This is an introduction to using Python for data analysis.",
        meta={"version": 1.0, "label": "chapter_one"},
    ),
    Document(
        content="Learn how to use Python libraries for machine learning.",
        meta={"version": 1.5, "label": "chapter_two"},
    ),
    Document(
        content="Advanced Python techniques for data visualization.",
        meta={"version": 2.0, "label": "chapter_three"},
    ),
]
document_store.write_documents(documents)

filters = {
    "operator": "AND",
    "conditions": [
        {"field": "meta.version", "operator": ">", "value": 1.2},
        {"field": "meta.label", "operator": "in", "value": ["chapter_one", "chapter_three"]},
    ],
}

results = document_store.filter_documents(filters)
print(results)
