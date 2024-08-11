import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional
from haystack.dataclasses import Document
from haystack.components.embedders import AzureOpenAIDocumentEmbedder, AzureOpenAITextEmbedder
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    AzureOpenAIParameters
)

azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

def create_vectorizer(embedding_model_name: str) -> List[AzureOpenAIVectorizer]:
    """Known values are: "text-embedding-ada-002", "text-embedding-3-large", and "text-embedding-3-small"."""
    vectorizer= AzureOpenAIVectorizer(
            name="myVectorizer",
            azure_open_ai_parameters=AzureOpenAIParameters(
                resource_uri=azure_openai_endpoint,
                deployment_id=azure_openai_embedding_deployment,
                model_name=embedding_model_name,
                api_key=azure_openai_key
            )
        )
    return [vectorizer]


