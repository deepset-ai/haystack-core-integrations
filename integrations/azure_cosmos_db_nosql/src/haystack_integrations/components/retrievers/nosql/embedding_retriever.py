from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict
from haystack.dataclasses import Document

from integrations.azure_cosmos_db_nosql.src.haystack_integrations.document_stores.nosql.document_store import AzureCosmosDBNoSqlDocumentStore


@component
class AzureCosmosDBNoSqlEmbeddingRetriever:
    def __init__(
            self,
            *,
            document_store: AzureCosmosDBNoSqlDocumentStore,
            filters: Optional[Dict[str, Any]] = None,
            top_k: int = 10,
    ):
        if not isinstance(document_store, AzureCosmosDBNoSqlDocumentStore):
            msg = "document_store must be an instance of AzureCosmosDBNoSqlDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AzureCosmosDBNoSqlDocumentStore":
        data["init_parameters"]["document_store"] = AzureCosmosDBNoSqlDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
            self,
            query_embedding: List[float],
            filters: Optional[Dict[str, Any]] = None,
            top_k: Optional[int] = None,
    ) -> Dict[str, List[Document]]:
        top_k = top_k or self.top_k

        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
        )
        return {"documents": docs}
