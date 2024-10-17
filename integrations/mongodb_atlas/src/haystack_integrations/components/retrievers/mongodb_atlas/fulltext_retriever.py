from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


@component
class MongoDBAtlasFullTextRetriever:

    def __init__(
        self,
        *,
        document_store: MongoDBAtlasDocumentStore,
        search_path: Union[str, List[str]] = "content",
        top_k: int = 10,
    ):
        """
        Create the MongoDBAtlasFullTextRetriever component.

        :param document_store: An instance of MongoDBAtlasDocumentStore.
        :param search_path: Field(s) to search within, e.g., "content" or ["content", "title"].
        :param top_k: Maximum number of Documents to return.
        :raises ValueError: If `document_store` is not an instance of `MongoDBAtlasDocumentStore`.
        """

        if not isinstance(document_store, MongoDBAtlasDocumentStore):
            msg = "document_store must be an instance of MongoDBAtlasDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.top_k = top_k
        self.search_path = search_path

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            top_k=self.top_k,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MongoDBAtlasFullTextRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
              Deserialized component.
        """
        data["init_parameters"]["document_store"] = MongoDBAtlasDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents from the MongoDBAtlasDocumentStore, based on the provided query.

        :param query: Text query.
        :param top_k: Maximum number of Documents to return. Overrides the value specified at initialization.
        :returns: A dictionary with the following keys:
            - `documents`: List of Documents most similar to the given `query`
        """
        top_k = top_k or self.top_k

        docs = self.document_store._fulltext_retrieval(query=query, top_k=top_k, search_path=self.search_path)
        return {"documents": docs}
