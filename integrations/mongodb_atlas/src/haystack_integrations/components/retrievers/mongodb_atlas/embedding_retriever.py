# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


@component
class MongoDBAtlasEmbeddingRetriever:
    """
    Retrieves documents from the MongoDBAtlasDocumentStore by embedding similarity.

    The similarity is dependent on the vector_search_index used in the MongoDBAtlasDocumentStore and the chosen metric
    during the creation of the index (i.e. cosine, dot product, or euclidean). See MongoDBAtlasDocumentStore for more
    information.

    Usage example:
    ```python
    import numpy as np
    from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
    from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever

    store = MongoDBAtlasDocumentStore(database_name="haystack_integration_test",
                                      collection_name="test_embeddings_collection",
                                      vector_search_index="cosine_index")
    retriever = MongoDBAtlasEmbeddingRetriever(document_store=store)

    results = retriever.run(query_embedding=np.random.random(768).tolist())
    print(results["documents"])
    ```

    The example above retrieves the 10 most similar documents to a random query embedding from the
    MongoDBAtlasDocumentStore. Note that dimensions of the query_embedding must match the dimensions of the embeddings
    stored in the MongoDBAtlasDocumentStore.
    """

    def __init__(
        self,
        *,
        document_store: MongoDBAtlasDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ):
        """
        Create the MongoDBAtlasDocumentStore component.

        :param document_store: An instance of MongoDBAtlasDocumentStore.
        :param filters: Filters applied to the retrieved Documents. Make sure that the fields used in the filters are
            included in the configuration of the `vector_search_index`. The configuration must be done manually
            in the Web UI of MongoDB Atlas.
        :param top_k: Maximum number of Documents to return.

        :raises ValueError: If `document_store` is not an instance of `MongoDBAtlasDocumentStore`.
        """
        if not isinstance(document_store, MongoDBAtlasDocumentStore):
            msg = "document_store must be an instance of MongoDBAtlasDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MongoDBAtlasEmbeddingRetriever":
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
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, List[Document]]:
        """
        Retrieve documents from the MongoDBAtlasDocumentStore, based on the provided embedding similarity.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents. Overrides the value specified at initialization.
        :param top_k: Maximum number of Documents to return. Overrides the value specified at initialization.
        :returns: A dictionary with the following keys:
            - `documents`: List of Documents most similar to the given `query_embedding`
        """
        filters = filters or self.filters
        top_k = top_k or self.top_k

        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
        )
        return {"documents": docs}
