# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_to_dict
from haystack.dataclasses import Document

from haystack_integrations.document_stores.oracle import OracleConnectionConfig

from .text_embedder import OracleTextEmbedder


@component
class OracleDocumentEmbedder(OracleTextEmbedder):
    """
    Embeds Haystack Documents with Oracle Database embedding functions.
    """

    def __init__(
        self,
        *,
        connection_config: OracleConnectionConfig,
        embedding_params: dict[str, Any] | None = None,
        use_connection_pool: bool = False,
        proxy: Any | None = None,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
    ) -> None:
        OracleTextEmbedder.__init__(
            self,
            connection_config=connection_config,
            embedding_params=embedding_params,
            use_connection_pool=use_connection_pool,
            proxy=proxy,
        )
        self.meta_fields_to_embed = list(meta_fields_to_embed or [])
        self.embedding_separator = embedding_separator

    def _prepare_texts_to_embed(self, documents: list[Document]) -> list[str]:
        texts: list[str] = []
        for document in documents:
            meta_values = [
                str(document.meta[field]) for field in self.meta_fields_to_embed if document.meta.get(field) is not None
            ]
            texts.append(self.embedding_separator.join([*meta_values, document.content or ""]))
        return texts

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Compute embeddings and assign them to ``Document.embedding``.
        """
        if not isinstance(documents, list) or any(not isinstance(document, Document) for document in documents):
            msg = "OracleDocumentEmbedder expects a list of Document objects."
            raise TypeError(msg)
        embeddings = self._embed_documents(self._prepare_texts_to_embed(documents))
        for document, embedding in zip(documents, embeddings, strict=True):
            document.embedding = embedding
        return {"documents": documents, "meta": self.embedding_params}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, Any]:
        """
        Compute embeddings asynchronously and assign them to ``Document.embedding``.
        """
        if not isinstance(documents, list) or any(not isinstance(document, Document) for document in documents):
            msg = "OracleDocumentEmbedder expects a list of Document objects."
            raise TypeError(msg)
        embeddings = await self._embed_documents_async(self._prepare_texts_to_embed(documents))
        for document, embedding in zip(documents, embeddings, strict=True):
            document.embedding = embedding
        return {"documents": documents, "meta": self.embedding_params}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.
        """
        return default_to_dict(
            self,
            connection_config=self.connection_config.to_dict(),
            embedding_params=self.embedding_params,
            use_connection_pool=self.use_connection_pool,
            proxy=self._serialize_proxy(),
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )
