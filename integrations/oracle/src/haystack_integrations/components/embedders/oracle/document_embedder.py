# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.document_stores.oracle import OracleConnectionConfig

from ._base import _OracleEmbedderBase


@component
class OracleDocumentEmbedder(_OracleEmbedderBase):
    """
    Embeds Haystack Documents with Oracle Database embedding functions.

    The component embeds each document's content and can prepend selected metadata
    values before sending text to Oracle. It returns copied documents with the
    resulting vectors populated in ``Document.embedding``.
    """

    def __init__(
        self,
        *,
        connection_config: OracleConnectionConfig,
        embedding_params: dict[str, Any] | None = None,
        use_connection_pool: bool = False,
        proxy: Secret | str | None = None,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
    ) -> None:
        """
        Create an Oracle document embedder.

        :param connection_config: Oracle connection settings, including user, password, DSN, and optional wallet.
        :param embedding_params: JSON-serializable Oracle embedding parameters, such as provider and model.
        :param use_connection_pool: When ``True``, reuse a python-oracledb connection pool.
        :param proxy: Optional HTTP proxy set in the Oracle session with ``UTL_HTTP.SET_PROXY``.
        :param meta_fields_to_embed: Metadata keys to prepend to document content before embedding.
            Missing keys and ``None`` values are skipped.
        :param embedding_separator: Separator used between metadata values and document content.
        :raises ValueError: If ``connection_config`` or ``embedding_params`` is missing.
        """
        _OracleEmbedderBase.__init__(
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
        Compute embeddings and return documents with ``Document.embedding`` populated.

        :param documents: Documents to embed. Each item must be a Haystack ``Document``.
        :returns: A dictionary containing ``documents`` and ``meta``. The returned documents are copies of the input
            documents with ``embedding`` populated.
        :raises TypeError: If ``documents`` is not a list of Haystack ``Document`` objects.
        """
        if not isinstance(documents, list) or any(not isinstance(document, Document) for document in documents):
            msg = "OracleDocumentEmbedder expects a list of Document objects."
            raise TypeError(msg)
        embeddings = self._embed_documents(self._prepare_texts_to_embed(documents))
        embedded_documents = [
            replace(document, embedding=embedding) for document, embedding in zip(documents, embeddings, strict=True)
        ]
        return {"documents": embedded_documents, "meta": self.embedding_params}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, Any]:
        """
        Compute embeddings asynchronously and return documents with ``Document.embedding`` populated.

        :param documents: Documents to embed. Each item must be a Haystack ``Document``.
        :returns: A dictionary containing ``documents`` and ``meta``. The returned documents are copies of the input
            documents with ``embedding`` populated.
        :raises TypeError: If ``documents`` is not a list of Haystack ``Document`` objects.
        """
        if not isinstance(documents, list) or any(not isinstance(document, Document) for document in documents):
            msg = "OracleDocumentEmbedder expects a list of Document objects."
            raise TypeError(msg)
        embeddings = await self._embed_documents_async(self._prepare_texts_to_embed(documents))
        embedded_documents = [
            replace(document, embedding=embedding) for document, embedding in zip(documents, embeddings, strict=True)
        ]
        return {"documents": embedded_documents, "meta": self.embedding_params}

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleDocumentEmbedder":
        """
        Deserializes the component from a dictionary.
        """
        params = data.get("init_parameters", {})
        connection_config = params.get("connection_config")
        if isinstance(connection_config, Mapping):
            params["connection_config"] = OracleConnectionConfig.from_dict(dict(connection_config))
        if isinstance(params.get("proxy"), dict) and "type" in params["proxy"]:
            deserialize_secrets_inplace(params, keys=["proxy"])
        return default_from_dict(cls, data)
