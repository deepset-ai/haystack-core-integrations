# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.document_stores.oracle import OracleConnectionConfig

from ._base import _OracleEmbedderBase


@component
class OracleTextEmbedder(_OracleEmbedderBase):
    """
    Embeds strings with Oracle Database embedding functions.

    The component calls ``DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS`` with the configured
    Oracle embedding parameters and returns one dense vector for each input text.
    """

    def __init__(
        self,
        *,
        connection_config: OracleConnectionConfig,
        embedding_params: dict[str, Any] | None = None,
        use_connection_pool: bool = False,
        proxy: Secret | str | None = None,
    ) -> None:
        """
        Create an Oracle text embedder.

        :param connection_config: Oracle connection settings, including user, password, DSN, and optional wallet.
        :param embedding_params: JSON-serializable Oracle embedding parameters, such as provider and model.
        :param use_connection_pool: When ``True``, reuse a python-oracledb connection pool.
        :param proxy: Optional HTTP proxy set in the Oracle session with ``UTL_HTTP.SET_PROXY``.
        :raises ValueError: If ``connection_config`` or ``embedding_params`` is missing.
        """
        _OracleEmbedderBase.__init__(
            self,
            connection_config=connection_config,
            embedding_params=embedding_params,
            use_connection_pool=use_connection_pool,
            proxy=proxy,
        )

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    def run(self, text: str) -> dict[str, Any]:
        """
        Compute one embedding for a single input string.

        :param text: Text to embed.
        :returns: A dictionary containing ``embedding`` and ``meta``. ``meta`` contains the embedding parameters.
        :raises TypeError: If ``text`` is not a string.
        """
        if not isinstance(text, str):
            msg = "OracleTextEmbedder expects a string input."
            raise TypeError(msg)
        return {"embedding": self._embed_documents([text])[0], "meta": self.embedding_params}

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    async def run_async(self, text: str) -> dict[str, Any]:
        """
        Compute one embedding for a single input string asynchronously.

        :param text: Text to embed.
        :returns: A dictionary containing ``embedding`` and ``meta``. ``meta`` contains the embedding parameters.
        :raises TypeError: If ``text`` is not a string.
        """
        if not isinstance(text, str):
            msg = "OracleTextEmbedder expects a string input."
            raise TypeError(msg)
        return {"embedding": (await self._embed_documents_async([text]))[0], "meta": self.embedding_params}

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
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OracleTextEmbedder":
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
