# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
import logging
from collections.abc import Mapping
from typing import Any

import oracledb
from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.document_stores.oracle import OracleConnectionConfig

logger = logging.getLogger(__name__)


def _resolve_secret(value: Any) -> Any:
    if isinstance(value, Secret):
        return value.resolve_value()
    return value


def _serialize_secret(value: Any) -> Any:
    if isinstance(value, Secret):
        return value.to_dict()
    return value


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@component
class OracleTextEmbedder:
    """
    Embeds strings with Oracle Database embedding functions.
    """

    def __init__(
        self,
        *,
        connection_config: OracleConnectionConfig,
        embedding_params: dict[str, Any] | None = None,
        use_connection_pool: bool = False,
        proxy: Secret | str | None = None,
    ) -> None:
        if connection_config is None:
            msg = "connection_config must be provided."
            raise ValueError(msg)
        if embedding_params is None:
            msg = "embedding_params must be provided."
            raise ValueError(msg)

        self.connection_config = connection_config
        self.embedding_params = dict(embedding_params)
        self.use_connection_pool = use_connection_pool
        self.proxy = proxy

        self._client: Any | None = None
        self._client_async: Any | None = None

    def _connect_kwargs(self, *, pool_options: bool) -> dict[str, Any]:
        cfg = self.connection_config
        password = cfg.password.resolve_value()
        connect_kwargs: dict[str, Any] = {
            "user": cfg.user.resolve_value(),
            "password": password,
            "dsn": cfg.dsn.resolve_value(),
        }
        if pool_options:
            connect_kwargs["min"] = cfg.min_connections
            connect_kwargs["max"] = cfg.max_connections
            connect_kwargs["increment"] = 1
        if cfg.wallet_location:
            connect_kwargs["config_dir"] = cfg.wallet_location
            connect_kwargs["wallet_location"] = cfg.wallet_location
            connect_kwargs["wallet_password"] = cfg.wallet_password.resolve_value() if cfg.wallet_password else password
        return connect_kwargs

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self.use_connection_pool:
            self._client = oracledb.create_pool(**self._connect_kwargs(pool_options=True))
        else:
            self._client = oracledb.connect(**self._connect_kwargs(pool_options=False))
        return self._client

    def _connection_context(self) -> Any:
        if self.use_connection_pool:
            return self._ensure_client().acquire()
        return oracledb.connect(**self._connect_kwargs(pool_options=False))

    async def _ensure_client_async(self) -> Any:
        if self._client_async is not None:
            return self._client_async
        if self.use_connection_pool:
            create_pool_async = getattr(oracledb, "create_pool_async", None)
            if create_pool_async is None:
                msg = "python-oracledb does not provide create_pool_async."
                raise RuntimeError(msg)
            pool = create_pool_async(**self._connect_kwargs(pool_options=True))
            self._client_async = await pool if inspect.isawaitable(pool) else pool
        else:
            self._client_async = await oracledb.connect_async(**self._connect_kwargs(pool_options=False))
        return self._client_async

    async def _connection_context_async(self) -> Any:
        if self.use_connection_pool:
            return (await self._ensure_client_async()).acquire()
        return await oracledb.connect_async(**self._connect_kwargs(pool_options=False))

    def _serialize_proxy(self) -> Any:
        return _serialize_secret(self.proxy)

    def _proxy_value(self) -> str | None:
        proxy = _resolve_secret(self.proxy)
        return str(proxy) if proxy else None

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        oracledb.defaults.fetch_lobs = False
        embeddings: list[list[float]] = []

        with self._connection_context() as connection, connection.cursor() as cursor:
            proxy_was_set = False
            proxy = self._proxy_value()
            if proxy:
                cursor.execute("BEGIN UTL_HTTP.SET_PROXY(:proxy); END;", proxy=proxy)
                proxy_was_set = True
            try:
                vector_array_type = connection.gettype("SYS.VECTOR_ARRAY_T")
                chunks = [json.dumps({"chunk_id": index, "chunk_data": text}) for index, text in enumerate(texts, 1)]
                inputs = vector_array_type.newobject(chunks)
                cursor.setinputsizes(None, oracledb.DB_TYPE_JSON)
                cursor.execute(
                    "SELECT t.* FROM DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS(:1, JSON(:2)) t",
                    [inputs, self.embedding_params],
                )
                for row in cursor:
                    if row is None:
                        embeddings.append([])
                        continue
                    row_data = json.loads(row[0])
                    embeddings.append(json.loads(row_data["embed_vector"]))
            except BaseException as exc:
                if proxy_was_set:
                    self._clear_proxy(cursor, exc)
                raise
            else:
                if proxy_was_set:
                    self._clear_proxy(cursor, None)
        return embeddings

    async def _embed_documents_async(self, texts: list[str]) -> list[list[float]]:
        oracledb.defaults.fetch_lobs = False
        embeddings: list[list[float]] = []

        connection_context = await self._connection_context_async()
        async with connection_context as connection:
            with connection.cursor() as cursor:
                proxy_was_set = False
                proxy = self._proxy_value()
                if proxy:
                    await _maybe_await(cursor.execute("BEGIN UTL_HTTP.SET_PROXY(:proxy); END;", proxy=proxy))
                    proxy_was_set = True
                try:
                    vector_array_type = await _maybe_await(connection.gettype("SYS.VECTOR_ARRAY_T"))
                    chunks = [
                        json.dumps({"chunk_id": index, "chunk_data": text}) for index, text in enumerate(texts, 1)
                    ]
                    inputs = vector_array_type.newobject()
                    for chunk in chunks:
                        clob = await _maybe_await(connection.createlob(oracledb.DB_TYPE_CLOB))
                        await _maybe_await(clob.write(chunk))
                        inputs.append(clob)
                    cursor.setinputsizes(None, oracledb.DB_TYPE_JSON)
                    await _maybe_await(
                        cursor.execute(
                            "SELECT t.* FROM DBMS_VECTOR_CHAIN.UTL_TO_EMBEDDINGS(:1, JSON(:2)) t",
                            [inputs, self.embedding_params],
                        )
                    )
                    async for row in cursor:
                        if row is None:
                            embeddings.append([])
                            continue
                        row_data = json.loads(row[0])
                        embeddings.append(json.loads(row_data["embed_vector"]))
                except BaseException as exc:
                    if proxy_was_set:
                        await self._clear_proxy_async(cursor, exc)
                    raise
                else:
                    if proxy_was_set:
                        await self._clear_proxy_async(cursor, None)
        return embeddings

    @staticmethod
    def _clear_proxy(cursor: Any, original_error: BaseException | None) -> None:
        try:
            cursor.execute("BEGIN UTL_HTTP.SET_PROXY(:proxy); END;", proxy=None)
        except Exception as cleanup_error:
            logger.exception("Failed to clear Oracle session proxy.")
            if original_error is not None:
                msg = "Failed to clear Oracle session proxy after embedding failed."
                raise RuntimeError(msg) from cleanup_error
            msg = "Failed to clear Oracle session proxy after embedding succeeded."
            raise RuntimeError(msg) from cleanup_error

    @staticmethod
    async def _clear_proxy_async(cursor: Any, original_error: BaseException | None) -> None:
        try:
            await _maybe_await(cursor.execute("BEGIN UTL_HTTP.SET_PROXY(:proxy); END;", proxy=None))
        except Exception as cleanup_error:
            logger.exception("Failed to clear Oracle session proxy.")
            if original_error is not None:
                msg = "Failed to clear Oracle session proxy after async embedding failed."
                raise RuntimeError(msg) from cleanup_error
            msg = "Failed to clear Oracle session proxy after async embedding succeeded."
            raise RuntimeError(msg) from cleanup_error

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    def run(self, text: str) -> dict[str, Any]:
        """
        Compute one embedding for a single input string.
        """
        if not isinstance(text, str):
            msg = "OracleTextEmbedder expects a string input."
            raise TypeError(msg)
        return {"embedding": self._embed_documents([text])[0], "meta": self.embedding_params}

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    async def run_async(self, text: str) -> dict[str, Any]:
        """
        Compute one embedding for a single input string asynchronously.
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
