# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses import Document
from haystack.utils import Secret

from haystack_integrations.components.embedders.oracle import OracleDocumentEmbedder, OracleTextEmbedder
from haystack_integrations.document_stores.oracle import OracleConnectionConfig


def _connection_config():
    return OracleConnectionConfig(
        user=Secret.from_env_var("ORACLE_USER", strict=False),
        password=Secret.from_env_var("ORACLE_PASSWORD", strict=False),
        dsn=Secret.from_env_var("ORACLE_DSN", strict=False),
    )


def test_text_embedder_requires_connection_config():
    with pytest.raises(ValueError, match="connection_config"):
        OracleTextEmbedder(
            connection_config=None,
            embedding_params={"provider": "database", "model": "demo"},
        )


def test_text_embedder_run_returns_single_embedding(monkeypatch):
    embedder = OracleTextEmbedder(
        connection_config=_connection_config(),
        embedding_params={"provider": "database", "model": "demo"},
    )

    def embed_documents(texts):
        assert texts == ["hello"]
        return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(embedder, "_embed_documents", embed_documents)

    result = embedder.run("hello")

    assert result["embedding"] == [0.1, 0.2, 0.3]
    assert result["meta"] == {"provider": "database", "model": "demo"}


def test_text_embedder_rejects_non_string():
    embedder = OracleTextEmbedder(
        connection_config=_connection_config(),
        embedding_params={"provider": "database", "model": "demo"},
    )
    with pytest.raises(TypeError, match="expects a string"):
        embedder.run(["not text"])


@pytest.mark.asyncio
async def test_text_embedder_async_awaits_gettype(monkeypatch):
    class FakeLob:
        def __init__(self):
            self.value = None

        async def write(self, value):
            self.value = value

    class FakeVectorArray(list):
        pass

    class FakeVectorArrayType:
        def newobject(self):
            return FakeVectorArray()

    class FakeCursor:
        def __init__(self):
            self.rows = iter([('{"embed_vector": "[0.1, 0.2, 0.3]"}',)])

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def setinputsizes(self, *_):
            return None

        async def execute(self, statement, params):
            assert "UTL_TO_EMBEDDINGS" in statement
            assert len(params[0]) == 1
            assert params[0][0].value == '{"chunk_id": 1, "chunk_data": "hello"}'

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.rows)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeConnection:
        def cursor(self):
            return FakeCursor()

        async def gettype(self, name):
            assert name == "SYS.VECTOR_ARRAY_T"
            return FakeVectorArrayType()

        async def createlob(self, *_):
            return FakeLob()

    class FakeConnectionContext:
        async def __aenter__(self):
            return FakeConnection()

        async def __aexit__(self, *_):
            return None

    embedder = OracleTextEmbedder(
        connection_config=_connection_config(),
        embedding_params={"provider": "database", "model": "demo"},
    )

    async def connection_context_async():
        return FakeConnectionContext()

    monkeypatch.setattr(embedder, "_connection_context_async", connection_context_async)

    result = await embedder.run_async("hello")

    assert result["embedding"] == [0.1, 0.2, 0.3]


def test_document_embedder_prepares_metadata_and_content():
    embedder = OracleDocumentEmbedder(
        connection_config=_connection_config(),
        embedding_params={"provider": "database", "model": "demo"},
        meta_fields_to_embed=["title", "missing"],
        embedding_separator=" | ",
    )

    texts = embedder._prepare_texts_to_embed([Document(content="body", meta={"title": "heading"})])

    assert texts == ["heading | body"]


def test_document_embedder_sets_document_embeddings(monkeypatch):
    embedder = OracleDocumentEmbedder(
        connection_config=_connection_config(),
        embedding_params={"provider": "database", "model": "demo"},
    )

    def embed_documents(texts):
        assert texts == ["one", "two"]
        return [[0.4, 0.5], [0.6, 0.7]]

    monkeypatch.setattr(embedder, "_embed_documents", embed_documents)
    documents = [Document(content="one"), Document(content="two")]

    result = embedder.run(documents)

    assert result["documents"] is documents
    assert documents[0].embedding == [0.4, 0.5]
    assert documents[1].embedding == [0.6, 0.7]


def test_embedder_to_dict_keeps_connection_config_secret_structured():
    embedder = OracleTextEmbedder(
        connection_config=_connection_config(),
        embedding_params={"provider": "database", "model": "demo"},
    )

    data = embedder.to_dict()

    assert "connection_params" not in data["init_parameters"]
    password = data["init_parameters"]["connection_config"]["password"]
    assert password["type"] == "env_var"
