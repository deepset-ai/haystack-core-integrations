# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import AsyncMock, patch

import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.components.embedders.twelvelabs import TwelveLabsDocumentEmbedder

PATCH_TARGET = "haystack_integrations.components.embedders.twelvelabs.document_embedder.embed_text"
ASYNC_PATCH_TARGET = "haystack_integrations.components.embedders.twelvelabs.document_embedder.embed_text_async"


def test_to_dict_and_from_dict(monkeypatch):
    monkeypatch.setenv("TWELVELABS_API_KEY", "tlk_env")
    data = TwelveLabsDocumentEmbedder().to_dict()
    assert data["type"].endswith("TwelveLabsDocumentEmbedder")
    restored = TwelveLabsDocumentEmbedder.from_dict(data)
    assert restored.model == "marengo3.0"


def test_run_sets_embeddings():
    embedder = TwelveLabsDocumentEmbedder(api_key=Secret.from_token("tlk_test"))
    docs = [Document(content="a cat playing piano"), Document(content="a dog")]
    with patch(PATCH_TARGET, return_value=[0.2] * 8) as mock_embed:
        result = embedder.run(documents=docs)
    assert mock_embed.call_count == 2
    assert all(d.embedding == [0.2] * 8 for d in result["documents"])
    assert result["meta"]["model"] == "marengo3.0"


def test_run_does_not_mutate_inputs():
    embedder = TwelveLabsDocumentEmbedder(api_key=Secret.from_token("tlk_test"))
    docs = [Document(content="a cat playing piano")]
    with patch(PATCH_TARGET, return_value=[0.2] * 8):
        result = embedder.run(documents=docs)
    # The input Document is left untouched; a new copy carries the embedding.
    assert docs[0].embedding is None
    assert result["documents"][0] is not docs[0]
    assert result["documents"][0].embedding == [0.2] * 8


def test_run_embeds_meta_fields_with_prefix_suffix():
    embedder = TwelveLabsDocumentEmbedder(
        api_key=Secret.from_token("tlk_test"),
        prefix="[",
        suffix="]",
        meta_fields_to_embed=["title"],
        embedding_separator=" | ",
    )
    docs = [Document(content="body", meta={"title": "headline"})]
    with patch(PATCH_TARGET, return_value=[0.3] * 8) as mock_embed:
        embedder.run(documents=docs)
    mock_embed.assert_called_once_with("[headline | body]", "marengo3.0", "tlk_test")


def test_run_rejects_non_documents():
    embedder = TwelveLabsDocumentEmbedder(api_key=Secret.from_token("tlk_test"))
    with pytest.raises(TypeError):
        embedder.run(documents="a string")


@pytest.mark.asyncio
async def test_run_async_sets_embeddings():
    embedder = TwelveLabsDocumentEmbedder(api_key=Secret.from_token("tlk_test"))
    docs = [Document(content="a cat playing piano"), Document(content="a dog")]
    with patch(ASYNC_PATCH_TARGET, new=AsyncMock(return_value=[0.2] * 8)) as mock_embed:
        result = await embedder.run_async(documents=docs)
    assert mock_embed.await_count == 2
    assert all(d.embedding == [0.2] * 8 for d in result["documents"])
    assert result["meta"]["model"] == "marengo3.0"
    # Inputs are left untouched (new Documents carry the embeddings).
    assert all(d.embedding is None for d in docs)


@pytest.mark.asyncio
async def test_run_async_embeds_meta_fields_with_prefix_suffix():
    embedder = TwelveLabsDocumentEmbedder(
        api_key=Secret.from_token("tlk_test"),
        prefix="[",
        suffix="]",
        meta_fields_to_embed=["title"],
        embedding_separator=" | ",
    )
    docs = [Document(content="body", meta={"title": "headline"})]
    with patch(ASYNC_PATCH_TARGET, new=AsyncMock(return_value=[0.3] * 8)) as mock_embed:
        await embedder.run_async(documents=docs)
    mock_embed.assert_awaited_once_with("[headline | body]", "marengo3.0", "tlk_test")


@pytest.mark.asyncio
async def test_run_async_rejects_non_documents():
    embedder = TwelveLabsDocumentEmbedder(api_key=Secret.from_token("tlk_test"))
    with pytest.raises(TypeError):
        await embedder.run_async(documents="a string")


@pytest.mark.skipif(not os.environ.get("TWELVELABS_API_KEY"), reason="TWELVELABS_API_KEY env var not set")
@pytest.mark.integration
def test_run_integration():
    embedder = TwelveLabsDocumentEmbedder()
    docs = [Document(content="a cat playing piano"), Document(content="a dog")]
    result = embedder.run(documents=docs)
    assert len(result["documents"]) == 2
    for doc in result["documents"]:
        assert isinstance(doc, Document)
        assert isinstance(doc.embedding, list)
        assert len(doc.embedding) > 0
        assert all(isinstance(x, int | float) for x in doc.embedding)
    assert result["meta"]["model"] == "marengo3.0"


@pytest.mark.skipif(not os.environ.get("TWELVELABS_API_KEY"), reason="TWELVELABS_API_KEY env var not set")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_async_integration():
    embedder = TwelveLabsDocumentEmbedder()
    docs = [Document(content="a cat playing piano"), Document(content="a dog")]
    result = await embedder.run_async(documents=docs)
    assert len(result["documents"]) == 2
    for doc in result["documents"]:
        assert isinstance(doc.embedding, list)
        assert len(doc.embedding) > 0
        assert all(isinstance(x, int | float) for x in doc.embedding)
