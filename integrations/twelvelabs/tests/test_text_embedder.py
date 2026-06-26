# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import AsyncMock, patch

import pytest
from haystack.utils import Secret

from haystack_integrations.components.embedders.twelvelabs import TwelveLabsTextEmbedder

PATCH_TARGET = "haystack_integrations.components.embedders.twelvelabs.text_embedder.embed_text"
ASYNC_PATCH_TARGET = "haystack_integrations.components.embedders.twelvelabs.text_embedder.embed_text_async"


def test_init_defaults(monkeypatch):
    monkeypatch.setenv("TWELVELABS_API_KEY", "tlk_env")
    embedder = TwelveLabsTextEmbedder()
    assert embedder.model == "marengo3.0"
    assert embedder.api_key.resolve_value() == "tlk_env"


def test_to_dict_and_from_dict(monkeypatch):
    monkeypatch.setenv("TWELVELABS_API_KEY", "tlk_env")
    data = TwelveLabsTextEmbedder(model="marengo3.0").to_dict()
    assert data["type"].endswith("TwelveLabsTextEmbedder")
    assert data["init_parameters"]["model"] == "marengo3.0"
    restored = TwelveLabsTextEmbedder.from_dict(data)
    assert restored.model == "marengo3.0"


def test_run():
    embedder = TwelveLabsTextEmbedder(api_key=Secret.from_token("tlk_test"))
    with patch(PATCH_TARGET, return_value=[0.1] * 8) as mock_embed:
        result = embedder.run(text="a cat playing piano")
    mock_embed.assert_called_once_with("a cat playing piano", "marengo3.0", "tlk_test")
    assert result["embedding"] == [0.1] * 8
    assert result["meta"]["model"] == "marengo3.0"


def test_run_applies_prefix_and_suffix():
    embedder = TwelveLabsTextEmbedder(api_key=Secret.from_token("tlk_test"), prefix="Q: ", suffix=" ?")
    with patch(PATCH_TARGET, return_value=[0.1] * 8) as mock_embed:
        embedder.run(text="a cat")
    mock_embed.assert_called_once_with("Q: a cat ?", "marengo3.0", "tlk_test")


def test_prefix_suffix_round_trip(monkeypatch):
    monkeypatch.setenv("TWELVELABS_API_KEY", "tlk_env")
    data = TwelveLabsTextEmbedder(prefix="p", suffix="s").to_dict()
    assert data["init_parameters"]["prefix"] == "p"
    assert data["init_parameters"]["suffix"] == "s"
    restored = TwelveLabsTextEmbedder.from_dict(data)
    assert restored.prefix == "p"
    assert restored.suffix == "s"


def test_run_rejects_non_string():
    embedder = TwelveLabsTextEmbedder(api_key=Secret.from_token("tlk_test"))
    with pytest.raises(TypeError):
        embedder.run(text=["not", "a", "string"])


@pytest.mark.asyncio
async def test_run_async():
    embedder = TwelveLabsTextEmbedder(api_key=Secret.from_token("tlk_test"))
    with patch(ASYNC_PATCH_TARGET, new=AsyncMock(return_value=[0.1] * 8)) as mock_embed:
        result = await embedder.run_async(text="a cat playing piano")
    mock_embed.assert_awaited_once_with("a cat playing piano", "marengo3.0", "tlk_test")
    assert result["embedding"] == [0.1] * 8
    assert result["meta"]["model"] == "marengo3.0"


@pytest.mark.asyncio
async def test_run_async_applies_prefix_and_suffix():
    embedder = TwelveLabsTextEmbedder(api_key=Secret.from_token("tlk_test"), prefix="Q: ", suffix=" ?")
    with patch(ASYNC_PATCH_TARGET, new=AsyncMock(return_value=[0.1] * 8)) as mock_embed:
        await embedder.run_async(text="a cat")
    mock_embed.assert_awaited_once_with("Q: a cat ?", "marengo3.0", "tlk_test")


@pytest.mark.asyncio
async def test_run_async_rejects_non_string():
    embedder = TwelveLabsTextEmbedder(api_key=Secret.from_token("tlk_test"))
    with pytest.raises(TypeError):
        await embedder.run_async(text=["not", "a", "string"])


@pytest.mark.skipif(not os.environ.get("TWELVELABS_API_KEY"), reason="TWELVELABS_API_KEY env var not set")
@pytest.mark.integration
def test_run_integration():
    embedder = TwelveLabsTextEmbedder()
    result = embedder.run(text="a cat playing piano")
    assert isinstance(result["embedding"], list)
    assert len(result["embedding"]) > 0
    assert all(isinstance(x, int | float) for x in result["embedding"])
    assert result["meta"]["model"] == "marengo3.0"


@pytest.mark.skipif(not os.environ.get("TWELVELABS_API_KEY"), reason="TWELVELABS_API_KEY env var not set")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_async_integration():
    embedder = TwelveLabsTextEmbedder()
    result = await embedder.run_async(text="a cat playing piano")
    assert isinstance(result["embedding"], list)
    assert len(result["embedding"]) > 0
    assert all(isinstance(x, int | float) for x in result["embedding"])
