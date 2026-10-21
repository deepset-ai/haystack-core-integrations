# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import AsyncMock, patch

import pytest
from haystack.utils import Secret

from haystack_integrations.components.embedders.twelvelabs import TwelveLabsMultimodalEmbedder

PATCH_TARGET = "haystack_integrations.components.embedders.twelvelabs.multimodal_embedder.embed_media"
ASYNC_PATCH_TARGET = "haystack_integrations.components.embedders.twelvelabs.multimodal_embedder.embed_media_async"


def test_init_defaults(monkeypatch):
    monkeypatch.setenv("TWELVELABS_API_KEY", "tlk_env")
    embedder = TwelveLabsMultimodalEmbedder()
    assert embedder.model == "marengo3.0"
    assert embedder.api_key.resolve_value() == "tlk_env"


def test_to_dict_and_from_dict(monkeypatch):
    monkeypatch.setenv("TWELVELABS_API_KEY", "tlk_env")
    data = TwelveLabsMultimodalEmbedder(model="marengo3.0").to_dict()
    assert data["type"].endswith("TwelveLabsMultimodalEmbedder")
    assert data["init_parameters"]["model"] == "marengo3.0"
    restored = TwelveLabsMultimodalEmbedder.from_dict(data)
    assert restored.model == "marengo3.0"


def test_run_infers_image():
    embedder = TwelveLabsMultimodalEmbedder(api_key=Secret.from_token("tlk_test"))
    with patch(PATCH_TARGET, return_value=[0.1] * 8) as mock_embed:
        result = embedder.run(source="https://ex.com/cat.jpg")
    mock_embed.assert_called_once_with("https://ex.com/cat.jpg", "image", "marengo3.0", "tlk_test")
    assert result["embedding"] == [0.1] * 8
    assert result["meta"] == {"model": "marengo3.0", "modality": "image"}


def test_run_explicit_modality_overrides_inference():
    embedder = TwelveLabsMultimodalEmbedder(api_key=Secret.from_token("tlk_test"))
    with patch(PATCH_TARGET, return_value=[0.2] * 4) as mock_embed:
        result = embedder.run(source="https://ex.com/stream", modality="video")
    mock_embed.assert_called_once_with("https://ex.com/stream", "video", "marengo3.0", "tlk_test")
    assert result["meta"]["modality"] == "video"


def test_run_rejects_non_string_source():
    embedder = TwelveLabsMultimodalEmbedder(api_key=Secret.from_token("tlk_test"))
    with pytest.raises(TypeError):
        embedder.run(source=123)


def test_run_uninferrable_modality_raises():
    embedder = TwelveLabsMultimodalEmbedder(api_key=Secret.from_token("tlk_test"))
    with pytest.raises(ValueError):
        embedder.run(source="mystery.xyz")


def test_run_bad_explicit_modality_raises():
    embedder = TwelveLabsMultimodalEmbedder(api_key=Secret.from_token("tlk_test"))
    with pytest.raises(ValueError):
        embedder.run(source="cat.jpg", modality="hologram")


@pytest.mark.asyncio
async def test_run_async_infers_audio():
    embedder = TwelveLabsMultimodalEmbedder(api_key=Secret.from_token("tlk_test"))
    with patch(ASYNC_PATCH_TARGET, new=AsyncMock(return_value=[0.3] * 8)) as mock_embed:
        result = await embedder.run_async(source="song.mp3")
    mock_embed.assert_awaited_once_with("song.mp3", "audio", "marengo3.0", "tlk_test")
    assert result["embedding"] == [0.3] * 8
    assert result["meta"]["modality"] == "audio"


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("TWELVELABS_API_KEY"), reason="TWELVELABS_API_KEY env var not set")
def test_live_image_embedding():
    embedder = TwelveLabsMultimodalEmbedder()
    result = embedder.run(source="https://upload.wikimedia.org/wikipedia/commons/1/15/Cat_August_2010-4.jpg")
    assert isinstance(result["embedding"], list)
    assert len(result["embedding"]) > 0
    assert result["meta"]["modality"] == "image"
