# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from haystack_integrations.components.embedders.twelvelabs._media_embed import (
    detect_modality,
    embed_media,
    embed_media_async,
)

MODULE = "haystack_integrations.components.embedders.twelvelabs._media_embed"


@pytest.mark.parametrize(
    "source,expected",
    [
        ("cat.jpg", "image"),
        ("/data/photo.PNG", "image"),
        ("https://ex.com/pic.jpeg?v=1", "image"),
        ("song.mp3", "audio"),
        ("clip.wav", "audio"),
        ("movie.mp4", "video"),
        ("https://ex.com/v.mov", "video"),
    ],
)
def test_detect_modality(source, expected):
    assert detect_modality(source) == expected


def test_detect_modality_unknown_raises():
    with pytest.raises(ValueError):
        detect_modality("mystery.xyz")


def _segment(vector, scope=None):
    seg = MagicMock()
    seg.float_ = vector
    seg.embedding_scope = scope
    return seg


def test_embed_media_image_url():
    with patch(f"{MODULE}.TwelveLabs") as cls:
        client = cls.return_value
        response = MagicMock()
        response.image_embedding.segments = [_segment([0.1, 0.2, 0.3])]
        client.embed.create.return_value = response
        out = embed_media("https://ex.com/cat.jpg", "image", "marengo3.0", "tlk")
    cls.assert_called_once_with(api_key="tlk")
    client.embed.create.assert_called_once_with(model_name="marengo3.0", image_url="https://ex.com/cat.jpg")
    assert out == [0.1, 0.2, 0.3]


def test_embed_media_audio_local_file(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"xx")
    with patch(f"{MODULE}.TwelveLabs") as cls:
        client = cls.return_value
        response = MagicMock()
        response.audio_embedding.segments = [_segment([0.5, 0.6])]
        client.embed.create.return_value = response
        out = embed_media(str(audio), "audio", "marengo3.0", "tlk")
    _, kwargs = client.embed.create.call_args
    assert kwargs["model_name"] == "marengo3.0"
    assert "audio_file" in kwargs
    assert out == [0.5, 0.6]


def test_embed_media_video_polls_until_ready():
    with patch(f"{MODULE}.TwelveLabs") as cls, patch(f"{MODULE}.time.sleep") as sleep:
        client = cls.return_value
        client.embed.tasks.create.return_value = MagicMock(id="task-1")
        client.embed.tasks.status.side_effect = [MagicMock(status="processing"), MagicMock(status="ready")]
        retrieved = MagicMock()
        retrieved.video_embedding.segments = [_segment([0.7, 0.8], scope="video")]
        client.embed.tasks.retrieve.return_value = retrieved
        out = embed_media("https://ex.com/v.mp4", "video", "marengo3.0", "tlk")
    client.embed.tasks.create.assert_called_once_with(
        model_name="marengo3.0", video_url="https://ex.com/v.mp4", video_embedding_scope=["video"]
    )
    assert client.embed.tasks.status.call_count == 2
    client.embed.tasks.retrieve.assert_called_once_with("task-1")
    sleep.assert_called_once()
    assert out == [0.7, 0.8]


def test_embed_media_video_failed_raises():
    with patch(f"{MODULE}.TwelveLabs") as cls, patch(f"{MODULE}.time.sleep"):
        client = cls.return_value
        client.embed.tasks.create.return_value = MagicMock(id="task-1")
        client.embed.tasks.status.return_value = MagicMock(status="failed")
        with pytest.raises(RuntimeError):
            embed_media("https://ex.com/v.mp4", "video", "marengo3.0", "tlk")


def test_embed_media_no_vector_raises():
    with patch(f"{MODULE}.TwelveLabs") as cls:
        client = cls.return_value
        response = MagicMock()
        response.image_embedding.segments = None
        client.embed.create.return_value = response
        with pytest.raises(RuntimeError):
            embed_media("https://ex.com/cat.jpg", "image", "marengo3.0", "tlk")


def test_embed_media_missing_local_file_raises():
    with patch(f"{MODULE}.TwelveLabs"), pytest.raises(FileNotFoundError):
        embed_media("/no/such/file.jpg", "image", "marengo3.0", "tlk")


@pytest.mark.asyncio
async def test_embed_media_async_image_url():
    with patch(f"{MODULE}.AsyncTwelveLabs") as cls:
        client = cls.return_value
        response = MagicMock()
        response.image_embedding.segments = [_segment([0.1, 0.2])]
        client.embed.create = AsyncMock(return_value=response)
        out = await embed_media_async("https://ex.com/cat.jpg", "image", "marengo3.0", "tlk")
    client.embed.create.assert_awaited_once_with(model_name="marengo3.0", image_url="https://ex.com/cat.jpg")
    assert out == [0.1, 0.2]


@pytest.mark.asyncio
async def test_embed_media_async_video_polls_until_ready():
    with patch(f"{MODULE}.AsyncTwelveLabs") as cls, patch(f"{MODULE}.asyncio.sleep", new=AsyncMock()) as sleep:
        client = cls.return_value
        client.embed.tasks.create = AsyncMock(return_value=MagicMock(id="task-1"))
        client.embed.tasks.status = AsyncMock(side_effect=[MagicMock(status="processing"), MagicMock(status="ready")])
        retrieved = MagicMock()
        retrieved.video_embedding.segments = [_segment([0.7, 0.8], scope="video")]
        client.embed.tasks.retrieve = AsyncMock(return_value=retrieved)
        out = await embed_media_async("https://ex.com/v.mp4", "video", "marengo3.0", "tlk")
    client.embed.tasks.create.assert_awaited_once_with(
        model_name="marengo3.0", video_url="https://ex.com/v.mp4", video_embedding_scope=["video"]
    )
    assert client.embed.tasks.status.await_count == 2
    client.embed.tasks.retrieve.assert_awaited_once_with("task-1")
    sleep.assert_awaited_once()
    assert out == [0.7, 0.8]


@pytest.mark.asyncio
async def test_embed_media_async_video_failed_raises():
    with patch(f"{MODULE}.AsyncTwelveLabs") as cls, patch(f"{MODULE}.asyncio.sleep", new=AsyncMock()):
        client = cls.return_value
        client.embed.tasks.create = AsyncMock(return_value=MagicMock(id="task-1"))
        client.embed.tasks.status = AsyncMock(return_value=MagicMock(status="failed"))
        with pytest.raises(RuntimeError):
            await embed_media_async("https://ex.com/v.mp4", "video", "marengo3.0", "tlk")
