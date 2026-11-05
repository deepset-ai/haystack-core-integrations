# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
import mimetypes
import time
from pathlib import Path, PurePath
from typing import Any
from urllib.parse import urlparse

from twelvelabs import AsyncTwelveLabs, TwelveLabs

# The modalities Marengo can embed into its shared vector space (text is handled by the
# dedicated text embedder).
MODALITIES = ("image", "audio", "video")

# Video embeddings are produced by an asynchronous TwelveLabs task; these bound the poll loop.
_POLL_INTERVAL = 3.0
_TASK_TIMEOUT = 1800.0
# Request a single whole-video embedding (rather than per-clip segments) so this maps to one vector.
_VIDEO_SCOPE = ["video"]

# Extension -> modality, checked before falling back to the standard-library MIME database.
_EXTENSION_MODALITY = {
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".gif": "image",
    ".webp": "image",
    ".bmp": "image",
    ".tif": "image",
    ".tiff": "image",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    ".flac": "audio",
    ".ogg": "audio",
    ".oga": "audio",
    ".aac": "audio",
    ".mp4": "video",
    ".mov": "video",
    ".webm": "video",
    ".mkv": "video",
    ".avi": "video",
    ".m4v": "video",
}


def _is_url(source: str) -> bool:
    """Whether `source` is an http(s) URL (as opposed to a local file path)."""
    return urlparse(source).scheme in ("http", "https")


def _local_path(source: str) -> Path:
    """Resolve a local media path, raising a clear error if it does not exist."""
    path = Path(source).expanduser()
    if not path.is_file():
        msg = f"Media file not found: {source!r}."
        raise FileNotFoundError(msg)
    return path


def detect_modality(source: str) -> str:
    """
    Infer the media modality of a URL or local path.

    :param source: A publicly accessible URL or a local file path.
    :returns: One of `"image"`, `"audio"` or `"video"`.
    :raises ValueError: If the modality cannot be inferred from the extension or MIME type.
    """
    path = urlparse(source).path if _is_url(source) else source
    suffix = PurePath(path).suffix.lower()
    if suffix in _EXTENSION_MODALITY:
        return _EXTENSION_MODALITY[suffix]
    mime, _ = mimetypes.guess_type(path)
    if mime:
        top_level = mime.split("/", 1)[0]
        if top_level in MODALITIES:
            return top_level
    msg = f"Could not infer the modality of {source!r}. Pass an explicit modality, one of {MODALITIES}."
    raise ValueError(msg)


def _extract_vector(result: Any, model: str, modality: str) -> list[float]:
    """Pull the first float vector out of an image/audio embedding result."""
    segments = result.segments if result else None
    if segments:
        vector = segments[0].float_
        if vector:
            return [float(x) for x in vector]
    msg = f"TwelveLabs returned no {modality} embedding for model {model!r}."
    raise RuntimeError(msg)


def _extract_video_vector(result: Any, model: str) -> list[float]:
    """Pull the whole-video float vector out of a retrieved video-embedding task."""
    embedding = getattr(result, "video_embedding", None)
    segments = embedding.segments if embedding else None
    if segments:
        # Prefer the whole-video segment; fall back to the first segment if scope is absent.
        chosen = next((s for s in segments if (getattr(s, "embedding_scope", "") or "") == "video"), segments[0])
        vector = chosen.float_
        if vector:
            return [float(x) for x in vector]
    msg = f"TwelveLabs returned no video embedding for model {model!r}."
    raise RuntimeError(msg)


def embed_media(source: str, modality: str, model: str, api_key: str) -> list[float]:
    """
    Embed a single image, audio or video source with Marengo and return its vector.

    Images and audio are embedded synchronously. Video is embedded through the asynchronous
    TwelveLabs video-embedding task: this submits the task and blocks (polling) until it is ready.

    :param source: A publicly accessible URL or a local file path.
    :param modality: One of `"image"`, `"audio"` or `"video"`.
    :param model: The Marengo model name.
    :param api_key: The TwelveLabs API key.
    :returns: The embedding vector in Marengo's shared multimodal space.
    """
    client = TwelveLabs(api_key=api_key)
    if modality == "video":
        return _embed_video(client, source, model)
    if _is_url(source):
        params: dict[str, Any] = {f"{modality}_url": source}
        response = client.embed.create(model_name=model, **params)
    else:
        with _local_path(source).open("rb") as handle:
            params = {f"{modality}_file": handle}
            response = client.embed.create(model_name=model, **params)
    return _extract_vector(getattr(response, f"{modality}_embedding"), model, modality)


def _embed_video(client: TwelveLabs, source: str, model: str) -> list[float]:
    task_id = _create_video_task(client, source, model)
    _await_video_task(client, task_id)
    result = client.embed.tasks.retrieve(task_id)
    return _extract_video_vector(result, model)


def _create_video_task(client: TwelveLabs, source: str, model: str) -> str:
    if _is_url(source):
        task = client.embed.tasks.create(model_name=model, video_url=source, video_embedding_scope=_VIDEO_SCOPE)
    else:
        with _local_path(source).open("rb") as handle:
            task = client.embed.tasks.create(model_name=model, video_file=handle, video_embedding_scope=_VIDEO_SCOPE)
    task_id = getattr(task, "id", None)
    if not task_id:
        msg = "TwelveLabs did not return an id for the video embedding task."
        raise RuntimeError(msg)
    return task_id


def _await_video_task(client: TwelveLabs, task_id: str) -> None:
    deadline = time.monotonic() + _TASK_TIMEOUT
    while True:
        status = (client.embed.tasks.status(task_id).status or "").lower()
        if status == "ready":
            return
        if status == "failed":
            msg = f"TwelveLabs video embedding task {task_id} failed."
            raise RuntimeError(msg)
        if time.monotonic() >= deadline:
            msg = f"TwelveLabs video embedding task {task_id} did not finish within {_TASK_TIMEOUT:.0f}s."
            raise TimeoutError(msg)
        time.sleep(_POLL_INTERVAL)


async def embed_media_async(source: str, modality: str, model: str, api_key: str) -> list[float]:
    """
    Asynchronously embed a single image, audio or video source with Marengo.

    See :func:`embed_media` for parameter and return details.
    """
    client = AsyncTwelveLabs(api_key=api_key)
    if modality == "video":
        return await _embed_video_async(client, source, model)
    if _is_url(source):
        params: dict[str, Any] = {f"{modality}_url": source}
        response = await client.embed.create(model_name=model, **params)
    else:
        with _local_path(source).open("rb") as handle:
            params = {f"{modality}_file": handle}
            response = await client.embed.create(model_name=model, **params)
    return _extract_vector(getattr(response, f"{modality}_embedding"), model, modality)


async def _embed_video_async(client: AsyncTwelveLabs, source: str, model: str) -> list[float]:
    task_id = await _create_video_task_async(client, source, model)
    await _await_video_task_async(client, task_id)
    result = await client.embed.tasks.retrieve(task_id)
    return _extract_video_vector(result, model)


async def _create_video_task_async(client: AsyncTwelveLabs, source: str, model: str) -> str:
    if _is_url(source):
        task = await client.embed.tasks.create(model_name=model, video_url=source, video_embedding_scope=_VIDEO_SCOPE)
    else:
        with _local_path(source).open("rb") as handle:
            task = await client.embed.tasks.create(
                model_name=model, video_file=handle, video_embedding_scope=_VIDEO_SCOPE
            )
    task_id = getattr(task, "id", None)
    if not task_id:
        msg = "TwelveLabs did not return an id for the video embedding task."
        raise RuntimeError(msg)
    return task_id


async def _await_video_task_async(client: AsyncTwelveLabs, task_id: str) -> None:
    deadline = time.monotonic() + _TASK_TIMEOUT
    while True:
        status_response = await client.embed.tasks.status(task_id)
        status = (status_response.status or "").lower()
        if status == "ready":
            return
        if status == "failed":
            msg = f"TwelveLabs video embedding task {task_id} failed."
            raise RuntimeError(msg)
        if time.monotonic() >= deadline:
            msg = f"TwelveLabs video embedding task {task_id} did not finish within {_TASK_TIMEOUT:.0f}s."
            raise TimeoutError(msg)
        await asyncio.sleep(_POLL_INTERVAL)
