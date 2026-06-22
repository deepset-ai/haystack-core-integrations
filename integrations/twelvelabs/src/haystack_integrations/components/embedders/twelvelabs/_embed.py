# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import httpx

API_BASE = "https://api.twelvelabs.io/v1.3"


def _extract_vector(payload: dict[str, Any]) -> list[float]:
    for key in ("text_embedding", "image_embedding", "audio_embedding", "video_embedding"):
        embedding = payload.get(key)
        if isinstance(embedding, dict):
            segments = embedding.get("segments") or []
            if segments and isinstance(segments[0], dict):
                vector = segments[0].get("float")
                if isinstance(vector, list):
                    return [float(x) for x in vector]
    msg = f"TwelveLabs embed returned no vector: {str(payload)[:200]}"
    raise RuntimeError(msg)


def _fields(model: str, text: str) -> dict[str, Any]:
    # `(None, value)` tuples send plain fields as multipart/form-data, which the
    # TwelveLabs embed endpoint requires.
    return {"model_name": (None, model), "text": (None, text)}


def _check(response: httpx.Response) -> None:
    if response.status_code >= httpx.codes.BAD_REQUEST:
        msg = f"TwelveLabs embed failed: HTTP {response.status_code} {response.text[:300]}"
        raise RuntimeError(msg)


def embed_text(text: str, model: str, api_key: str) -> list[float]:
    """Embed a single string with Marengo and return its float vector."""
    with httpx.Client() as client:
        response = client.post(
            f"{API_BASE}/embed",
            files=_fields(model, text),
            headers={"x-api-key": api_key},
            timeout=120.0,
        )
    _check(response)
    return _extract_vector(response.json())


async def embed_text_async(text: str, model: str, api_key: str) -> list[float]:
    """Asynchronously embed a single string with Marengo."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE}/embed",
            files=_fields(model, text),
            headers={"x-api-key": api_key},
            timeout=120.0,
        )
    _check(response)
    return _extract_vector(response.json())
