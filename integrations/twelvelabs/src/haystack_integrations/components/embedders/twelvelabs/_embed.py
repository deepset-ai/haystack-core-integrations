# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from twelvelabs import AsyncTwelveLabs, TwelveLabs
from twelvelabs.types.embedding_response import EmbeddingResponse


def _extract_vector(response: EmbeddingResponse) -> list[float]:
    """Pull the float vector out of a Marengo text-embedding response."""
    text_embedding = response.text_embedding
    segments = text_embedding.segments if text_embedding else None
    if segments:
        vector = segments[0].float_
        if vector:
            return [float(x) for x in vector]
    msg = f"TwelveLabs embed returned no vector for model {response.model_name}"
    raise RuntimeError(msg)


def embed_text(text: str, model: str, api_key: str) -> list[float]:
    """Embed a single string with Marengo and return its float vector."""
    client = TwelveLabs(api_key=api_key)
    response = client.embed.create(model_name=model, text=text)
    return _extract_vector(response)


async def embed_text_async(text: str, model: str, api_key: str) -> list[float]:
    """Asynchronously embed a single string with Marengo."""
    client = AsyncTwelveLabs(api_key=api_key)
    response = await client.embed.create(model_name=model, text=text)
    return _extract_vector(response)
