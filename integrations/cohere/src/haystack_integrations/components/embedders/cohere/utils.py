# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from cohere import AsyncClient, Client


async def get_async_response(cohere_async_client: AsyncClient, texts: List[str], model_name, input_type, truncate):
    """Embeds a list of texts asynchronously using the Cohere API.

    :param cohere_async_client: the Cohere `AsyncClient`
    :param texts: the texts to embed
    :param model_name: the name of the model to use
    :param input_type: one of "classification", "clustering", "search_document", "search_query".
        The type of input text provided to embed.
    :param truncate: one of "NONE", "START", "END". How the API handles text longer than the maximum token length.

    :returns: A tuple of the embeddings and metadata.

    :raises ValueError: If an error occurs while querying the Cohere API.
    """
    all_embeddings: List[List[float]] = []
    metadata: Dict[str, Any] = {}

    response = await cohere_async_client.embed(texts=texts, model=model_name, input_type=input_type, truncate=truncate)
    if response.meta is not None:
        metadata = response.meta
    for emb in response.embeddings:
        all_embeddings.append(emb)

    return all_embeddings, metadata


def get_response(
    cohere_client: Client, texts: List[str], model_name, input_type, truncate, batch_size=32, progress_bar=False
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Embeds a list of texts using the Cohere API.

    :param cohere_client: the Cohere `Client`
    :param texts: the texts to embed
    :param model_name: the name of the model to use
    :param input_type: one of "classification", "clustering", "search_document", "search_query".
        The type of input text provided to embed.
    :param truncate: one of "NONE", "START", "END". How the API handles text longer than the maximum token length.
    :param batch_size: the batch size to use
    :param progress_bar: if `True`, show a progress bar

    :returns: A tuple of the embeddings and metadata.

    :raises ValueError: If an error occurs while querying the Cohere API.
    """

    all_embeddings: List[List[float]] = []
    metadata: Dict[str, Any] = {}

    for i in tqdm(
        range(0, len(texts), batch_size),
        disable=not progress_bar,
        desc="Calculating embeddings",
    ):
        batch = texts[i : i + batch_size]
        response = cohere_client.embed(texts=batch, model=model_name, input_type=input_type, truncate=truncate)
        for emb in response.embeddings:
            all_embeddings.append(emb)
        if response.meta is not None:
            metadata = response.meta

    return all_embeddings, metadata
