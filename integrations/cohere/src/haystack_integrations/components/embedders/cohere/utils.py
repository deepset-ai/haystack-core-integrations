# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from cohere import AsyncClientV2, ClientV2

from .embedding_types import EmbeddingTypes


async def get_async_response(
    cohere_async_client: AsyncClientV2,
    texts: List[str],
    model_name: str,
    input_type: str,
    truncate: str,
    embedding_type: Optional[EmbeddingTypes] = None,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Embeds a list of texts asynchronously using the Cohere API.

    :param cohere_async_client: the Cohere `AsyncClient`
    :param texts: the texts to embed
    :param model_name: the name of the model to use
    :param input_type: one of "classification", "clustering", "search_document", "search_query".
        The type of input text provided to embed.
    :param truncate: one of "NONE", "START", "END". How the API handles text longer than the maximum token length.
    :param embedding_type: the type of embeddings to return. Defaults to float embeddings.

    :returns: A tuple of the embeddings and metadata.

    :raises ValueError: If an error occurs while querying the Cohere API.
    """
    all_embeddings: List[List[float]] = []
    metadata: Dict[str, Any] = {}

    embedding_type = embedding_type or EmbeddingTypes.FLOAT
    response = await cohere_async_client.embed(
        texts=texts,
        model=model_name,
        input_type=input_type,
        truncate=truncate,
        embedding_types=[embedding_type.value],
    )
    if response.meta is not None:
        metadata = response.meta.model_dump()
    for emb_tuple in response.embeddings:
        # emb_tuple[0] is a str denoting the embedding type (e.g. "float", "int8", etc.)
        if emb_tuple[1] is not None:
            # ok we have embeddings for this type, let's take all
            # the embeddings (a list of embeddings) and break the loop
            all_embeddings.extend(emb_tuple[1])
            break

    return all_embeddings, metadata


def get_response(
    cohere_client: ClientV2,
    texts: List[str],
    model_name: str,
    input_type: str,
    truncate: str,
    batch_size: int = 32,
    progress_bar: bool = False,
    embedding_type: Optional[EmbeddingTypes] = None,
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
    :param embedding_type: the type of embeddings to return. Defaults to float embeddings.

    :returns: A tuple of the embeddings and metadata.

    :raises ValueError: If an error occurs while querying the Cohere API.
    """

    all_embeddings: List[List[float]] = []
    metadata: Dict[str, Any] = {}
    embedding_type = embedding_type or EmbeddingTypes.FLOAT

    for i in tqdm(
        range(0, len(texts), batch_size),
        disable=not progress_bar,
        desc="Calculating embeddings",
    ):
        batch = texts[i : i + batch_size]
        response = cohere_client.embed(
            texts=batch,
            model=model_name,
            input_type=input_type,
            truncate=truncate,
            embedding_types=[embedding_type.value],
        )
        ## response.embeddings always returns 5 tuples, one tuple per embedding type
        ## let's take first non None tuple as that's the one we want
        for emb_tuple in response.embeddings:
            # emb_tuple[0] is a str denoting the embedding type (e.g. "float", "int8", etc.)
            if emb_tuple[1] is not None:
                # ok we have embeddings for this type, let's take all the embeddings (a list of embeddings)
                all_embeddings.extend(emb_tuple[1])
        if response.meta is not None:
            metadata = response.meta.model_dump()

    return all_embeddings, metadata
