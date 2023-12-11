# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Tuple

from cohere import AsyncClient, Client, CohereError
from tqdm import tqdm

API_BASE_URL = "https://api.cohere.ai/v1/embed"


async def get_async_response(cohere_async_client: AsyncClient, texts: List[str], model_name, input_type, truncate):
    all_embeddings: List[List[float]] = []
    metadata: Dict[str, Any] = {}
    try:
        response = await cohere_async_client.embed(
            texts=texts, model=model_name, input_type=input_type, truncate=truncate
        )
        if response.meta is not None:
            metadata = response.meta
        for emb in response.embeddings:
            all_embeddings.append(emb)

        return all_embeddings, metadata

    except CohereError as error_response:
        msg = error_response.message
        raise ValueError(msg) from error_response


def get_response(
    cohere_client: Client, texts: List[str], model_name, input_type, truncate, batch_size=32, progress_bar=False
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    We support batching with the sync client.
    """
    all_embeddings: List[List[float]] = []
    metadata: Dict[str, Any] = {}

    try:
        for i in tqdm(
            range(0, len(texts), batch_size),
            disable=not progress_bar,
            desc="Calculating embeddings",
        ):
            batch = texts[i : i + batch_size]
            response = cohere_client.embed(batch, model=model_name, input_type=input_type, truncate=truncate)
            for emb in response.embeddings:
                all_embeddings.append(emb)
            embeddings = [list(map(float, emb)) for emb in response.embeddings]
            all_embeddings.extend(embeddings)
            if response.meta is not None:
                metadata = response.meta

        return all_embeddings, metadata

    except CohereError as error_response:
        msg = error_response.message
        raise ValueError(msg) from error_response
