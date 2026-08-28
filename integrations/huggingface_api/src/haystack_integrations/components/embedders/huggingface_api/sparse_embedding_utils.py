# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
from haystack.dataclasses import SparseEmbedding
from haystack.utils import Secret


def _build_client_kwargs(
    *, api_base_url: str, timeout: float | None, headers: dict[str, str], token: Secret | None
) -> dict[str, Any]:
    """
    Build the `httpx` client options shared by the sync and async clients.

    `headers` is applied after the token, so an `Authorization` header passed at initialization wins over the one
    derived from `token`. Otherwise, an explicit header would be silently replaced whenever `HF_TOKEN` happens to be
    set in the environment.
    """
    request_headers: dict[str, str] = {}
    if token and (token_value := token.resolve_value()):
        request_headers["Authorization"] = f"Bearer {token_value}"
    request_headers.update(headers)
    return {"base_url": f"{api_base_url.rstrip('/')}/", "timeout": timeout, "headers": request_headers}


def _parse_sparse_embeddings(result: Any, expected_count: int) -> list[SparseEmbedding]:
    if not isinstance(result, list) or len(result) != expected_count:
        msg = f"Expected one sparse embedding per input ({expected_count}), got {result!r}"
        raise ValueError(msg)

    embeddings = []
    for sparse_values in result:
        if not isinstance(sparse_values, list):
            msg = f"Expected each sparse embedding to be a list, got {sparse_values!r}"
            raise ValueError(msg)

        indices = []
        values = []
        for sparse_value in sparse_values:
            if not isinstance(sparse_value, dict) or "index" not in sparse_value or "value" not in sparse_value:
                msg = f"Expected sparse values with 'index' and 'value' fields, got {sparse_value!r}"
                raise ValueError(msg)
            index = sparse_value["index"]
            value = sparse_value["value"]
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not isinstance(value, int | float)
                or isinstance(value, bool)
            ):
                msg = f"Invalid sparse value returned by TEI: {sparse_value!r}"
                raise ValueError(msg)
            indices.append(index)
            values.append(float(value))

        embeddings.append(SparseEmbedding(indices=indices, values=values))
    return embeddings


def _embed_sparse(*, client: httpx.Client, inputs: str | list[str]) -> list[SparseEmbedding]:
    response = client.post("embed_sparse", json={"inputs": inputs})
    response.raise_for_status()
    return _parse_sparse_embeddings(response.json(), 1 if isinstance(inputs, str) else len(inputs))


async def _embed_sparse_async(*, client: httpx.AsyncClient, inputs: str | list[str]) -> list[SparseEmbedding]:
    response = await client.post("embed_sparse", json={"inputs": inputs})
    response.raise_for_status()
    return _parse_sparse_embeddings(response.json(), 1 if isinstance(inputs, str) else len(inputs))
