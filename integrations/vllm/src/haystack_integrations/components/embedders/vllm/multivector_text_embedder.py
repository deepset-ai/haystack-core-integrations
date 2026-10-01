# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
from haystack import component
from haystack.utils import Secret
from haystack.utils.http_client import init_http_client


@component
class VLLMMultivectorTextEmbedder:
    """
    Embed text into a token-level multivector using a model served with vLLM.

    The component calls vLLM's native `/pooling` API with the `token_embed` task. Start a server with a model that
    supports token embeddings, for example:

    ```bash
    vllm serve answerdotai/answerai-colbert-small-v1 --runner pooling --pooler-config '{"task":"token_embed"}'
    ```

    The `/pooling` endpoint is not below the OpenAI-compatible `/v1` path, so the default `api_base_url` is
    `http://localhost:8000`. For details, see the
    [vLLM Pooling API documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/#pooling-api).

    ### Usage example

    ```python
    from haystack_integrations.components.embedders.vllm import VLLMMultivectorTextEmbedder

    embedder = VLLMMultivectorTextEmbedder(model="answerdotai/answerai-colbert-small-v1")
    result = embedder.run("A document about Rome")
    print(result["multivector_embedding"])
    ```
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Secret | None = Secret.from_env_var("VLLM_API_KEY", strict=False),
        api_base_url: str = "http://localhost:8000",
        prefix: str = "",
        suffix: str = "",
        timeout: float | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
        extra_parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a VLLMMultivectorTextEmbedder.

        :param model: Name of the token-embedding model served by vLLM.
        :param api_key: vLLM API key. Defaults to the `VLLM_API_KEY` environment variable and is only required when
            the server was started with `--api-key`.
        :param api_base_url: Base URL of the vLLM server. Do not append `/v1`; the native `/pooling` endpoint is used.
        :param prefix: String added before the text.
        :param suffix: String added after the text.
        :param timeout: Request timeout in seconds. When set, this takes precedence over
            `http_client_kwargs["timeout"]`.
        :param http_client_kwargs: Keyword arguments passed to `httpx.Client` and `httpx.AsyncClient`.
        :param extra_parameters: Additional `/pooling` request fields. `model`, `input`, and `task` cannot be
            overridden.
        """
        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url.rstrip("/")
        self.prefix = prefix
        self.suffix = suffix
        self.timeout = timeout
        self.http_client_kwargs = http_client_kwargs
        self.extra_parameters = extra_parameters

        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def _client_kwargs(self) -> dict[str, Any]:
        headers = httpx.Headers((self.http_client_kwargs or {}).get("headers"))
        if self.api_key is not None and (resolved_key := self.api_key.resolve_value()):
            headers["Authorization"] = f"Bearer {resolved_key}"

        client_kwargs = self.http_client_kwargs.copy() if self.http_client_kwargs else {}
        client_kwargs["headers"] = headers
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout
        return client_kwargs

    def warm_up(self) -> None:
        """Create the synchronous HTTP client."""
        if self._client is None:
            client = init_http_client(http_client_kwargs=self._client_kwargs(), async_client=False)
            assert client is not None  # noqa: S101
            self._client = client

    async def warm_up_async(self) -> None:
        """Create the asynchronous HTTP client."""
        if self._async_client is None:
            client = init_http_client(http_client_kwargs=self._client_kwargs(), async_client=True)
            assert client is not None  # noqa: S101
            self._async_client = client

    def close(self) -> None:
        """Close the synchronous HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """Close the asynchronous HTTP client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def _prepare_input(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str):
            msg = (
                "VLLMMultivectorTextEmbedder expects a string as input. To embed Documents, use "
                "VLLMMultivectorDocumentEmbedder."
            )
            raise TypeError(msg)

        kwargs = dict(self.extra_parameters or {})
        kwargs.update({"model": self.model, "input": self.prefix + text + self.suffix, "task": "token_embed"})
        return kwargs

    @component.output_types(multivector_embedding=list[list[float]])
    def run(self, text: str) -> dict[str, list[list[float]]]:
        """
        Embed a string into a token-level multivector.

        :param text: Text to embed.
        :returns: A dictionary containing the `multivector_embedding`.
        :raises TypeError: If `text` is not a string.
        :raises httpx.HTTPError: If the request fails.
        """
        kwargs = self._prepare_input(text)
        self.warm_up()
        assert self._client is not None  # noqa: S101
        response = self._client.post(f"{self.api_base_url}/pooling", json=kwargs)
        response.raise_for_status()
        return {"multivector_embedding": response.json()["data"][0]["data"]}

    @component.output_types(multivector_embedding=list[list[float]])
    async def run_async(self, text: str) -> dict[str, list[list[float]]]:
        """
        Asynchronously embed a string into a token-level multivector.

        :param text: Text to embed.
        :returns: A dictionary containing the `multivector_embedding`.
        :raises TypeError: If `text` is not a string.
        :raises httpx.HTTPError: If the request fails.
        """
        kwargs = self._prepare_input(text)
        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101
        response = await self._async_client.post(f"{self.api_base_url}/pooling", json=kwargs)
        response.raise_for_status()
        return {"multivector_embedding": response.json()["data"][0]["data"]}
