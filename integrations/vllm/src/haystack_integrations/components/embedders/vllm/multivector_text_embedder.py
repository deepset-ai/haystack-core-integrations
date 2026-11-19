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
    A component for computing text multivector embeddings using models served with [vLLM](https://docs.vllm.ai/).

    It uses the `/pooling` endpoint of a running vLLM server with the `token_embed` task.

    ### Starting the vLLM server

    Before using this component, start a vLLM server with a model that supports token embeddings:

    ```bash
    vllm serve answerdotai/answerai-colbert-small-v1 --runner pooling --pooler-config '{"task":"token_embed"}'
    ```

    For details on server options, see the [vLLM CLI docs](https://docs.vllm.ai/en/stable/cli/serve/).

    ### Usage example

    ```python
    from haystack_integrations.components.embedders.vllm import VLLMMultivectorTextEmbedder

    text_embedder = VLLMMultivectorTextEmbedder(model="answerdotai/answerai-colbert-small-v1")
    print(text_embedder.run("I love pizza!"))
    ```

    ### Usage example with vLLM-specific parameters

    Pass vLLM-specific parameters via the `extra_parameters` dictionary. They are merged into the
    request body sent to the `/pooling` endpoint.

    ```python
    text_embedder = VLLMMultivectorTextEmbedder(
        model="answerdotai/answerai-colbert-small-v1",
        extra_parameters={"truncate_prompt_tokens": 256},
    )
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
        Creates an instance of VLLMMultivectorTextEmbedder.

        :param model: The name of the token-embedding model served by vLLM. See the
            [vLLM token embedding docs](https://docs.vllm.ai/en/stable/models/pooling_models/token_embed/)
            for information on supported models.
        :param api_key: The vLLM API key. Defaults to the `VLLM_API_KEY` environment variable.
            Only required if the server was started with `--api-key`.
        :param api_base_url: Base URL of the vLLM server. Do not append `/v1`; the native `/pooling` endpoint is used.
        :param prefix: String added before the text.
        :param suffix: String added after the text.
        :param timeout: Timeout in seconds for vLLM client calls. When set, this takes precedence over
            `http_client_kwargs["timeout"]`. If `None`, the timeout from `http_client_kwargs` is used;
            if neither is set, the HTTPX default applies.
        :param http_client_kwargs: A dictionary of keyword arguments to configure a custom `httpx.Client` or
            `httpx.AsyncClient`. For more information, see the
            [HTTPX documentation](https://www.python-httpx.org/api/#client).
        :param extra_parameters: Additional parameters merged into the request body sent to the vLLM
            `/pooling` endpoint, such as `truncate_prompt_tokens`. `model`, `input`, and `task` cannot be
            overridden. See the
            [vLLM Pooling API docs](https://docs.vllm.ai/en/stable/models/pooling_models/#pooling-api).
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
        :returns: A dictionary with:
            - `multivector_embedding`: The token embeddings as a list of lists of floats, with shape
              `[number_of_tokens, embedding_dimension]`.
        :raises TypeError: `text` is not a string.
        :raises httpx.HTTPError: A request fails.
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
        :returns: A dictionary with:
            - `multivector_embedding`: The token embeddings as a list of lists of floats, with shape
              `[number_of_tokens, embedding_dimension]`.
        :raises TypeError: `text` is not a string.
        :raises httpx.HTTPError: A request fails.
        """
        kwargs = self._prepare_input(text)
        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101
        response = await self._async_client.post(f"{self.api_base_url}/pooling", json=kwargs)
        response.raise_for_status()
        return {"multivector_embedding": response.json()["data"][0]["data"]}
