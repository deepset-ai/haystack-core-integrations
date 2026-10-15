# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_to_dict
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils.auth import Secret

from haystack_integrations.common.otari.utils import _warn_if_eu_key_on_default_url


@component
class OtariTextEmbedder(OpenAITextEmbedder):
    """
    Embeds strings using embedding models served through an [Otari](https://github.com/mozilla-ai/otari) gateway.

    By default, this component calls the otari.ai gateway. For an account in otari.ai's EU region, whose API keys
    start with `otk_v1_eu_`, set `api_base_url` to `"https://eu.api.otari.ai/api/v1"`. To call a gateway you run
    yourself, set `api_base_url`, for example to `"http://localhost:8000/api/v1"`.

    Name models with a selector the gateway accepts, such as `"openai:text-embedding-3-small"`.

    ### Usage example

    ```python
    from haystack_integrations.components.embedders.otari import OtariTextEmbedder

    # Reads the API key from the OTARI_API_KEY environment variable
    text_embedder = OtariTextEmbedder()
    print(text_embedder.run("I love pizza!"))
    # >> {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    # >>  'meta': {'model': 'text-embedding-3-small', 'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("OTARI_API_KEY"),
        model: str = "openai:text-embedding-3-small",
        dimensions: int | None = None,
        api_base_url: str = "https://api.otari.ai/api/v1",
        prefix: str = "",
        suffix: str = "",
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an OtariTextEmbedder component.

        :param api_key:
            The Otari API key: a key issued by your gateway, or an otari.ai API key.
            Defaults to the `OTARI_API_KEY` environment variable.
        :param model:
            The embedding model selector, for example `"openai:text-embedding-3-small"`.
            The gateway must have the provider configured.
        :param dimensions:
            The number of dimensions of the resulting embeddings. Only some models support this parameter.
        :param api_base_url:
            The OpenAI-compatible API root of the Otari gateway, including the `/api/v1` path.
            Defaults to the otari.ai gateway. For an account in otari.ai's EU region, use
            `"https://eu.api.otari.ai/api/v1"`.
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param timeout:
            Timeout for the API call. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
            variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact Otari after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        super(OtariTextEmbedder, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            api_base_url=api_base_url,
            prefix=prefix,
            suffix=suffix,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=http_client_kwargs,
        )

    def warm_up(self) -> None:
        """
        Initialize the synchronous OpenAI client.

        Logs a warning if the API key belongs to otari.ai's EU region but `api_base_url` is left at its default.
        """
        creates_client = self.client is None
        super(OtariTextEmbedder, self).warm_up()  # noqa: UP008
        if creates_client:
            _warn_if_eu_key_on_default_url(self.api_key, self.api_base_url)

    async def warm_up_async(self) -> None:
        """
        Initialize the asynchronous OpenAI client.

        Logs a warning if the API key belongs to otari.ai's EU region but `api_base_url` is left at its default.
        """
        creates_client = self.async_client is None
        await super(OtariTextEmbedder, self).warm_up_async()  # noqa: UP008
        if creates_client:
            _warn_if_eu_key_on_default_url(self.api_key, self.api_base_url)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            dimensions=self.dimensions,
            api_base_url=self.api_base_url,
            prefix=self.prefix,
            suffix=self.suffix,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )
