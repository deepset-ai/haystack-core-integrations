# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils.auth import Secret

from haystack_integrations.common.otari.utils import _warn_if_eu_key_on_default_url

_INIT_PARAMETERS: tuple[str, ...] = (
    "api_key",
    "model",
    "dimensions",
    "api_base_url",
    "prefix",
    "suffix",
    "batch_size",
    "progress_bar",
    "meta_fields_to_embed",
    "embedding_separator",
    "timeout",
    "max_retries",
    "http_client_kwargs",
    "raise_on_failure",
)


@component
class OtariDocumentEmbedder(OpenAIDocumentEmbedder):
    """
    Computes Document embeddings using models served through an [Otari](https://github.com/mozilla-ai/otari) gateway.

    The embedding of each Document is stored in the `embedding` field of the Document.

    By default, this component calls the otari.ai gateway. For an account in otari.ai's EU region, whose API keys
    start with `otk_v1_eu_`, set `api_base_url` to `"https://eu.api.otari.ai/api/v1"`. To call a gateway you run
    yourself, set `api_base_url`, for example to `"http://localhost:8000/api/v1"`.

    Name models with a selector the gateway accepts, such as `"openai:text-embedding-3-small"`.

    When the gateway prices a request, `meta["usage"]` contains its cost in US dollars as `cost_usd`, and
    `pricing_source`. If the Documents are sent in more than one batch, the token counts in `meta["usage"]` cover all
    batches, but `cost_usd` and `pricing_source` come from the first batch only.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.otari import OtariDocumentEmbedder

    # Reads the API key from the OTARI_API_KEY environment variable
    document_embedder = OtariDocumentEmbedder()
    result = document_embedder.run([Document(content="I love pizza!")])
    print(result["documents"][0].embedding)
    # >> [0.017020374536514282, -0.023255806416273117, ...]
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
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
        raise_on_failure: bool = False,
    ) -> None:
        """
        Creates an OtariDocumentEmbedder component.

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
        :param batch_size:
            Number of Documents to encode at once.
        :param progress_bar:
            Whether to show a progress bar or not. Can be helpful to disable in production deployments to keep
            the logs clean.
        :param meta_fields_to_embed:
            List of meta fields that should be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
        :param timeout:
            Timeout for the API call. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
            variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact Otari after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        :param raise_on_failure:
            Whether to raise an exception if the embedding request fails. If `False`, the component logs the error
            and continues processing the remaining documents. If `True`, it raises an exception on failure.
        """
        super(OtariDocumentEmbedder, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            api_base_url=api_base_url,
            prefix=prefix,
            suffix=suffix,
            batch_size=batch_size,
            progress_bar=progress_bar,
            meta_fields_to_embed=meta_fields_to_embed,
            embedding_separator=embedding_separator,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=http_client_kwargs,
            raise_on_failure=raise_on_failure,
        )

    def warm_up(self) -> None:
        """
        Initialize the synchronous OpenAI client.

        Logs a warning if the API key belongs to otari.ai's EU region but `api_base_url` is left at its default.
        """
        creates_client = self.client is None
        super(OtariDocumentEmbedder, self).warm_up()  # noqa: UP008
        if creates_client:
            _warn_if_eu_key_on_default_url(self.api_key, self.api_base_url)

    async def warm_up_async(self) -> None:
        """
        Initialize the asynchronous OpenAI client.

        Logs a warning if the API key belongs to otari.ai's EU region but `api_base_url` is left at its default.
        """
        creates_client = self.async_client is None
        await super(OtariDocumentEmbedder, self).warm_up_async()  # noqa: UP008
        if creates_client:
            _warn_if_eu_key_on_default_url(self.api_key, self.api_base_url)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        data = super(OtariDocumentEmbedder, self).to_dict()  # noqa: UP008
        # the parent also serializes the OpenAI-only organization parameter that this component lacks
        data["init_parameters"] = {
            key: value for key, value in data["init_parameters"].items() if key in _INIT_PARAMETERS
        }
        return data
