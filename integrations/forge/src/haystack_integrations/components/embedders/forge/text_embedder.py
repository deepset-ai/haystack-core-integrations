# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, ClassVar

from haystack import component, default_from_dict, default_to_dict
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils.auth import Secret


@component
class ForgeTextEmbedder(OpenAITextEmbedder):
    """
    A component for embedding strings using Forge models.

    Forge exposes an OpenAI-compatible embeddings API, so this component subclasses Haystack's
    built-in `OpenAITextEmbedder` and points it at the Forge API base URL.

    Usage example:
     ```python
    from haystack_integrations.components.embedders.forge import ForgeTextEmbedder

    text_to_embed = "I love pizza!"
    text_embedder = ForgeTextEmbedder()
    print(text_embedder.run(text_to_embed))

    # output:
    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    #  'meta': {'model': 'forge-pro', 'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "forge-turbo",
        "forge-pro",
        "forge-ultra-4k",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]
    """A list of model strings accepted by the Forge embeddings API.
    The `forge-*` models are native Forge models; the `text-embedding-*` strings are accepted as
    OpenAI-compatible aliases. See [Forge](https://voxell.ai) for more information."""

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("FORGE_API_KEY"),
        model: str = "forge-pro",
        api_base_url: str | None = "https://api.voxell.ai/v1",
        dimensions: int | None = None,
        prefix: str = "",
        suffix: str = "",
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates a ForgeTextEmbedder component.

        :param api_key:
            The Forge API key.
        :param model:
            The name of the Forge embedding model to be used.
        :param api_base_url:
            The Forge API base URL.
        :param dimensions:
            The number of dimensions of the resulting output embeddings. Forge models support
            Matryoshka representation learning, so a smaller dimension can be requested. If not set,
            the model's default dimensionality is used.
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param timeout:
            Timeout for Forge client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
            variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact Forge after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """

        super(ForgeTextEmbedder, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            dimensions=dimensions,
            api_base_url=api_base_url,
            organization=None,
            prefix=prefix,
            suffix=suffix,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=http_client_kwargs,
        )
        # We add these since they were only added in Haystack 2.14.0
        self.timeout = timeout
        self.max_retries = max_retries

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
            api_base_url=self.api_base_url,
            dimensions=self.dimensions,
            prefix=self.prefix,
            suffix=self.suffix,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForgeTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)
