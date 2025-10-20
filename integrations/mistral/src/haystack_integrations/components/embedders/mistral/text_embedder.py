# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

from haystack import component, default_to_dict
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils.auth import Secret


@component
class MistralTextEmbedder(OpenAITextEmbedder):
    """
    A component for embedding strings using Mistral models.

    Usage example:
     ```python
    from haystack_integrations.components.embedders.mistral.text_embedder import MistralTextEmbedder

    text_to_embed = "I love pizza!"
    text_embedder = MistralTextEmbedder()
    print(text_embedder.run(text_to_embed))

    # output:
    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    # 'meta': {'model': 'mistral-embed',
    #          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
        model: str = "mistral-embed",
        api_base_url: Optional[str] = "https://api.mistral.ai/v1",
        prefix: str = "",
        suffix: str = "",
        *,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an MistralTextEmbedder component.

        :param api_key:
            The Mistral API key.
        :param model:
            The name of the Mistral embedding model to be used.
        :param api_base_url:
            The Mistral API Base url.
            For more details, see Mistral [docs](https://docs.mistral.ai/api/).
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param timeout:
            Timeout for Mistral client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
            variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact Mistral after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        super(MistralTextEmbedder, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            dimensions=None,
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

    def to_dict(self) -> Dict[str, Any]:
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
            prefix=self.prefix,
            suffix=self.suffix,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )
