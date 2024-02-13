# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from haystack import component
from haystack.utils.auth import Secret
from haystack.components.embedders import OpenAITextEmbedder


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

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    # 'meta': {'model': 'text-embedding-ada-002-v2',
    #          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
    ```
    """
    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
        model: str = "mistral-embed",
        dimensions: Optional[int] = None,
        api_base_url: Optional[str] = "https://api.mistral.ai/v1",
        organization: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
    ):
        """
        Create an MistralTextEmbedder component.

        :param api_key: The Misttal API key.
        :param model: The name of the Mistral embedding models to be used.
        :param dimensions: Not yet supported with Mistral embedding models
        :param organization: The Organization ID, defaults to `None`. 
        :param api_base_url: The Mistral API Base url, defaults to `https://api.mistral.ai/v1`. For more details, see Mistral [docs](https://docs.mistral.ai/api/).
        :param prefix: Not yet supported with Mistral embedding models
        :param suffix: Not yet supported with Mistral embedding models
        """
        super(MistralTextEmbedder, self).__init__(api_key,
                                                  model,
                                                  dimensions,
                                                  api_base_url,
                                                  organization,
                                                  prefix,
                                                  suffix)