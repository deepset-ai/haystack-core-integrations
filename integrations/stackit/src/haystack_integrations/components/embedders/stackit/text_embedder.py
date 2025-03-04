# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from haystack import component
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils.auth import Secret


@component
class STACKITTextEmbedder(OpenAITextEmbedder):
    """
    A component for embedding strings using STACKIT as model provider.

    Usage example:
     ```python
    from haystack_integrations.components.embedders.stackit import STACKITTextEmbedder

    text_to_embed = "I love pizza!"
    text_embedder = STACKITTextEmbedder()
    print(text_embedder.run(text_to_embed))
    ```
    """

    def __init__(
        self,
        model: str,
        api_key: Secret = Secret.from_env_var("STACKIT_API_KEY"),
        api_base_url: Optional[str] = "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
        prefix: str = "",
        suffix: str = "",
    ):
        """
        Creates a STACKITTextEmbedder component.

        :param api_key:
            The STACKIT API key.
        :param model:
            The name of the STACKIT embedding model to be used.
        :param api_base_url:
            The STACKIT API Base url.
            For more details, see STACKIT [docs](https://docs.stackit.cloud/stackit/en/basic-concepts-stackit-model-serving-319914567.html).
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        """
        super(STACKITTextEmbedder, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            dimensions=None,
            api_base_url=api_base_url,
            organization=None,
            prefix=prefix,
            suffix=suffix,
        )
