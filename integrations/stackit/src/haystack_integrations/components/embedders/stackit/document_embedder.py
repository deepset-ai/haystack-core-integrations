# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional

from haystack import component
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils.auth import Secret


@component
class STACKITDocumentEmbedder(OpenAIDocumentEmbedder):
    """
    A component for computing Document embeddings using STACKIT as model provider.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.stackit import STACKITDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = STACKITDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        model: str,
        api_key: Secret = Secret.from_env_var("STACKIT_API_KEY"),
        api_base_url: Optional[str] = "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Creates a STACKITDocumentEmbedder component.

        :param api_key:
            The STACKIT API key.
        :param model:
            The name of the model to use.
        :param api_base_url:
            The STACKIT API Base url.
            For more details, see STACKIT [docs](https://docs.stackit.cloud/stackit/en/basic-concepts-stackit-model-serving-319914567.html).
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
        """
        super(STACKITDocumentEmbedder, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            dimensions=None,
            api_base_url=api_base_url,
            organization=None,
            prefix=prefix,
            suffix=suffix,
            batch_size=batch_size,
            progress_bar=progress_bar,
            meta_fields_to_embed=meta_fields_to_embed,
            embedding_separator=embedding_separator,
        )
