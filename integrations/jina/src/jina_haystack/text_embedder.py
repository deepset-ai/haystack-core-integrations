# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, Dict, List, Optional

import requests
from haystack import component, default_to_dict

JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"


@component
class JinaTextEmbedder:
    """
    A component for embedding strings using Jina models.

    Usage example:
    ```python
    from jina_haystack import JinaTextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = JinaTextEmbedder()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    # 'meta': {'model': 'jina-embeddings-v2-base-en',
    #          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
    ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jina-embeddings-v2-base-en",
        prefix: str = "",
        suffix: str = "",
    ):
        """
        Create an JinaTextEmbedder component.

        :param api_key: The Jina API key. It can be explicitly provided or automatically read from the
            environment variable JINA_API_KEY (recommended).
        :param model: The name of the Jina model to use. Check the list of available models on `https://jina.ai/embeddings/`
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        """

        api_key = api_key or os.environ.get("JINA_API_KEY")
        # we check whether api_key is None or an empty string
        if not api_key:
            msg = (
                "JinaTextEmbedder expects an API key. "
                "Set the JINA_API_KEY environment variable (recommended) or pass it explicitly."
            )
            raise ValueError(msg)

        self.model_name = model
        self.prefix = prefix
        self.suffix = suffix
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept-Encoding": "identity",
                "Content-type": "application/json",
            }
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def to_dict(self) -> Dict[str, Any]:
        """
        This method overrides the default serializer in order to avoid leaking the `api_key` value passed
        to the constructor.
        """

        return default_to_dict(self, model=self.model_name, prefix=self.prefix, suffix=self.suffix)

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(self, text: str):
        """Embed a string."""
        if not isinstance(text, str):
            msg = (
                "JinaTextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the JinaDocumentEmbedder."
            )
            raise TypeError(msg)

        text_to_embed = self.prefix + text + self.suffix

        resp = self._session.post(JINA_API_URL, json={"input": [text_to_embed], "model": self.model_name}).json()
        if "data" not in resp:
            raise RuntimeError(resp["detail"])

        metadata = {"model": resp["model"], "usage": dict(resp["usage"].items())}
        embedding = resp["data"][0]["embedding"]

        return {"embedding": embedding, "meta": metadata}
