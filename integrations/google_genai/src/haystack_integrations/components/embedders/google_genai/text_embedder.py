# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

from google import genai
from google.genai import types
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


@component
class GoogleGenAITextEmbedder:
    """
    Embeds strings using Google AI models.

    You can use it to embed user query and send it to an embedding Retriever.

    ### Usage example

    ```python
    from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = GoogleGenAITextEmbedder()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    # 'meta': {'model': 'text-embedding-004-v2',
    #          'usage': {'prompt_tokens': 4, 'total_tokens': 4}}}
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var(["GOOGLE_API_KEY", "GEMINI_API_KEY"]),
        model: str = "text-embedding-004",
        prefix: str = "",
        suffix: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Creates an GoogleGenAITextEmbedder component.

        :param api_key:
            The Google API key.
            You can set it with the environment variable `GOOGLE_API_KEY` or `GEMINI_API_KEY`, or pass it via
            this parameter during initialization.
        :param model:
            The name of the model to use for calculating embeddings.
            The default model is `text-embedding-004`.
        :param prefix:
            A string to add at the beginning of each text to embed.
        :param suffix:
            A string to add at the end of each text to embed.
        :param config:
            A dictionary of keyword arguments to configure embedding content configuration `types.EmbedContentConfig`.
            If not specified, it defaults to {"task_type": "SEMANTIC_SIMILARITY"}.
            For more information, see the [Google AI Task types](https://ai.google.dev/gemini-api/docs/embeddings#task-types).
        """

        self._api_key = api_key
        self._model_name = model
        self._prefix = prefix
        self._suffix = suffix
        self._config = config if config is not None else {"task_type": "SEMANTIC_SIMILARITY"}
        self._client = genai.Client(api_key=api_key.resolve_value())

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self._model_name,
            api_key=self._api_key.to_dict(),
            prefix=self._prefix,
            suffix=self._suffix,
            config=self._config,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoogleGenAITextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_input(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str):
            error_message_text = (
                "GoogleGenAITextEmbedder expects a string as an input. "
                "In case you want to embed a list of Documents, please use the GoogleAIDocumentEmbedder."
            )

            raise TypeError(error_message_text)

        text_to_embed = self._prefix + text + self._suffix

        kwargs: Dict[str, Any] = {"model": self._model_name, "contents": text_to_embed}
        if self._config:
            kwargs["config"] = types.EmbedContentConfig(**self._config)

        return kwargs

    def _prepare_output(self, result: types.EmbedContentResponse) -> Dict[str, Any]:
        embedding = result.embeddings[0].values if result.embeddings else []
        return {"embedding": embedding, "meta": {"model": self._model_name}}

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(self, text: str) -> Union[Dict[str, List[float]], Dict[str, Any]]:
        """
        Embeds a single string.

        :param text:
            Text to embed.

        :returns:
            A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
            - `meta`: Information about the usage of the model.
        """
        create_kwargs = self._prepare_input(text=text)
        response = self._client.models.embed_content(**create_kwargs)
        return self._prepare_output(result=response)
