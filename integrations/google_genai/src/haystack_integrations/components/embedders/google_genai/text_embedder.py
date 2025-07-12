# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Literal, Optional, Union

from google.genai import types
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_integrations.components.common.google_genai.utils import _get_client

logger = logging.getLogger(__name__)


@component
class GoogleGenAITextEmbedder:
    """
    Embeds strings using Google AI models.

    You can use it to embed user query and send it to an embedding Retriever.

    ### Authentication examples

    **1. Gemini Developer API (API Key Authentication)**
    ```python
    from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder

    # export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
    text_embedder = GoogleGenAITextEmbedder(model="text-embedding-004")

    **2. Vertex AI (Application Default Credentials)**
    ```python
    from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder

    # Using Application Default Credentials (requires gcloud auth setup)
    text_embedder = GoogleGenAITextEmbedder(
        api="vertex",
        vertex_ai_project="my-project",
        vertex_ai_location="us-central1",
        model="text-embedding-004"
    )
    ```

    **3. Vertex AI (API Key Authentication)**
    ```python
    from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder

    # export the environment variable (GOOGLE_API_KEY or GEMINI_API_KEY)
    text_embedder = GoogleGenAITextEmbedder(
        api="vertex",
        model="text-embedding-004"
    )
    ```


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
        api_key: Secret = Secret.from_env_var(["GOOGLE_API_KEY", "GEMINI_API_KEY"], strict=False),
        api: Literal["gemini", "vertex"] = "gemini",
        vertex_ai_project: Optional[str] = None,
        vertex_ai_location: Optional[str] = None,
        model: str = "text-embedding-004",
        prefix: str = "",
        suffix: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Creates an GoogleGenAITextEmbedder component.

        :param api_key: Google API key, defaults to the `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables.
            Not needed if using Vertex AI with Application Default Credentials.
            Go to https://aistudio.google.com/app/apikey for a Gemini API key.
            Go to https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys for a Vertex AI API key.
        :param api: Which API to use. Either "gemini" for the Gemini Developer API or "vertex" for Vertex AI.
        :param vertex_ai_project: Google Cloud project ID for Vertex AI. Required when using Vertex AI with
            Application Default Credentials.
        :param vertex_ai_location: Google Cloud location for Vertex AI (e.g., "us-central1", "europe-west1").
            Required when using Vertex AI with Application Default Credentials.
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
        self._api = api
        self._vertex_ai_project = vertex_ai_project
        self._vertex_ai_location = vertex_ai_location
        self._model_name = model
        self._prefix = prefix
        self._suffix = suffix
        self._config = config if config is not None else {"task_type": "SEMANTIC_SIMILARITY"}
        self._client = _get_client(
            api_key=api_key,
            api=api,
            vertex_ai_project=vertex_ai_project,
            vertex_ai_location=vertex_ai_location,
        )

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
            api=self._api,
            vertex_ai_project=self._vertex_ai_project,
            vertex_ai_location=self._vertex_ai_location,
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

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    async def run_async(self, text: str) -> Union[Dict[str, List[float]], Dict[str, Any]]:
        """
        Asynchronously embed a single string.

        This is the asynchronous version of the `run` method. It has the same parameters and return values
        but can be used with `await` in async code.

        :param text:
            Text to embed.

        :returns:
            A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
            - `meta`: Information about the usage of the model.
        """
        create_kwargs = self._prepare_input(text=text)
        response = await self._client.aio.models.embed_content(**create_kwargs)
        return self._prepare_output(result=response)
