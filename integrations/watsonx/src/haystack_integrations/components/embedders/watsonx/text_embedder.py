# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings


@component
class WatsonxTextEmbedder:
    """
    Embeds strings using IBM watsonx.ai foundation models.

    You can use it to embed user query and send it to an embedding Retriever.

    ### Usage example

    ```python
    from haystack_integrations.components.embedders.watsonx.text_embedder import WatsonxTextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = WatsonxTextEmbedder(
        model="ibm/slate-30m-english-rtrvr",
        api_key=Secret.from_env_var("WATSONX_API_KEY"),
        api_base_url="https://us-south.ml.cloud.ibm.com",
        project_id=Secret.from_env_var("WATSONX_PROJECT_ID"),
    )

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    #  'meta': {'model': 'ibm/slate-30m-english-rtrvr',
    #           'truncated_input_tokens': 3}}
    ```
    """

    def __init__(
        self,
        *,
        model: str = "ibm/slate-30m-english-rtrvr",
        api_key: Secret = Secret.from_env_var("WATSONX_API_KEY"),  # noqa: B008
        api_base_url: str = "https://us-south.ml.cloud.ibm.com",
        project_id: Secret = Secret.from_env_var("WATSONX_PROJECT_ID"),  # noqa: B008
        truncate_input_tokens: int | None = None,
        prefix: str = "",
        suffix: str = "",
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        """
        Creates an WatsonxTextEmbedder component.

        :param model:
            The name of the IBM watsonx model to use for calculating embeddings.
            Default is "ibm/slate-30m-english-rtrvr".
        :param api_key:
            The WATSONX API key. Can be set via environment variable WATSONX_API_KEY.
        :param api_base_url:
            The WATSONX URL for the watsonx.ai service.
            Default is "https://us-south.ml.cloud.ibm.com".
        :param project_id:
            The ID of the Watson Studio project.
            Can be set via environment variable WATSONX_PROJECT_ID.
        :param truncate_input_tokens:
            Maximum number of tokens to use from the input text.
            If set to `None` (or not provided), the full input text is used, up to the model's maximum token limit.
        :param prefix:
            A string to add at the beginning of each text to embed.
        :param suffix:
            A string to add at the end of each text to embed.
        :param timeout:
            Timeout for API requests in seconds.
        :param max_retries:
            Maximum number of retries for API requests.
        """

        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.project_id = project_id
        self.truncate_input_tokens = truncate_input_tokens
        self.prefix = prefix
        self.suffix = suffix
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize the embeddings client
        credentials = Credentials(api_key=api_key.resolve_value(), url=api_base_url)

        params = {}
        if truncate_input_tokens is not None:
            params["truncate_input_tokens"] = truncate_input_tokens

        self.embedder = Embeddings(
            model_id=model,
            credentials=credentials,
            project_id=project_id.resolve_value(),
            params=params if params else None,
            max_retries=max_retries,
        )

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model,
            api_key=self.api_key.to_dict(),
            api_base_url=self.api_base_url,
            project_id=self.project_id.to_dict(),
            truncate_input_tokens=self.truncate_input_tokens,
            prefix=self.prefix,
            suffix=self.suffix,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatsonxTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key", "project_id"])
        return default_from_dict(cls, data)

    def _prepare_input(self, text: str) -> str:
        if not isinstance(text, str):
            msg = (
                "WatsonxTextEmbedder expects a string as an input. "
                "In case you want to embed a list of Documents, please use the WatsonxDocumentEmbedder."
            )
            raise TypeError(msg)
        return self.prefix + text + self.suffix

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    def run(self, text: str) -> dict[str, list[float] | dict[str, Any]]:
        """
        Embeds a single string.

        :param text: Text to embed.
        :returns: A dictionary with:
            - 'embedding': The embedding of the input text
            - 'meta': Information about the model usage
        """
        text_to_embed = self._prepare_input(text=text)
        embedding = self.embedder.embed_query(text_to_embed)
        return {
            "embedding": embedding,
            "meta": {"model": self.model, "truncated_input_tokens": self.truncate_input_tokens},
        }
