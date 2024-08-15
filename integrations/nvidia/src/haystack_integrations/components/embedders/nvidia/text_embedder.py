import warnings
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack_integrations.utils.nvidia import NimBackend, is_hosted, url_validation

from .truncate import EmbeddingTruncateMode

_DEFAULT_API_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia"


@component
class NvidiaTextEmbedder:
    """
    A component for embedding strings using embedding models provided by
    [NVIDIA NIMs](https://ai.nvidia.com).

    For models that differentiate between query and document inputs,
    this component embeds the input string as a query.

    Usage example:
    ```python
    from haystack_integrations.components.embedders.nvidia import NvidiaTextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = NvidiaTextEmbedder(model="NV-Embed-QA", api_url="https://ai.api.nvidia.com/v1/retrieval/nvidia")
    text_embedder.warm_up()

    print(text_embedder.run(text_to_embed))
    ```
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
        api_url: str = _DEFAULT_API_URL,
        prefix: str = "",
        suffix: str = "",
        truncate: Optional[Union[EmbeddingTruncateMode, str]] = None,
    ):
        """
        Create a NvidiaTextEmbedder component.

        :param model:
            Embedding model to use.
            If no specific model along with locally hosted API URL is provided,
            the system defaults to the available model found using /models API.
        :param api_key:
            API key for the NVIDIA NIM.
        :param api_url:
            Custom API URL for the NVIDIA NIM.
            Format for API URL is http://host:port
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param truncate:
            Specifies how inputs longer that the maximum token length should be truncated.
            If None the behavior is model-dependent, see the official documentation for more information.
        """

        self.api_key = api_key
        self.model = model
        self.api_url = url_validation(api_url, _DEFAULT_API_URL, ["v1/embeddings"])
        self.prefix = prefix
        self.suffix = suffix

        if isinstance(truncate, str):
            truncate = EmbeddingTruncateMode.from_str(truncate)
        self.truncate = truncate

        self.backend: Optional[Any] = None
        self._initialized = False

        if is_hosted(api_url) and not self.model:  # manually set default model
            self.model = "NV-Embed-QA"

    def default_model(self):
        """Set default model in local NIM mode."""
        valid_models = [
            model.id for model in self.backend.models() if not model.base_model or model.base_model == model.id
        ]
        name = next(iter(valid_models), None)
        if name:
            warnings.warn(
                f"Default model is set as: {name}. \n"
                "Set model using model parameter. \n"
                "To get available models use available_models property.",
                UserWarning,
                stacklevel=2,
            )
            self.model = self.backend.model = name
        else:
            error_message = "No locally hosted model was found."
            raise ValueError(error_message)

    def warm_up(self):
        """
        Initializes the component.
        """
        if self._initialized:
            return

        model_kwargs = {"input_type": "query"}
        if self.truncate is not None:
            model_kwargs["truncate"] = str(self.truncate)
        self.backend = NimBackend(
            self.model,
            api_url=self.api_url,
            api_key=self.api_key,
            model_kwargs=model_kwargs,
        )

        self._initialized = True

        if not self.model:
            self.default_model()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict() if self.api_key else None,
            model=self.model,
            api_url=self.api_url,
            prefix=self.prefix,
            suffix=self.suffix,
            truncate=str(self.truncate) if self.truncate is not None else None,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NvidiaTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    @component.output_types(embedding=List[float], meta=Dict[str, Any])
    def run(self, text: str):
        """
        Embed a string.

        :param text:
            The text to embed.
        :returns:
            A dictionary with the following keys and values:
            - `embedding` - Embeddng of the text.
            - `meta` - Metadata on usage statistics, etc.
        :raises RuntimeError:
            If the component was not initialized.
        :raises TypeError:
            If the input is not a string.
        """
        if not self._initialized:
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)
        elif not isinstance(text, str):
            msg = (
                "NvidiaTextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the NvidiaDocumentEmbedder."
            )
            raise TypeError(msg)

        assert self.backend is not None
        text_to_embed = self.prefix + text + self.suffix
        sorted_embeddings, meta = self.backend.embed([text_to_embed])

        return {"embedding": sorted_embeddings[0], "meta": meta}
