from typing import Any, Dict, List, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack_integrations.utils.nvidia import NvidiaCloudFunctionsClient

from ._schema import EmbeddingsRequest, EmbeddingsResponse, get_model_nvcf_id
from .models import NvidiaEmbeddingModel


@component
class NvidiaTextEmbedder:
    """
    A component for embedding strings using embedding models provided by
    [NVIDIA AI Foundation Endpoints](https://www.nvidia.com/en-us/ai-data-science/foundation-models/).

    For models that differentiate between query and document inputs,
    this component embeds the input string as a query.

    Usage example:
    ```python
    from haystack_integrations.components.embedders.nvidia import NvidiaTextEmbedder, NvidiaEmbeddingModel

    text_to_embed = "I love pizza!"

    text_embedder = NvidiaTextEmbedder(model=NvidiaEmbeddingModel.NVOLVE_40K)
    text_embedder.warm_up()

    print(text_embedder.run(text_to_embed))
    ```
    """

    def __init__(
        self,
        model: Union[str, NvidiaEmbeddingModel],
        api_key: Secret = Secret.from_env_var("NVIDIA_API_KEY"),
        prefix: str = "",
        suffix: str = "",
    ):
        """
        Create a NvidiaTextEmbedder component.

        :param model:
            Embedding model to use.
        :param api_key:
            API key for the NVIDIA AI Foundation Endpoints.
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        """

        if isinstance(model, str):
            model = NvidiaEmbeddingModel.from_str(model)

        self.api_key = api_key
        self.model = model
        self.prefix = prefix
        self.suffix = suffix
        self.client = NvidiaCloudFunctionsClient(
            api_key=api_key,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        self.nvcf_id = None
        self._initialized = False

    def warm_up(self):
        """
        Initializes the component.
        """
        if self._initialized:
            return

        self.nvcf_id = self.client.get_model_nvcf_id(str(self.model))
        self._initialized = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=str(self.model),
            prefix=self.prefix,
            suffix=self.suffix,
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
        data["init_parameters"]["model"] = NvidiaEmbeddingModel.from_str(data["init_parameters"]["model"])
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

        assert self.nvcf_id is not None
        text_to_embed = self.prefix + text + self.suffix
        request = EmbeddingsRequest(input=text_to_embed, model="query").to_dict()
        json_response = self.client.query_function(self.nvcf_id, request)
        response = EmbeddingsResponse.from_dict(json_response)

        return {"embedding": response.data[0].embedding, "meta": {"usage": response.usage.to_dict()}}
