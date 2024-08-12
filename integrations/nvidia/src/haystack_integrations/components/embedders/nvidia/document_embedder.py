import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack_integrations.utils.nvidia import NimBackend, is_hosted, url_validation
from tqdm import tqdm

from .truncate import EmbeddingTruncateMode

_DEFAULT_API_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia"


@component
class NvidiaDocumentEmbedder:
    """
    A component for embedding documents using embedding models provided by
    [NVIDIA NIMs](https://ai.nvidia.com).

    Usage example:
    ```python
    from haystack_integrations.components.embedders.nvidia import NvidiaDocumentEmbedder

    doc = Document(content="I love pizza!")

    text_embedder = NvidiaDocumentEmbedder(model="NV-Embed-QA", api_url="https://ai.api.nvidia.com/v1/retrieval/nvidia")
    text_embedder.warm_up()

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)
    ```
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[Secret] = Secret.from_env_var("NVIDIA_API_KEY"),
        api_url: str = _DEFAULT_API_URL,
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
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
        :param batch_size:
            Number of Documents to encode at once.
            Cannot be greater than 50.
        :param progress_bar:
            Whether to show a progress bar or not.
        :param meta_fields_to_embed:
            List of meta fields that should be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
        :param truncate:
            Specifies how inputs longer that the maximum token length should be truncated.
            If None the behavior is model-dependent, see the official documentation for more information.
        """

        self.api_key = api_key
        self.model = model
        self.api_url = url_validation(api_url, _DEFAULT_API_URL, ["v1/embeddings"])
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

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

        model_kwargs = {"input_type": "passage"}
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
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            truncate=str(self.truncate) if self.truncate is not None else None,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NvidiaDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]
            text_to_embed = (
                self.prefix + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self.suffix
            )
            texts_to_embed.append(text_to_embed)

        return texts_to_embed

    def _embed_batch(self, texts_to_embed: List[str], batch_size: int) -> Tuple[List[List[float]], Dict[str, Any]]:
        all_embeddings: List[List[float]] = []
        usage_prompt_tokens = 0
        usage_total_tokens = 0

        assert self.backend is not None

        for i in tqdm(
            range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + batch_size]

            sorted_embeddings, meta = self.backend.embed(batch)
            all_embeddings.extend(sorted_embeddings)

            usage_prompt_tokens += meta.get("usage", {}).get("prompt_tokens", 0)
            usage_total_tokens += meta.get("usage", {}).get("total_tokens", 0)

        return all_embeddings, {"usage": {"prompt_tokens": usage_prompt_tokens, "total_tokens": usage_total_tokens}}

    @component.output_types(documents=List[Document], meta=Dict[str, Any])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.

        The embedding of each Document is stored in the `embedding` field of the Document.

        :param documents:
            A list of Documents to embed.
        :returns:
            A dictionary with the following keys and values:
            - `documents` - List of processed Documents with embeddings.
            - `meta` - Metadata on usage statistics, etc.
        :raises RuntimeError:
            If the component was not initialized.
        :raises TypeError:
            If the input is not a string.
        """
        if not self._initialized:
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)
        elif not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            msg = (
                "NvidiaDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the NvidiaTextEmbedder."
            )
            raise TypeError(msg)

        texts_to_embed = self._prepare_texts_to_embed(documents)
        embeddings, metadata = self._embed_batch(texts_to_embed, self.batch_size)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents, "meta": metadata}
