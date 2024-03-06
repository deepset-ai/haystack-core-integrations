from typing import Any, Dict, List, Optional, Tuple, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack_integrations.utils.nvidia import NvidiaCloudFunctionsClient
from tqdm import tqdm

from ._schema import MAX_INPUTS, EmbeddingsRequest, EmbeddingsResponse, Usage, get_model_nvcf_id
from .models import NvidiaEmbeddingModel


@component
class NvidiaDocumentEmbedder:
    """
    A component for embedding documents using embedding models provided by
    [NVIDIA AI Foundation Endpoints](https://www.nvidia.com/en-us/ai-data-science/foundation-models/).

    Usage example:
    ```python
    from haystack_integrations.components.embedders.nvidia import NvidiaDocumentEmbedder, NvidiaEmbeddingModel

    doc = Document(content="I love pizza!")

    text_embedder = NvidiaDocumentEmbedder(model=NvidiaEmbeddingModel.NVOLVE_40K)
    text_embedder.warm_up()

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)
    ```
    """

    def __init__(
        self,
        model: Union[str, NvidiaEmbeddingModel],
        api_key: Secret = Secret.from_env_var("NVIDIA_API_KEY"),
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
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
        :param batch_size:
            Number of Documents to encode at once.
            Cannot be greater than 50.
        :param progress_bar:
            Whether to show a progress bar or not.
        :param meta_fields_to_embed:
            List of meta fields that should be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
        """

        if isinstance(model, str):
            model = NvidiaEmbeddingModel.from_str(model)

        # Upper-limit for the endpoint.
        if batch_size > MAX_INPUTS:
            msg = f"NVIDIA Cloud Functions currently support a maximum batch size of {MAX_INPUTS}."
            raise ValueError(msg)

        self.api_key = api_key
        self.model = model
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

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

        self.nvcf_id = get_model_nvcf_id(self.model, self.client)
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
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
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
        data["init_parameters"]["model"] = NvidiaEmbeddingModel.from_str(data["init_parameters"]["model"])
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
        usage = Usage(prompt_tokens=0, total_tokens=0)
        assert self.nvcf_id is not None

        for i in tqdm(
            range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + batch_size]

            request = EmbeddingsRequest(input=batch, model="passage").to_dict()
            json_response = self.client.query_function(self.nvcf_id, request)
            response = EmbeddingsResponse.from_dict(json_response)

            # Sort resulting embeddings by index
            assert all(isinstance(r.embedding, list) for r in response.data)
            sorted_embeddings: List[List[float]] = [r.embedding for r in sorted(response.data, key=lambda e: e.index)]  # type: ignore
            all_embeddings.extend(sorted_embeddings)

            usage.prompt_tokens += response.usage.prompt_tokens
            usage.total_tokens += response.usage.total_tokens

        return all_embeddings, {"usage": usage.to_dict()}

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
