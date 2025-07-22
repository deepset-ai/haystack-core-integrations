from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret

from ._backend import _EmbedderBackend, _EmbedderParams
from .optimization import OptimumEmbedderOptimizationConfig
from .pooling import OptimumEmbedderPooling
from .quantization import OptimumEmbedderQuantizationConfig


@component
class OptimumDocumentEmbedder:
    """
    A component for computing `Document` embeddings using models loaded with the
    [HuggingFace Optimum](https://huggingface.co/docs/optimum/index) library,
    leveraging the ONNX runtime for high-speed inference.

    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack.dataclasses import Document
    from haystack_integrations.components.embedders.optimum import OptimumDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = OptimumDocumentEmbedder(model="sentence-transformers/all-mpnet-base-v2")
    document_embedder.warm_up()

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),  # noqa: B008
        prefix: str = "",
        suffix: str = "",
        normalize_embeddings: bool = True,
        onnx_execution_provider: str = "CPUExecutionProvider",
        pooling_mode: Optional[Union[str, OptimumEmbedderPooling]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        working_dir: Optional[str] = None,
        optimizer_settings: Optional[OptimumEmbedderOptimizationConfig] = None,
        quantizer_settings: Optional[OptimumEmbedderQuantizationConfig] = None,
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Create a OptimumDocumentEmbedder component.

        :param model:
            A string representing the model id on HF Hub.
        :param token:
            The HuggingFace token to use as HTTP bearer authorization.
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param normalize_embeddings:
            Whether to normalize the embeddings to unit length.
        :param onnx_execution_provider:
            The [execution provider](https://onnxruntime.ai/docs/execution-providers/)
            to use for ONNX models.

                Note: Using the TensorRT execution provider
                TensorRT requires to build its inference engine ahead of inference,
                which takes some time due to the model optimization and nodes fusion.
                To avoid rebuilding the engine every time the model is loaded, ONNX
                Runtime provides a pair of options to save the engine: `trt_engine_cache_enable`
                and `trt_engine_cache_path`. We recommend setting these two provider
                options using the `model_kwargs` parameter, when using the TensorRT execution provider.
                The usage is as follows:
                ```python
                embedder = OptimumDocumentEmbedder(
                    model="sentence-transformers/all-mpnet-base-v2",
                    onnx_execution_provider="TensorrtExecutionProvider",
                    model_kwargs={
                        "provider_options": {
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": "tmp/trt_cache",
                        }
                    },
                )
                ```
        :param pooling_mode:
            The pooling mode to use. When `None`, pooling mode will be inferred from the model config.
        :param model_kwargs:
            Dictionary containing additional keyword arguments to pass to the model.
            In case of duplication, these kwargs override `model`, `onnx_execution_provider`
            and `token` initialization parameters.
        :param working_dir:
            The directory to use for storing intermediate files
            generated during model optimization/quantization. Required
            for optimization and quantization.
        :param optimizer_settings:
            Configuration for Optimum Embedder Optimization.
            If `None`, no additional optimization is be applied.
        :param quantizer_settings:
            Configuration for Optimum Embedder Quantization.
            If `None`, no quantization is be applied.
        :param batch_size:
            Number of Documents to encode at once.
        :param progress_bar:
            Whether to show a progress bar or not.
        :param meta_fields_to_embed:
            List of meta fields that should be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
        """
        params = _EmbedderParams(
            model=model,
            token=token,
            prefix=prefix,
            suffix=suffix,
            normalize_embeddings=normalize_embeddings,
            onnx_execution_provider=onnx_execution_provider,
            batch_size=batch_size,
            progress_bar=progress_bar,
            pooling_mode=pooling_mode,
            model_kwargs=model_kwargs,
            working_dir=working_dir,
            optimizer_settings=optimizer_settings,
            quantizer_settings=quantizer_settings,
        )
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

        self._backend = _EmbedderBackend(params)
        self._initialized = False

    def warm_up(self):
        """
        Initializes the component.
        """
        if self._initialized:
            return

        self._backend.warm_up()
        self._initialized = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        init_params = self._backend.parameters.serialize()
        init_params["meta_fields_to_embed"] = self.meta_fields_to_embed
        init_params["embedding_separator"] = self.embedding_separator
        return default_to_dict(self, **init_params)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimumDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        _EmbedderParams.deserialize_inplace(data["init_parameters"])
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]

            text_to_embed = (
                self._backend.parameters.prefix
                + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""])
                + self._backend.parameters.suffix
            )

            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Embed a list of Documents.
        The embedding of each Document is stored in the `embedding` field of the Document.

        :param documents:
            A list of Documents to embed.
        :returns:
            The updated Documents with their embeddings.
        :raises RuntimeError:
            If the component was not initialized.
        :raises TypeError:
            If the input is not a list of Documents.
        """
        if not self._initialized:
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = (
                "OptimumDocumentEmbedder expects a list of Documents as input."
                " In case you want to embed a string, please use the OptimumTextEmbedder."
            )
            raise TypeError(msg)

        # Return empty list if no documents
        if not documents:
            return {"documents": []}

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)
        embeddings = self._backend.embed_texts(texts_to_embed)
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents}
