from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.hf import HFModelType, check_valid_model, deserialize_hf_model_kwargs, serialize_hf_model_kwargs
from haystack_integrations.components.embedders.optimum_backend import OptimumEmbeddingBackend
from haystack_integrations.components.embedders.pooling import HFPoolingMode, PoolingMode


@component
class OptimumDocumentEmbedder:
    """
    A component for computing Document embeddings using models loaded with the HuggingFace Optimum library.
    This component is designed to seamlessly inference models using the high speed ONNX runtime.

    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack.dataclasses import Document
    from haystack_integrations.components.embedders import OptimumDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = OptimumDocumentEmbedder(model="sentence-transformers/all-mpnet-base-v2")
    document_embedder.warm_up()

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```

    Key Features and Compatibility:
        - **Primary Compatibility**: Designed to work seamlessly with any embedding model present on the Hugging Face
        Hub.
        - **Conversion to ONNX**: The models are converted to ONNX using the HuggingFace Optimum library. This is
        performed in real-time, during the warm-up step.
        - **Accelerated Inference on GPU**: Supports using different execution providers such as CUDA and TensorRT, to
        accelerate ONNX Runtime inference on GPUs.
        Simply pass the execution provider as the onnx_execution_provider parameter. Additonal parameters can be passed
        to the model using the model_kwargs parameter.
        For more details refer to the HuggingFace documentation:
        https://huggingface.co/docs/optimum/onnxruntime/usage_guides/gpu.
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),  # noqa: B008
        prefix: str = "",
        suffix: str = "",
        normalize_embeddings: bool = True,
        onnx_execution_provider: str = "CPUExecutionProvider",
        pooling_mode: Optional[Union[str, PoolingMode]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):
        """
        Create a OptimumDocumentEmbedder component.

        :param model: A string representing the model id on HF Hub.
        :param token: The HuggingFace token to use as HTTP bearer authorization.
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param normalize_embeddings: Whether to normalize the embeddings to unit length.
        :param onnx_execution_provider: The execution provider to use for ONNX models. See
            https://onnxruntime.ai/docs/execution-providers/ for possible providers.

            Note: Using the TensorRT execution provider
            TensorRT requires to build its inference engine ahead of inference, which takes some time due to the model
            optimization and nodes fusion. To avoid rebuilding the engine every time the model is loaded, ONNX Runtime
            provides a pair of options to save the engine: `trt_engine_cache_enable` and `trt_engine_cache_path`. We
            recommend setting these two provider options using the model_kwargs parameter, when using the TensorRT
            execution provider. The usage is as follows:
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
        :param pooling_mode: The pooling mode to use. When None, pooling mode will be inferred from the model config.
            The supported pooling modes are:
            - "cls": Perform CLS Pooling on the output of the embedding model. Uses the first token (CLS token) as text
                representations.
            - "max": Perform Max Pooling on the output of the embedding model. Uses max in each dimension over all
                the tokens.
            - "mean": Perform Mean Pooling on the output of the embedding model.
            - "mean_sqrt_len": Perform mean-pooling on the output of the embedding model, but divide by sqrt
                (input_length).
            - "weighted_mean": Perform Weighted (position) Mean Pooling on the output of the embedding model. See
                https://arxiv.org/abs/2202.08904.
            - "last_token": Perform Last Token Pooling on the output of the embedding model. See
                https://arxiv.org/abs/2202.08904 & https://arxiv.org/abs/2201.10005.
        :param model_kwargs: Dictionary containing additional keyword arguments to pass to the model.
            In case of duplication, these kwargs override `model`, `onnx_execution_provider`, and `token` initialization
            parameters.
        :param batch_size: Number of Documents to encode at once.
        :param progress_bar: Whether to show a progress bar or not. Can be helpful to disable in production deployments
            to keep the logs clean.
        :param meta_fields_to_embed: List of meta fields that should be embedded along with the Document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the Document text.
        """
        check_valid_model(model, HFModelType.EMBEDDING, token)
        self.model = model

        self.token = token
        token = token.resolve_value() if token else None

        if isinstance(pooling_mode, str):
            self.pooling_mode = PoolingMode.from_str(pooling_mode)
        # Infer pooling mode from model config if not provided,
        if pooling_mode is None:
            self.pooling_mode = HFPoolingMode.get_pooling_mode(model, token)
        # Raise error if pooling mode is not found in model config and not specified by user
        if self.pooling_mode is None:
            modes = {e.value: e for e in PoolingMode}
            msg = (
                f"Pooling mode not found in model config and not specified by user."
                f" Supported modes are: {list(modes.keys())}"
            )
            raise ValueError(msg)

        self.prefix = prefix
        self.suffix = suffix
        self.normalize_embeddings = normalize_embeddings
        self.onnx_execution_provider = onnx_execution_provider
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

        model_kwargs = model_kwargs or {}

        # Check if the model_kwargs contain the parameters, otherwise, populate them with values from init parameters
        model_kwargs.setdefault("model_id", model)
        model_kwargs.setdefault("provider", onnx_execution_provider)
        model_kwargs.setdefault("use_auth_token", token)

        self.model_kwargs = model_kwargs
        self.embedding_backend = None

    def warm_up(self):
        """
        Load the embedding backend.
        """
        if self.embedding_backend is None:
            self.embedding_backend = OptimumEmbeddingBackend(
                model=self.model, token=self.token, model_kwargs=self.model_kwargs
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        serialization_dict = default_to_dict(
            self,
            model=self.model,
            prefix=self.prefix,
            suffix=self.suffix,
            normalize_embeddings=self.normalize_embeddings,
            onnx_execution_provider=self.onnx_execution_provider,
            pooling_mode=self.pooling_mode.value,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            model_kwargs=self.model_kwargs,
            token=self.token.to_dict() if self.token else None,
        )

        model_kwargs = serialization_dict["init_parameters"]["model_kwargs"]
        model_kwargs.pop("token", None)

        serialize_hf_model_kwargs(model_kwargs)
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimumDocumentEmbedder":
        """
        Deserialize this component from a dictionary.
        """
        data["init_parameters"]["pooling_mode"] = PoolingMode.from_str(data["init_parameters"]["pooling_mode"])
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        deserialize_hf_model_kwargs(data["init_parameters"]["model_kwargs"])
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
                self.prefix + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self.suffix
            )

            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Embed a list of Documents.
        The embedding of each Document is stored in the `embedding` field of the Document.

        :param documents: A list of Documents to embed.
        :return: A dictionary containing the updated Documents with their embeddings.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            msg = (
                "OptimumDocumentEmbedder expects a list of Documents as input."
                " In case you want to embed a string, please use the OptimumTextEmbedder."
            )
            raise TypeError(msg)

        if self.embedding_backend is None:
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)

        # Return empty list if no documents
        if not documents:
            return {"documents": []}

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings = self.embedding_backend.embed(
            texts_to_embed=texts_to_embed,
            normalize_embeddings=self.normalize_embeddings,
            pooling_mode=self.pooling_mode,
            progress_bar=self.progress_bar,
            batch_size=self.batch_size,
        )

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents}
