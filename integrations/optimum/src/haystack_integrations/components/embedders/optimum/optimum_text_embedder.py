from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret

from ._backend import _EmbedderBackend, _EmbedderParams
from .pooling import OptimumEmbedderPooling


@component
class OptimumTextEmbedder:
    """
    A component to embed text using models loaded with the HuggingFace Optimum library.
    This component is designed to seamlessly inference models using the high speed ONNX runtime.

    Usage example:
    ```python
    from haystack_integrations.components.optimum.embedders import OptimumTextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = OptimumTextEmbedder(model="sentence-transformers/all-mpnet-base-v2")
    text_embedder.warm_up()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [-0.07804739475250244, 0.1498992145061493,, ...]}
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
        pooling_mode: Optional[Union[str, OptimumEmbedderPooling]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a OptimumTextEmbedder component.

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
            embedder = OptimumTextEmbedder(
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
            Refer to the OptimumEmbedderPooling enum for supported pooling modes.
        :param model_kwargs: Dictionary containing additional keyword arguments to pass to the model.
            In case of duplication, these kwargs override `model`, `onnx_execution_provider`, and `token` initialization
            parameters.
        """
        params = _EmbedderParams(
            model=model,
            token=token,
            prefix=prefix,
            suffix=suffix,
            normalize_embeddings=normalize_embeddings,
            onnx_execution_provider=onnx_execution_provider,
            batch_size=1,
            progress_bar=False,
            pooling_mode=pooling_mode,
            model_kwargs=model_kwargs,
        )
        self._backend = _EmbedderBackend(params)
        self._initialized = False

    def warm_up(self):
        """
        Load the embedding backend.
        """
        if self._initialized:
            return

        self._backend.warm_up()
        self._initialized = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        init_params = self._backend.parameters.serialize()
        # Remove init params that are not provided to the text embedder.
        init_params.pop("batch_size")
        init_params.pop("progress_bar")
        return default_to_dict(self, **init_params)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimumTextEmbedder":
        """
        Deserialize this component from a dictionary.
        """
        _EmbedderParams.deserialize_inplace(data["init_parameters"])
        return default_from_dict(cls, data)

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """
        Embed a string.

        :param text: The text to embed.
        :return: The embeddings of the text.
        """
        if not self._initialized:
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)

        if not isinstance(text, str):
            msg = (
                "OptimumTextEmbedder expects a string as an input. "
                "In case you want to embed a list of Documents, please use the OptimumDocumentEmbedder."
            )
            raise TypeError(msg)

        text_to_embed = self._backend.parameters.prefix + text + self._backend.parameters.suffix
        embedding = self._backend.embed_texts(text_to_embed)
        return {"embedding": embedding}
