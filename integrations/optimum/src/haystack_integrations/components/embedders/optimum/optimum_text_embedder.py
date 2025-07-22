from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret

from ._backend import _EmbedderBackend, _EmbedderParams
from .optimization import OptimumEmbedderOptimizationConfig
from .pooling import OptimumEmbedderPooling
from .quantization import OptimumEmbedderQuantizationConfig


@component
class OptimumTextEmbedder:
    """
    A component to embed text using models loaded with the
    [HuggingFace Optimum](https://huggingface.co/docs/optimum/index) library,
    leveraging the ONNX runtime for high-speed inference.

    Usage example:
    ```python
    from haystack_integrations.components.embedders.optimum import OptimumTextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = OptimumTextEmbedder(model="sentence-transformers/all-mpnet-base-v2")
    text_embedder.warm_up()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [-0.07804739475250244, 0.1498992145061493,, ...]}
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
    ):
        """
        Create a OptimumTextEmbedder component.

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
            working_dir=working_dir,
            optimizer_settings=optimizer_settings,
            quantizer_settings=quantizer_settings,
        )
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
        # Remove init params that are not provided to the text embedder.
        init_params.pop("batch_size")
        init_params.pop("progress_bar")
        return default_to_dict(self, **init_params)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimumTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        _EmbedderParams.deserialize_inplace(data["init_parameters"])
        return default_from_dict(cls, data)

    @component.output_types(embedding=List[float])
    def run(self, text: str) -> Dict[str, List[float]]:
        """
        Embed a string.

        :param text:
            The text to embed.
        :returns:
            The embeddings of the text.
        :raises RuntimeError:
            If the component was not initialized.
        :raises TypeError:
            If the input is not a string.
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
