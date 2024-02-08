from typing import Any, Dict, List, Optional

import torch
from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.hf import HFModelType, check_valid_model, deserialize_hf_model_kwargs, serialize_hf_model_kwargs
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer


class OptimumTextEmbedder:
    """
    A component to embed text using models loaded with the HuggingFace Optimum library.
    This component is designed to seamlessly inference models using the high speed ONNX runtime.

    Usage example:
    ```python
    from haystack_integrations.components.embedders import OptimumTextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = OptimumTextEmbedder(model="BAAI/bge-small-en-v1.5")
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
        model: str = "BAAI/bge-small-en-v1.5",
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),  # noqa: B008
        prefix: str = "",
        suffix: str = "",
        normalize_embeddings: bool = True,
        onnx_execution_provider: str = "CPUExecutionProvider",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a OptimumTextEmbedder component.

        :param model: A string representing the model id on HF Hub. Default is "BAAI/bge-small-en-v1.5".
        :param token: The HuggingFace token to use as HTTP bearer authorization.
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        :param normalize_embeddings: Whether to normalize the embeddings to unit length.
        :param onnx_execution_provider: The execution provider to use for ONNX models. Defaults to
        "CPUExecutionProvider".
        :param model_kwargs: Dictionary containing additional keyword arguments to pass to the model.
            In case of duplication, these kwargs override `model`, `onnx_execution_provider`, and `token` initialization
            parameters.
        """
        check_valid_model(model, HFModelType.EMBEDDING, token)
        self.model = model

        self.token = token
        token = token.resolve_value() if token else None

        self.prefix = prefix
        self.suffix = suffix
        self.normalize_embeddings = normalize_embeddings
        self.onnx_execution_provider = onnx_execution_provider

        model_kwargs = model_kwargs or {}

        # Check if the model_kwargs contain the parameters, otherwise, populate them with values from init parameters
        model_kwargs.setdefault("model_id", model)
        model_kwargs.setdefault("provider", onnx_execution_provider)
        model_kwargs.setdefault("use_auth_token", token)

        self.model_kwargs = model_kwargs
        self.embedding_model = None
        self.tokenizer = None

    def warm_up(self):
        """
        Convert the model to ONNX.

        The model is cached if the "TensorrtExecutionProvider" is used, since it takes a while to build the TensorRT
        engine.
        """
        if self.embedding_model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)

            if self.onnx_execution_provider == "TensorrtExecutionProvider":
                # Cache engine for TensorRT
                provider_options = {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": f"tmp/trt_cache_{self.model}",
                }
                self.embedding_model = ORTModelForFeatureExtraction.from_pretrained(
                    **self.model_kwargs, use_cache=False, provider_options=provider_options
                )
            else:
                # export=True converts the model to ONNX on the fly
                self.embedding_model = ORTModelForFeatureExtraction.from_pretrained(**self.model_kwargs, export=True)

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
            model_kwargs=self.model_kwargs,
            token=self.token.to_dict() if self.token else None,
        )

        model_kwargs = serialization_dict["init_parameters"]["model_kwargs"]
        model_kwargs.pop("token", None)

        serialize_hf_model_kwargs(model_kwargs)
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimumTextEmbedder":
        """
        Deserialize this component from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        deserialize_hf_model_kwargs(data["init_parameters"]["model_kwargs"])
        return default_from_dict(cls, data)

    def mean_pooling(self, model_output: torch.tensor, attention_mask: torch.tensor) -> torch.tensor:
        """
        Perform Mean Pooling on the output of the Embedding model.

        :param model_output: The output of the embedding model.
        :param attention_mask: The attention mask of the tokenized text.
        :return: The embeddings of the text after mean pooling.
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """Embed a string.

        :param text: The text to embed.
        :return: The embeddings of the text.
        """
        if not isinstance(text, str):
            msg = (
                "OptimumTextEmbedder expects a string as an input. "
                "In case you want to embed a list of Documents, please use the OptimumDocumentEmbedder."
            )
            raise TypeError(msg)

        if not (self.embedding_model and self.tokenizer):
            msg = "The embedding model has not been loaded. Please call warm_up() before running."
            raise RuntimeError(msg)

        text_to_embed = self.prefix + text + self.suffix

        # Determine device for tokenizer output
        device = (
            "cuda"
            if self.onnx_execution_provider
            in ["CUDAExecutionProvider", "ROCMExecutionProvider", "TensorrtExecutionProvider"]
            else "cpu"
        )

        encoded_input = self.tokenizer([text_to_embed], padding=True, truncation=True, return_tensors="pt").to(device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)

        # Perform mean pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"].cpu())

        # Normalize Embeddings
        if self.normalize_embeddings:
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        embedding = sentence_embeddings.tolist()[0]

        return {"embedding": embedding}
