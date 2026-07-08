# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from haystack.utils import Secret
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError


class HFGenerationAPIType(Enum):
    """
    API type to use for Hugging Face API Generators.
    """

    # HF [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference).
    TEXT_GENERATION_INFERENCE = "text_generation_inference"

    # HF [Inference Endpoints](https://huggingface.co/inference-endpoints).
    INFERENCE_ENDPOINTS = "inference_endpoints"

    # HF [Serverless Inference API](https://huggingface.co/inference-api).
    SERVERLESS_INFERENCE_API = "serverless_inference_api"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_str(string: str) -> "HFGenerationAPIType":
        """
        Convert a string to a HFGenerationAPIType enum.

        :param string: The string to convert.
        :return: The corresponding HFGenerationAPIType enum.
        """
        enum_map = {e.value: e for e in HFGenerationAPIType}
        mode = enum_map.get(string)
        if mode is None:
            msg = f"Unknown Hugging Face API type '{string}'. Supported types are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return mode


class HFEmbeddingAPIType(Enum):
    """
    API type to use for Hugging Face API Embedders.
    """

    # HF [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference).
    TEXT_EMBEDDINGS_INFERENCE = "text_embeddings_inference"

    # HF [Inference Endpoints](https://huggingface.co/inference-endpoints).
    INFERENCE_ENDPOINTS = "inference_endpoints"

    # HF [Serverless Inference API](https://huggingface.co/inference-api).
    SERVERLESS_INFERENCE_API = "serverless_inference_api"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_str(string: str) -> "HFEmbeddingAPIType":
        """
        Convert a string to a HFEmbeddingAPIType enum.

        :param string: The string to convert.
        :return: The corresponding HFEmbeddingAPIType enum.
        """
        enum_map = {e.value: e for e in HFEmbeddingAPIType}
        mode = enum_map.get(string)
        if mode is None:
            msg = f"Unknown Hugging Face API type '{string}'. Supported types are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return mode


class HFModelType(Enum):
    EMBEDDING = 1
    GENERATION = 2


def _check_valid_model(model_id: str, model_type: HFModelType, token: Secret | None) -> None:
    """
    Check if the provided model ID corresponds to a valid model on HuggingFace Hub.

    Also check if the model is an embedding or generation model.

    :param model_id: A string representing the HuggingFace model ID.
    :param model_type: the model type, HFModelType.EMBEDDING or HFModelType.GENERATION
    :param token: The optional authentication token.
    :raises ValueError: If the model is not found or is not a embedding model.
    """
    api = HfApi()
    try:
        model_info = api.model_info(model_id, token=token.resolve_value() if token else None)
    except RepositoryNotFoundError as e:
        msg = f"Model {model_id} not found on HuggingFace Hub. Please provide a valid HuggingFace model_id."
        raise ValueError(msg) from e

    if model_type == HFModelType.EMBEDDING:
        allowed_model = model_info.pipeline_tag in ["sentence-similarity", "feature-extraction"]
        error_msg = f"Model {model_id} is not a embedding model. Please provide a embedding model."
    elif model_type == HFModelType.GENERATION:
        allowed_model = model_info.pipeline_tag in ["text-generation", "text2text-generation", "image-text-to-text"]
        error_msg = f"Model {model_id} is not a text generation model. Please provide a text generation model."
    else:
        allowed_model = False
        error_msg = f"Unknown model type for {model_id}"

    if not allowed_model:
        raise ValueError(error_msg)
