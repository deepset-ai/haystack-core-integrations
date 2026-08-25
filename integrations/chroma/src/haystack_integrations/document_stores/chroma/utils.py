# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import (
    CohereEmbeddingFunction,
    DefaultEmbeddingFunction,
    GooglePalmEmbeddingFunction,
    GoogleVertexEmbeddingFunction,
    HuggingFaceEmbeddingFunction,
    InstructorEmbeddingFunction,
    OllamaEmbeddingFunction,
    ONNXMiniLM_L6_V2,
    OpenAIEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
    Text2VecEmbeddingFunction,
)

from .errors import ChromaDocumentStoreConfigError

FUNCTION_REGISTRY = {
    "default": DefaultEmbeddingFunction,
    "SentenceTransformerEmbeddingFunction": SentenceTransformerEmbeddingFunction,
    "CohereEmbeddingFunction": CohereEmbeddingFunction,
    "GooglePalmEmbeddingFunction": GooglePalmEmbeddingFunction,
    "GoogleVertexEmbeddingFunction": GoogleVertexEmbeddingFunction,
    "HuggingFaceEmbeddingFunction": HuggingFaceEmbeddingFunction,
    "InstructorEmbeddingFunction": InstructorEmbeddingFunction,
    "OllamaEmbeddingFunction": OllamaEmbeddingFunction,
    "ONNXMiniLM_L6_V2": ONNXMiniLM_L6_V2,
    "OpenAIEmbeddingFunction": OpenAIEmbeddingFunction,
    "Text2VecEmbeddingFunction": Text2VecEmbeddingFunction,
}


def get_embedding_function(function_name: str, **kwargs: Any) -> EmbeddingFunction:
    """
    Load an embedding function by name.

    :param function_name: the name of the embedding function.
    :param kwargs: additional arguments to pass to the embedding function.
    :returns: the loaded embedding function.
    :raises ChromaDocumentStoreConfigError: if the function name is invalid or `trust_remote_code` is set.
    """
    if kwargs.get("trust_remote_code"):
        # CVE-2026-45829/45833: trust_remote_code lets a model repo run arbitrary code on load.
        msg = "Setting 'trust_remote_code=True' in embedding_function_params is not allowed."
        raise ChromaDocumentStoreConfigError(msg)
    try:
        return FUNCTION_REGISTRY[function_name](**kwargs)
    except KeyError:
        msg = f"Invalid function name: {function_name}"
        raise ChromaDocumentStoreConfigError(msg) from KeyError
