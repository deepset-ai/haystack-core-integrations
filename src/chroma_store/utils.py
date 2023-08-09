# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
from chromadb.api.types import EmbeddingFunction
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
    CohereEmbeddingFunction,
    GooglePalmEmbeddingFunction,
    GoogleVertexEmbeddingFunction,
    HuggingFaceEmbeddingFunction,
    InstructorEmbeddingFunction,
    ONNXMiniLM_L6_V2,
    OpenAIEmbeddingFunction,
    Text2VecEmbeddingFunction,
    DefaultEmbeddingFunction,
)

from chroma_store.errors import ChromaDocumentStoreConfigError

FUNCTION_REGISTRY = {
    "default": DefaultEmbeddingFunction,
    "SentenceTransformerEmbeddingFunction": SentenceTransformerEmbeddingFunction,
    "CohereEmbeddingFunction": CohereEmbeddingFunction,
    "GooglePalmEmbeddingFunction": GooglePalmEmbeddingFunction,
    "GoogleVertexEmbeddingFunction": GoogleVertexEmbeddingFunction,
    "HuggingFaceEmbeddingFunction": HuggingFaceEmbeddingFunction,
    "InstructorEmbeddingFunction": InstructorEmbeddingFunction,
    "ONNXMiniLM_L6_V2": ONNXMiniLM_L6_V2,
    "OpenAIEmbeddingFunction": OpenAIEmbeddingFunction,
    "Text2VecEmbeddingFunction": Text2VecEmbeddingFunction,
}


def get_embedding_function(function_name: str) -> EmbeddingFunction:
    try:
        return FUNCTION_REGISTRY[function_name]
    except KeyError:
        raise ChromaDocumentStoreConfigError(f"Invalid function name: {function_name}")
