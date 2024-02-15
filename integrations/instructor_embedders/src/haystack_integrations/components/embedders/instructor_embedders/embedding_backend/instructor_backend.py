# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import ClassVar, Dict, List, Optional

from haystack.utils.auth import Secret
from InstructorEmbedding import INSTRUCTOR


class _InstructorEmbeddingBackendFactory:
    """
    Factory class to create instances of INSTRUCTOR embedding backends.
    """

    _instances: ClassVar[Dict[str, "_InstructorEmbeddingBackend"]] = {}

    @staticmethod
    def get_embedding_backend(model: str, device: Optional[str] = None, token: Optional[Secret] = None):
        embedding_backend_id = f"{model}{device}{token}"

        if embedding_backend_id in _InstructorEmbeddingBackendFactory._instances:
            return _InstructorEmbeddingBackendFactory._instances[embedding_backend_id]

        embedding_backend = _InstructorEmbeddingBackend(model=model, device=device, token=token)
        _InstructorEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
        return embedding_backend


class _InstructorEmbeddingBackend:
    """
    Class to manage INSTRUCTOR embeddings.
    """

    def __init__(self, model: str, device: Optional[str] = None, token: Optional[Secret] = None):
        self.model = INSTRUCTOR(
            model_name_or_path=model,
            device=device,
            use_auth_token=token.resolve_value() if token else None,
        )

    def embed(self, data: List[List[str]], **kwargs) -> List[List[float]]:
        embeddings = self.model.encode(data, **kwargs).tolist()
        return embeddings
