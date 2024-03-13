from typing import ClassVar, Dict, List, Optional

import numpy as np

from fastembed import TextEmbedding
from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding

class _FastembedEmbeddingBackendFactory:
    """
    Factory class to create instances of fastembed embedding backends.
    """

    _instances: ClassVar[Dict[str, "_FastembedEmbeddingBackend"]] = {}

    @staticmethod
    def get_embedding_backend(
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
    ):
        embedding_backend_id = f"{model_name}{cache_dir}{threads}"

        if embedding_backend_id in _FastembedEmbeddingBackendFactory._instances:
            return _FastembedEmbeddingBackendFactory._instances[embedding_backend_id]

        embedding_backend = _FastembedEmbeddingBackend(model_name=model_name, cache_dir=cache_dir, threads=threads)
        _FastembedEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
        return embedding_backend


class _FastembedEmbeddingBackend:
    """
    Class to manage fastembed embeddings.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
    ):
        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads)

    def embed(self, data: List[List[str]], **kwargs) -> List[List[float]]:
        # the embed method returns a Iterable[np.ndarray], so we convert it to a list of lists
        embeddings = [np_array.tolist() for np_array in self.model.embed(data, **kwargs)]
        return embeddings

class _FastembedSparseEmbeddingBackendFactory:
    """
    Factory class to create instances of fastembed sparse embedding backends.
    """

    _instances: ClassVar[Dict[str, "_FastembedSparseEmbeddingBackend"]] = {}

    @staticmethod
    def get_embedding_backend(
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
    ):
        embedding_backend_id = f"{model_name}{cache_dir}{threads}"

        if embedding_backend_id in _FastembedSparseEmbeddingBackendFactory._instances:
            return _FastembedSparseEmbeddingBackendFactory._instances[embedding_backend_id]

        embedding_backend = _FastembedSparseEmbeddingBackend(model_name=model_name, cache_dir=cache_dir, threads=threads)
        _FastembedSparseEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
        return embedding_backend

class _FastembedSparseEmbeddingBackend:
    """
    Class to manage fastembed sparse embeddings.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
    ):
        self.model = SparseTextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads)

    def embed(self, data: List[List[str]], **kwargs) -> List[Dict[str, np.ndarray]]:
        # The embed method returns a Iterable[SparseEmbedding], so we convert it to a list of dictionaries
        sparse_embeddings = [sparse_embedding.as_object() for sparse_embedding in self.model.embed(data, **kwargs)]
        return sparse_embeddings
