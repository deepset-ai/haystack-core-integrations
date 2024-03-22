from typing import ClassVar, Dict, List, Optional

from haystack.dataclasses.sparse_embedding import SparseEmbedding
from tqdm import tqdm

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

    def embed(self, data: List[str], progress_bar=True, **kwargs) -> List[List[float]]:
        # the embed method returns a Iterable[np.ndarray], so we convert it to a list of lists
        embeddings = []
        embeddings_iterable = self.model.embed(data, **kwargs)
        for np_array in tqdm(
            embeddings_iterable, disable=not progress_bar, desc="Calculating embeddings", total=len(data)
        ):
            embeddings.append(np_array.tolist())
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

        embedding_backend = _FastembedSparseEmbeddingBackend(
            model_name=model_name, cache_dir=cache_dir, threads=threads
        )
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

    def embed(self, data: List[List[str]], progress_bar=True, **kwargs) -> List[SparseEmbedding]:
        # The embed method returns a Iterable[SparseEmbedding], so we convert to Haystack SparseEmbedding type.
        # Each SparseEmbedding contains an `indices` key containing a list of int and
        # an `values` key containing a list of floats.

        sparse_embeddings = []
        sparse_embeddings_iterable = self.model.embed(data, **kwargs)
        for sparse_embedding in tqdm(
            sparse_embeddings_iterable, disable=not progress_bar, desc="Calculating sparse embeddings", total=len(data)
        ):
            sparse_embeddings.append(
                SparseEmbedding(indices=sparse_embedding.indices.tolist(), values=sparse_embedding.values.tolist())
            )

        return sparse_embeddings
