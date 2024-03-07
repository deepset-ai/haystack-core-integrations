from typing import ClassVar, Dict, List, Optional

from tqdm import tqdm

from fastembed import TextEmbedding


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
