from typing import ClassVar, Dict, List

from fastembed import TextEmbedding


class _FastembedEmbeddingBackendFactory:
    """
    Factory class to create instances of fastembed embedding backends.
    """

    _instances: ClassVar[Dict[str, "_FastembedEmbeddingBackend"]] = {}

    @staticmethod
    def get_embedding_backend(
        model_name: str,
    ):
        embedding_backend_id = f"{model_name}"

        if embedding_backend_id in _FastembedEmbeddingBackendFactory._instances:
            return _FastembedEmbeddingBackendFactory._instances[embedding_backend_id]

        embedding_backend = _FastembedEmbeddingBackend(
            model_name=model_name,
        )
        _FastembedEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
        return embedding_backend


class _FastembedEmbeddingBackend:
    """
    Class to manage fastembed embeddings.
    """

    def __init__(
        self,
        model_name: str,
    ):
        self.model = TextEmbedding(model_name=model_name)

    def embed(self, data: List[List[str]], **kwargs) -> List[List[float]]:
        # the embed method returns a Iterable[np.ndarray], so we convert it to a list of lists
        embeddings = [np_array.tolist() for np_array in self.model.embed(data, **kwargs)]
        return embeddings
