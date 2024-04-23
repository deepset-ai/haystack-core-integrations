import numpy as np
import pytest
from haystack.dataclasses import SparseEmbedding


@pytest.fixture
def generate_sparse_embedding():
    """
    This fixture generates a random SparseEmbedding object each time it is used.
    """

    def _generate_random_sparse_embedding():
        random_indice_length = np.random.randint(3, 15)
        indices = list(range(random_indice_length))
        values = [np.random.random_sample() for _ in range(random_indice_length)]
        return SparseEmbedding(indices=indices, values=values)

    return _generate_random_sparse_embedding
