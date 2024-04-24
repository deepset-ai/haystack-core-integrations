import numpy as np
import pytest
from haystack.dataclasses import SparseEmbedding


@pytest.fixture(scope="session")
def generate_sparse_embedding():
    """
    This fixture returns a function that generates a random SparseEmbedding each time it is called.
    """

    def _generate_random_sparse_embedding():
        random_indice_length = np.random.randint(3, 15)
        indices = list(range(random_indice_length))
        values = [np.random.random_sample() for _ in range(random_indice_length)]
        return SparseEmbedding(indices=indices, values=values)

    return _generate_random_sparse_embedding
