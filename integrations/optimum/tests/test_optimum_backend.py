import pytest
from haystack_integrations.components.embedders.optimum.optimum_backend import OptimumEmbeddingBackend
from haystack_integrations.components.embedders.optimum.pooling import PoolingMode


@pytest.fixture
def backend():
    model = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"model_id": model}
    backend = OptimumEmbeddingBackend(model=model, model_kwargs=model_kwargs, token=None)
    return backend


class TestOptimumBackend:
    def test_embed_output_order(self, backend):
        texts_to_embed = ["short text", "text that is longer than the other", "medium length text"]
        embeddings = backend.embed(texts_to_embed, normalize_embeddings=False, pooling_mode=PoolingMode.MEAN)

        # Compute individual embeddings in order
        expected_embeddings = []
        for text in texts_to_embed:
            expected_embeddings.append(backend.embed(text, normalize_embeddings=False, pooling_mode=PoolingMode.MEAN))

        # Assert that the embeddings are in the same order
        assert embeddings == expected_embeddings

    def test_run_pooling_modes(self, backend):
        for pooling_mode in PoolingMode:
            embedding = backend.embed("test text", normalize_embeddings=False, pooling_mode=pooling_mode)

            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)
