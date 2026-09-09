# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.utils.auth import Secret
from huggingface_hub.utils import RepositoryNotFoundError
from numpy import array

from haystack_integrations.common.huggingface_api.utils import HFEmbeddingAPIType
from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPITextEmbedder


@pytest.fixture
def mock_check_valid_model():
    with (
        patch(
            "haystack_integrations.components.embedders.huggingface_api.text_embedder._check_valid_model",
            MagicMock(return_value=None),
        ) as mock,
        patch(
            "haystack_integrations.components.embedders.huggingface_api.text_embedder._check_valid_model_async",
            AsyncMock(return_value=None),
        ),
    ):
        yield mock


class TestInitializationAndSerialization:
    def test_init_invalid_api_type(self):
        with pytest.raises(ValueError):
            HuggingFaceAPITextEmbedder(api_type="invalid_api_type", api_params={})

    def test_init_serverless(self):
        model = "BAAI/bge-small-en-v1.5"
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": model}
        )

        assert embedder.api_type == HFEmbeddingAPIType.SERVERLESS_INFERENCE_API
        assert embedder.api_params == {"model": model}
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.truncate
        assert not embedder.normalize
        assert embedder._client is None
        assert embedder._async_client is None

    def test_init_serverless_no_model(self):
        with pytest.raises(ValueError):
            HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"param": "irrelevant"}
            )

    def test_init_tei(self):
        url = "https://some_model.com"

        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE, api_params={"url": url}
        )

        assert embedder.api_type == HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE
        assert embedder.api_params == {"url": url}
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.truncate
        assert not embedder.normalize
        assert embedder._client is None
        assert embedder._async_client is None

    def test_init_tei_invalid_url(self):
        with pytest.raises(ValueError):
            HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE, api_params={"url": "invalid_url"}
            )

    def test_init_tei_no_url(self):
        with pytest.raises(ValueError):
            HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE, api_params={"param": "irrelevant"}
            )

    def test_to_dict(self):
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "BAAI/bge-small-en-v1.5"},
            prefix="prefix",
            suffix="suffix",
            truncate=False,
            normalize=True,
        )

        data = embedder.to_dict()

        assert data == {
            "type": "haystack_integrations.components.embedders.huggingface_api.text_embedder"
            ".HuggingFaceAPITextEmbedder",
            "init_parameters": {
                "api_type": "serverless_inference_api",
                "api_params": {"model": "BAAI/bge-small-en-v1.5"},
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "prefix": "prefix",
                "suffix": "suffix",
                "truncate": False,
                "normalize": True,
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.embedders.huggingface_api.text_embedder"
            ".HuggingFaceAPITextEmbedder",
            "init_parameters": {
                "api_type": HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                "api_params": {"model": "BAAI/bge-small-en-v1.5"},
                "token": {"env_vars": ["HF_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "prefix": "prefix",
                "suffix": "suffix",
                "truncate": False,
                "normalize": True,
            },
        }

        embedder = HuggingFaceAPITextEmbedder.from_dict(data)

        assert embedder.api_type == HFEmbeddingAPIType.SERVERLESS_INFERENCE_API
        assert embedder.api_params == {"model": "BAAI/bge-small-en-v1.5"}
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert not embedder.truncate
        assert embedder.normalize


class TestComponentLifecycle:
    def test_key_resolved_at_warm_up_not_init(self, monkeypatch):
        monkeypatch.delenv("MISSING_HF_TOKEN", raising=False)
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://example.com"},
            token=Secret.from_env_var("MISSING_HF_TOKEN"),
        )

        with pytest.raises(ValueError, match="MISSING_HF_TOKEN"):
            embedder.warm_up()

    def test_invalid_model_is_checked_at_warm_up(self, mock_check_valid_model):
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id", response=MagicMock())
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "invalid_model_id"}
        )
        with pytest.raises(RepositoryNotFoundError):
            embedder.warm_up()

    @patch("haystack_integrations.components.embedders.huggingface_api.text_embedder.InferenceClient")
    def test_sync_lifecycle(self, mock_client_cls):
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://example.com"},
            token=Secret.from_token("test-token"),
        )
        client = mock_client_cls.return_value

        embedder.warm_up()
        assert embedder._client is client
        assert embedder._async_client is None

        embedder.close()
        client.close.assert_called_once_with()
        assert embedder._client is None

        embedder.warm_up()
        assert mock_client_cls.call_count == 2

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.embedders.huggingface_api.text_embedder.AsyncInferenceClient")
    async def test_async_lifecycle(self, mock_client_cls):
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://example.com"},
            token=Secret.from_token("test-token"),
        )
        client = MagicMock(close=AsyncMock())
        mock_client_cls.return_value = client

        await embedder.warm_up_async()
        assert embedder._async_client is client
        assert embedder._client is None

        await embedder.close_async()
        client.close.assert_awaited_once_with()
        assert embedder._async_client is None

        await embedder.warm_up_async()
        assert mock_client_cls.call_count == 2

    @patch("haystack_integrations.components.embedders.huggingface_api.text_embedder.InferenceClient")
    def test_warm_up_is_idempotent(self, mock_client_cls):
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://example.com"},
            token=None,
        )
        embedder.warm_up()
        embedder.warm_up()
        mock_client_cls.assert_called_once_with(model="https://example.com", token=None)

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.embedders.huggingface_api.text_embedder.AsyncInferenceClient")
    async def test_warm_up_async_is_idempotent(self, mock_client_cls):
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://example.com"},
            token=None,
        )
        await embedder.warm_up_async()
        await embedder.warm_up_async()
        mock_client_cls.assert_called_once_with(model="https://example.com", token=None)

    @pytest.mark.asyncio
    async def test_close_is_safe_without_warm_up(self):
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://example.com"},
            token=None,
        )
        embedder.close()
        await embedder.close_async()
        assert embedder._client is None
        assert embedder._async_client is None

    @pytest.mark.asyncio
    @patch("haystack_integrations.components.embedders.huggingface_api.text_embedder.AsyncInferenceClient")
    @patch("haystack_integrations.components.embedders.huggingface_api.text_embedder.InferenceClient")
    async def test_close_and_close_async_are_independent(self, mock_sync_cls, mock_async_cls):
        sync_client = mock_sync_cls.return_value
        async_client = MagicMock(close=AsyncMock())
        mock_async_cls.return_value = async_client
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE, api_params={"url": "https://example.com"}
        )
        embedder.warm_up()
        await embedder.warm_up_async()

        embedder.close()
        assert embedder._client is None
        assert embedder._async_client is async_client
        async_client.close.assert_not_awaited()

        await embedder.close_async()
        assert embedder._async_client is None
        sync_client.close.assert_called_once_with()


class TestRun:
    def test_run_wrong_input_format(self, mock_check_valid_model):
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "BAAI/bge-small-en-v1.5"}
        )

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError):
            embedder.run(text=list_integers_input)

    def test_run(self, mock_check_valid_model, caplog):
        with patch("huggingface_hub.InferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.return_value = array([[random.random() for _ in range(384)]])

            embedder = HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
                prefix="prefix ",
                suffix=" suffix",
            )

            result = embedder.run(text="The food was delicious")

            mock_embedding_patch.assert_called_once_with(
                text="prefix The food was delicious suffix", truncate=None, normalize=None
            )

        assert len(result["embedding"]) == 384
        assert all(isinstance(x, float) for x in result["embedding"])

        # Check that warnings about ignoring truncate and normalize are raised
        assert len(caplog.records) == 2
        assert "truncate" in caplog.records[0].message
        assert "normalize" in caplog.records[1].message

    @pytest.mark.asyncio
    async def test_run_async(self, mock_check_valid_model, caplog):
        with patch("huggingface_hub.AsyncInferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.return_value = array([[random.random() for _ in range(384)]])

            embedder = HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
                prefix="prefix ",
                suffix=" suffix",
            )

            result = await embedder.run_async(text="The food was delicious")

            mock_embedding_patch.assert_called_once_with(
                text="prefix The food was delicious suffix", truncate=None, normalize=None
            )

        assert len(result["embedding"]) == 384
        assert all(isinstance(x, float) for x in result["embedding"])

        # Check that warnings about ignoring truncate and normalize are raised
        assert len(caplog.records) == 2
        assert "truncate" in caplog.records[0].message
        assert "normalize" in caplog.records[1].message

    def test_run_wrong_embedding_shape(self, mock_check_valid_model):
        # embedding ndim > 2
        with patch("huggingface_hub.InferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.return_value = array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]])

            embedder = HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "BAAI/bge-small-en-v1.5"}
            )

            with pytest.raises(ValueError):
                embedder.run(text="The food was delicious")

        # embedding ndim == 2 but shape[0] != 1
        with patch("huggingface_hub.InferenceClient.feature_extraction") as mock_embedding_patch:
            mock_embedding_patch.return_value = array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

            embedder = HuggingFaceAPITextEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "BAAI/bge-small-en-v1.5"}
            )

            with pytest.raises(ValueError):
                embedder.run(text="The food was delicious")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN", None),
    reason="Export an env var called HF_TOKEN containing the Hugging Face token to run this test.",
)
class TestIntegration:
    def test_live_run_serverless(self):
        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "sentence-transformers/all-MiniLM-L6-v2"},
        )
        embedder.warm_up()
        assert embedder._client is not None
        embedder._client.timeout = 10  # we want to fail fast if the server is not responding
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 384
        assert all(isinstance(x, float) for x in result["embedding"])

    @pytest.mark.asyncio
    async def test_live_run_async_serverless(self):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        embedder = HuggingFaceAPITextEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": model_name}
        )
        await embedder.warm_up_async()
        assert embedder._async_client is not None
        embedder._async_client.timeout = 10  # we want to fail fast if the server is not responding

        text = "This is a test sentence for embedding."
        result = await embedder.run_async(text=text)

        assert "embedding" in result
        assert isinstance(result["embedding"], list)
        assert all(isinstance(x, float) for x in result["embedding"])
        assert len(result["embedding"]) == 384  # MiniLM-L6-v2 has 384 dimensions
