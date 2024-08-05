from unittest.mock import MagicMock, NonCallableMagicMock

import numpy as np
import pytest
from gradientai.openapi.client.models.generate_embedding_success import (
    GenerateEmbeddingSuccess,
)
from haystack.utils import Secret

from haystack_integrations.components.embedders.gradient import GradientTextEmbedder

access_token = "access_token"
workspace_id = "workspace_id"
model = "bge-large"


@pytest.fixture
def tokens_from_env(monkeypatch):
    monkeypatch.setenv("GRADIENT_ACCESS_TOKEN", access_token)
    monkeypatch.setenv("GRADIENT_WORKSPACE_ID", workspace_id)


class TestGradientTextEmbedder:
    def test_init_from_env(self, tokens_from_env):
        embedder = GradientTextEmbedder()
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token

    def test_init_without_access_token(self, monkeypatch):
        monkeypatch.delenv("GRADIENT_ACCESS_TOKEN", raising=False)

        with pytest.raises(ValueError):
            GradientTextEmbedder()

    def test_init_without_workspace(self, monkeypatch):
        monkeypatch.delenv("GRADIENT_WORKSPACE_ID", raising=False)

        with pytest.raises(ValueError):
            GradientTextEmbedder()

    def test_init_from_params(self):
        embedder = GradientTextEmbedder(
            access_token=Secret.from_token(access_token),
            workspace_id=Secret.from_token(workspace_id),
        )
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token

    def test_init_from_params_precedence(self, monkeypatch):
        monkeypatch.setenv("GRADIENT_ACCESS_TOKEN", "env_access_token")
        monkeypatch.setenv("GRADIENT_WORKSPACE_ID", "env_workspace_id")

        embedder = GradientTextEmbedder(
            access_token=Secret.from_token(access_token),
            workspace_id=Secret.from_token(workspace_id),
        )
        assert embedder is not None
        assert embedder._gradient.workspace_id == workspace_id
        assert embedder._gradient._api_client.configuration.access_token == access_token

    def test_to_dict(self, tokens_from_env):
        component = GradientTextEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.embedders.gradient.gradient_text_embedder.GradientTextEmbedder",
            "init_parameters": {
                "access_token": {
                    "env_vars": ["GRADIENT_ACCESS_TOKEN"],
                    "strict": True,
                    "type": "env_var",
                },
                "host": None,
                "model": "bge-large",
                "workspace_id": {
                    "env_vars": ["GRADIENT_WORKSPACE_ID"],
                    "strict": True,
                    "type": "env_var",
                },
            },
        }

    def test_warmup(self, tokens_from_env):
        embedder = GradientTextEmbedder()
        embedder._gradient.get_embeddings_model = MagicMock()
        embedder.warm_up()
        embedder._gradient.get_embeddings_model.assert_called_once_with(
            slug="bge-large"
        )

    def test_warmup_doesnt_reload(self, tokens_from_env):
        embedder = GradientTextEmbedder()
        embedder._gradient.get_embeddings_model = MagicMock(
            default_return_value="fake model"
        )
        embedder.warm_up()
        embedder.warm_up()
        embedder._gradient.get_embeddings_model.assert_called_once_with(
            slug="bge-large"
        )

    def test_run_fail_if_not_warmed_up(self, tokens_from_env):
        embedder = GradientTextEmbedder()

        with pytest.raises(RuntimeError, match="warm_up()"):
            embedder.run(text="The food was delicious")

    def test_run_fail_when_no_embeddings_returned(self, tokens_from_env):
        embedder = GradientTextEmbedder()
        embedder._embedding_model = NonCallableMagicMock()
        embedder._embedding_model.embed.return_value = GenerateEmbeddingSuccess(
            embeddings=[]
        )

        with pytest.raises(RuntimeError):
            _result = embedder.run(text="The food was delicious")
            embedder._embedding_model.embed.assert_called_once_with(
                inputs=[{"input": "The food was delicious"}]
            )

    def test_run_empty_string(self, tokens_from_env):
        embedder = GradientTextEmbedder()
        embedder._embedding_model = NonCallableMagicMock()
        embedder._embedding_model.embed.return_value = GenerateEmbeddingSuccess(
            embeddings=[{"embedding": np.random.rand(1024).tolist(), "index": 0}]
        )

        result = embedder.run(text="")
        embedder._embedding_model.embed.assert_called_once_with(inputs=[{"input": ""}])

        assert len(result["embedding"]) == 1024  # 1024 is the bge-large embedding size
        assert all(isinstance(x, float) for x in result["embedding"])

    def test_run(self, tokens_from_env):
        embedder = GradientTextEmbedder()
        embedder._embedding_model = NonCallableMagicMock()
        embedder._embedding_model.embed.return_value = GenerateEmbeddingSuccess(
            embeddings=[{"embedding": np.random.rand(1024).tolist(), "index": 0}]
        )

        result = embedder.run(text="The food was delicious")
        embedder._embedding_model.embed.assert_called_once_with(
            inputs=[{"input": "The food was delicious"}]
        )

        assert len(result["embedding"]) == 1024  # 1024 is the bge-large embedding size
        assert all(isinstance(x, float) for x in result["embedding"])
