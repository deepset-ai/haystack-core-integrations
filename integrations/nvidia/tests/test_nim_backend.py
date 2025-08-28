# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import patch

import pytest
import requests

from haystack_integrations.utils.nvidia import DEFAULT_API_URL, Client, NimBackend
from haystack_integrations.utils.nvidia.nim_backend import DEFAULT_MODELS, REQUEST_TIMEOUT


def mock_embed_post_response(*args, **kwargs):  # noqa: ARG001
    inputs = kwargs["json"]["input"]
    model = kwargs["json"]["model"]
    mock_response = requests.Response()
    mock_response.status_code = 200
    data = [
        {"object": "embedding", "index": i, "embedding": [i / 10.0, 0.1, 0.1]} for i in reversed(range(len(inputs)))
    ]
    mock_response._content = json.dumps(
        {"model": model, "object": "list", "usage": {"total_tokens": 4, "prompt_tokens": 4}, "data": data}
    ).encode()
    return mock_response


def mock_generate_post_response(*args, **kwargs):  # noqa: ARG001
    prompt = kwargs["json"]["messages"][0]["content"]
    model = kwargs["json"]["model"]
    mock_response = requests.Response()
    mock_response.status_code = 200
    choices = [
        {
            "index": i,
            "message": {"role": "assistant", "content": f"Response {i} to '{prompt}'"},
            "finish_reason": "stop",
        }
        for i in range(3)
    ]
    mock_response._content = json.dumps(
        {
            "model": model,
            "object": "chat.completion",
            "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
            "choices": choices,
        }
    ).encode()
    return mock_response


def mock_models_get_response(*args, **kwargs):  # noqa: ARG001
    mock_response = requests.Response()
    mock_response.status_code = 200
    data = [{"id": f"model_{i}", "object": "model", "root": f"base_model_{i}"} for i in range(3)]
    mock_response._content = json.dumps({"object": "list", "data": data}).encode()
    return mock_response


def mock_rank_post_response(*args, **kwargs):  # noqa: ARG001
    passages = kwargs["json"]["passages"]
    model = kwargs["json"]["model"]
    mock_response = requests.Response()
    mock_response.status_code = 200
    data = [
        {"object": "ranking", "index": i, "text": passage["text"], "score": 1.0 - (i * 0.1)}
        for i, passage in enumerate(passages)
    ]
    mock_response._content = json.dumps(
        {"model": model, "object": "list", "usage": {"total_tokens": 4, "prompt_tokens": 4}, "rankings": data}
    ).encode()
    return mock_response


class TestNimBackend:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        backend = NimBackend(model="nvidia/nv-embedqa-e5-v5", api_url=DEFAULT_API_URL, client="NvidiaTextEmbedder")
        assert backend.api_url == DEFAULT_API_URL
        assert backend.client == Client.NVIDIA_TEXT_EMBEDDER
        assert backend.model == "nvidia/nv-embedqa-e5-v5"
        assert backend.model_kwargs == {}
        assert backend.model_type is None
        assert backend.session.headers["Content-Type"] == "application/json"
        assert backend.session.headers["accept"] == "application/json"
        assert backend.session.headers["authorization"] == "Bearer fake-api-key"
        assert backend.timeout == REQUEST_TIMEOUT

    def test_init_with_client_enum(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        backend = NimBackend(model="custom-model", api_url="http://localhost:8000", client=Client.NVIDIA_TEXT_EMBEDDER)
        assert backend.client == Client.NVIDIA_TEXT_EMBEDDER

    def test_init_without_client(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        backend = NimBackend(model="custom-model", api_url="http://localhost:8000")
        assert backend.client is None
        assert backend.model_type is None
        assert backend.model == "custom-model"

    def test_init_hosted_missing_model_and_model_type_raises_error(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")

        with pytest.raises(ValueError, match="`model_type` is required when `model` is not specified"):
            NimBackend(api_url=DEFAULT_API_URL, client="NvidiaTextEmbedder")

    def test_init_hosted_with_model_type_uses_default_model(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")

        backend = NimBackend(model_type="embedding", api_url=DEFAULT_API_URL, client="NvidiaTextEmbedder")
        assert backend.model == DEFAULT_MODELS["embedding"]

    def test_init_hosted_model_with_custom_endpoint_overrides_api_url(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        backend = NimBackend(
            model="nvidia/nv-rerankqa-mistral-4b-v3",
            api_url=DEFAULT_API_URL,
            client="NvidiaRanker",
            model_type="ranking",
        )
        assert backend.api_url == "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking"

    def test_init_hosted_model_without_custom_endpoint_keeps_original_url(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")

        backend = NimBackend(
            model="nvidia/nv-embedqa-e5-v5",  # This model has no custom endpoint
            api_url=DEFAULT_API_URL,
            client="NvidiaTextEmbedder",
        )
        assert backend.api_url == DEFAULT_API_URL
        assert backend.model == "nvidia/nv-embedqa-e5-v5"

    def test_init_with_unknown_hosted_model_raises_error(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")

        with pytest.raises(ValueError, match="Model unknown-model is unknown"):
            NimBackend(model="unknown-model", api_url=DEFAULT_API_URL, client="NvidiaTextEmbedder")

    def test_init_with_incompatible_client_raises_error(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")

        with pytest.raises(ValueError, match="Model nvidia/nv-embedqa-e5-v5 is incompatible with client"):
            NimBackend(
                model="nvidia/nv-embedqa-e5-v5",  # embedding model
                api_url=DEFAULT_API_URL,
                client="NvidiaGenerator",  # chat client
            )

    def test_init_with_non_hosted_model(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        backend = NimBackend(model="unknown-model", api_url="http://localhost:8000", client="NvidiaTextEmbedder")

        # validation is skipped for non-hosted models
        assert backend.model == "unknown-model"
        assert backend.api_url == "http://localhost:8000"
        assert backend.model_type is None

    def test_embed(self, monkeypatch):
        with patch("requests.sessions.Session.post", side_effect=mock_embed_post_response) as mock_post:
            monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
            backend = NimBackend(model="nvidia/nv-embedqa-e5-v5", api_url=DEFAULT_API_URL, client="NvidiaTextEmbedder")
            texts = ["a", "b", "c"]
            embeddings, meta = backend.embed(texts=texts)

            assert embeddings == [[0.0, 0.1, 0.1], [0.1, 0.1, 0.1], [0.2, 0.1, 0.1]]
            assert meta == {"usage": {"prompt_tokens": 4, "total_tokens": 4}}

            expected_url = DEFAULT_API_URL + "/embeddings"
            mock_post.assert_called_once_with(
                expected_url,
                json={
                    "model": "nvidia/nv-embedqa-e5-v5",
                    "input": texts,
                },
                timeout=60.0,
            )

    def test_generate(self, monkeypatch):
        with patch("requests.sessions.Session.post", side_effect=mock_generate_post_response) as mock_post:
            monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
            backend = NimBackend(model="meta/llama3-8b-instruct", api_url=DEFAULT_API_URL, client="NvidiaGenerator")
            prompt = "a"
            replies, meta = backend.generate(prompt=prompt)
            assert replies == ["Response 0 to 'a'", "Response 1 to 'a'", "Response 2 to 'a'"]
            assert meta == [
                {
                    "role": "assistant",
                    "usage": {"prompt_tokens": 5, "total_tokens": 10, "completion_tokens": 5},
                    "finish_reason": "stop",
                },
                {
                    "role": "assistant",
                    "usage": {"prompt_tokens": 5, "total_tokens": 10, "completion_tokens": 5},
                    "finish_reason": "stop",
                },
                {
                    "role": "assistant",
                    "usage": {"prompt_tokens": 5, "total_tokens": 10, "completion_tokens": 5},
                    "finish_reason": "stop",
                },
            ]

            expected_url = DEFAULT_API_URL + "/chat/completions"
            mock_post.assert_called_once_with(
                expected_url,
                json={
                    "model": "meta/llama3-8b-instruct",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                },
                timeout=60.0,
            )

    def test_models(self, monkeypatch):
        with patch("requests.sessions.Session.get", side_effect=mock_models_get_response) as mock_get:
            monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
            backend = NimBackend(
                model="nvidia/nv-embedqa-e5-v5", api_url=DEFAULT_API_URL, client="NvidiaDocumentEmbedder"
            )
            models = backend.models()

            assert len(models) == 3
            assert all(model.client == Client.NVIDIA_DOCUMENT_EMBEDDER for model in models)
            expected_url = DEFAULT_API_URL + "/models"
            mock_get.assert_called_once_with(
                expected_url,
                timeout=60.0,
            )

    def test_rank(self, monkeypatch):
        with patch("requests.sessions.Session.post", side_effect=mock_rank_post_response) as mock_post:
            monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
            backend = NimBackend(
                model="nvidia/llama-3.2-nv-rerankqa-1b-v2", api_url=DEFAULT_API_URL, client="NvidiaRanker"
            )
            query_text = "query"
            document_texts = ["text1", "text2"]
            rankings = backend.rank(query_text=query_text, document_texts=document_texts)

            assert rankings == [
                {"index": 0, "object": "ranking", "score": 1.0, "text": "text1"},
                {"index": 1, "object": "ranking", "score": 0.9, "text": "text2"},
            ]

            expected_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
            mock_post.assert_called_once_with(
                expected_url,
                json={
                    "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
                    "query": {"text": query_text},
                    "passages": [{"text": text} for text in document_texts],
                },
                timeout=60.0,
            )
