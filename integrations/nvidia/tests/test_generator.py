# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from haystack.utils import Secret
from requests_mock import Mocker

from haystack_integrations.components.generators.nvidia import NvidiaGenerator


@pytest.fixture
def mock_local_chat_completion(requests_mock: Mocker) -> None:
    requests_mock.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "choices": [
                {
                    "message": {"content": "Hello!", "role": "system"},
                    "usage": {"prompt_tokens": 3, "total_tokens": 5, "completion_tokens": 9},
                    "finish_reason": "stop",
                    "index": 0,
                },
                {
                    "message": {"content": "How are you?", "role": "system"},
                    "usage": {"prompt_tokens": 3, "total_tokens": 5, "completion_tokens": 9},
                    "finish_reason": "stop",
                    "index": 1,
                },
            ],
            "usage": {
                "prompt_tokens": 3,
                "total_tokens": 5,
                "completion_tokens": 9,
            },
        },
    )


class TestNvidiaGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaGenerator("playground_nv_llama2_rlhf_70b")

        assert generator._api_key == Secret.from_env_var("NVIDIA_API_KEY")
        assert generator._model == "playground_nv_llama2_rlhf_70b"
        assert generator._model_arguments == {}

    def test_init_with_parameters(self):
        generator = NvidiaGenerator(
            api_key=Secret.from_token("fake-api-key"),
            model="playground_nemotron_steerlm_8b",
            model_arguments={
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
                "seed": None,
                "bad": None,
                "stop": None,
            },
        )
        assert generator._api_key == Secret.from_token("fake-api-key")
        assert generator._model == "playground_nemotron_steerlm_8b"
        assert generator._model_arguments == {
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
            "seed": None,
            "bad": None,
            "stop": None,
        }

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        generator = NvidiaGenerator("playground_nemotron_steerlm_8b")
        with pytest.raises(ValueError):
            generator.warm_up()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaGenerator("playground_nemotron_steerlm_8b")
        data = generator.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator",
            "init_parameters": {
                "api_url": "https://integrate.api.nvidia.com/v1",
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "playground_nemotron_steerlm_8b",
                "model_arguments": {},
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaGenerator(
            model="playground_nemotron_steerlm_8b",
            api_url="https://my.url.com",
            model_arguments={
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
                "seed": None,
                "bad": None,
                "stop": None,
            },
        )
        data = generator.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "api_url": "https://my.url.com/v1",
                "model": "playground_nemotron_steerlm_8b",
                "model_arguments": {
                    "temperature": 0.2,
                    "top_p": 0.7,
                    "max_tokens": 1024,
                    "seed": None,
                    "bad": None,
                    "stop": None,
                },
            },
        }

    def test_setting_timeout(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaGenerator(timeout=10.0)
        generator.warm_up()
        assert generator.backend.timeout == 10.0

    def test_setting_timeout_env(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        monkeypatch.setenv("NVIDIA_TIMEOUT", "45")
        generator = NvidiaGenerator()
        generator.warm_up()
        assert generator.backend.timeout == 45.0

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_NIM_GENERATOR_MODEL", None) or not os.environ.get("NVIDIA_NIM_ENDPOINT_URL", None),
        reason="Export an env var called NVIDIA_NIM_GENERATOR_MODEL containing the hosted model name and "
        "NVIDIA_NIM_ENDPOINT_URL containing the local URL to call.",
    )
    @pytest.mark.integration
    def test_run_integration_with_nimbackend(self):
        model = os.environ["NVIDIA_NIM_GENERATOR_MODEL"]
        url = os.environ["NVIDIA_NIM_ENDPOINT_URL"]
        generator = NvidiaGenerator(
            model=model,
            api_url=url,
            api_key=None,
            model_arguments={
                "temperature": 0.2,
            },
        )
        generator.warm_up()
        result = generator.run(prompt="What is the answer?")

        assert result["replies"]
        assert result["meta"]

    @pytest.mark.integration
    @pytest.mark.usefixtures("mock_local_models")
    @pytest.mark.usefixtures("mock_local_chat_completion")
    def test_run_integration_with_default_model_nimbackend(self):
        model = None
        url = "http://localhost:8080/v1"
        generator = NvidiaGenerator(
            model=model,
            api_url=url,
            api_key=None,
            model_arguments={
                "temperature": 0.2,
            },
        )
        with pytest.warns(UserWarning) as record:
            generator.warm_up()
        assert len(record) == 1
        assert "Default model is set as:" in str(record[0].message)
        assert generator._model == "model1"
        assert not generator.is_hosted

        result = generator.run(prompt="What is the answer?")

        assert result["replies"]
        assert result["meta"]

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_integration_with_api_catalog(self):
        generator = NvidiaGenerator(
            model="meta/llama3-8b-instruct",
            api_url="https://integrate.api.nvidia.com/v1",
            api_key=Secret.from_env_var("NVIDIA_API_KEY"),
            model_arguments={
                "temperature": 0.2,
            },
        )
        generator.warm_up()
        result = generator.run(prompt="What is the answer?")

        assert result["replies"]
        assert result["meta"]

    @pytest.mark.usefixtures("mock_local_models")
    def test_local_nim_without_key(self) -> None:
        generator = NvidiaGenerator(
            model="model1",
            api_url="http://localhost:8080",
            api_key=None,
        )
        generator.warm_up()

    def test_hosted_nim_without_key(self):
        generator0 = NvidiaGenerator(
            model="BOGUS",
            api_url="https://integrate.api.nvidia.com/v1",
            api_key=None,
        )
        with pytest.raises(ValueError):
            generator0.warm_up()

        generator1 = NvidiaGenerator(
            model="BOGUS",
            api_key=None,
        )
        with pytest.raises(ValueError):
            generator1.warm_up()

    @pytest.mark.usefixtures("mock_local_models")
    def test_warm_up_falls_back_to_default_model(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaGenerator(api_url="http://localhost:8080/v1")

        with pytest.warns(UserWarning, match="Default model is set as:"):
            generator.warm_up()

        assert generator._model == "model1"
        assert generator.backend.model == "model1"
        assert generator.to_dict()["init_parameters"]["model"] == "model1"

    def test_default_model_raises_when_no_valid_models(self, monkeypatch, requests_mock):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        requests_mock.get(
            "http://localhost:8080/v1/models",
            json={"data": [{"id": "derived-model", "object": "model", "root": "base-model"}]},
        )
        generator = NvidiaGenerator(api_url="http://localhost:8080/v1")

        with pytest.raises(ValueError, match="No locally hosted model was found"):
            generator.warm_up()

    def test_warm_up_is_idempotent(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaGenerator("meta/llama3-8b-instruct")
        generator.warm_up()
        backend = generator.backend
        generator.warm_up()
        assert generator.backend is backend

    def test_available_models_without_backend(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaGenerator("meta/llama3-8b-instruct")
        assert generator.available_models == []

    @pytest.mark.usefixtures("mock_local_models")
    def test_available_models_with_backend(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaGenerator(model="model1", api_url="http://localhost:8080/v1")
        generator.warm_up()
        models = generator.available_models
        assert len(models) == 1
        assert models[0].id == "model1"

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "api_url": "https://my.url.com/v1",
                "model": "meta/llama3-8b-instruct",
                "model_arguments": {"temperature": 0.5},
            },
        }
        generator = NvidiaGenerator.from_dict(data)
        assert generator._model == "meta/llama3-8b-instruct"
        assert generator.api_url == "https://my.url.com/v1"
        assert generator._model_arguments == {"temperature": 0.5}

    def test_run(self, monkeypatch, mock_local_chat_completion):  # noqa: ARG002
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaGenerator(model="model1", api_url="http://localhost:8080/v1")

        result = generator.run(prompt="What is the answer?")

        assert result["replies"] == ["Hello!", "How are you?"]
        assert len(result["meta"]) == 2
