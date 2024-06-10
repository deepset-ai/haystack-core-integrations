# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from haystack.utils import Secret
from haystack_integrations.components.generators.nvidia import NvidiaGenerator


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
                "api_url": "https://my.url.com",
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

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_NIM_GENERATOR_MODEL", None) or not os.environ.get("NVIDIA_NIM_ENDPOINT_URL", None),
        reason="Export an env var called NVIDIA_NIM_GENERATOR_MODEL containing the hosted model name and "
        "NVIDIA_NIM_ENDPOINT_URL containing the local URL to call.",
    )
    @pytest.mark.integration
    def test_run_integration_with_nim_backend(self):
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

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_integration_with_api_catalog(self):
        generator = NvidiaGenerator(
            model="meta/llama3-70b-instruct",
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

    def test_local_nim_without_key(self) -> None:
        generator = NvidiaGenerator(
            model="BOGUS",
            api_url="http://localhost:8000",
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
