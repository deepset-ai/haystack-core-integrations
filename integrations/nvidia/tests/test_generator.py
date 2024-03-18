# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import Mock, patch

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
                "api_url": None,
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

    @patch("haystack_integrations.components.generators.nvidia._nvcf_backend.NvidiaCloudFunctionsClient")
    def test_run(self, mock_client_class):
        generator = NvidiaGenerator(
            model="playground_nemotron_steerlm_8b",
            api_key=Secret.from_token("fake-api-key"),
            model_arguments={
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
                "seed": None,
                "bad": None,
                "stop": None,
            },
        )
        mock_client = Mock(
            get_model_nvcf_id=Mock(return_value="some_id"),
            query_function=Mock(
                return_value={
                    "id": "some_id",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"content": "42", "role": "assistant"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"total_tokens": 21, "prompt_tokens": 19, "completion_tokens": 2},
                }
            ),
        )
        mock_client_class.return_value = mock_client
        generator.warm_up()

        result = generator.run(prompt="What is the answer?")
        mock_client.query_function.assert_called_once_with(
            "some_id",
            {
                "messages": [
                    {"content": "What is the answer?", "role": "user"},
                ],
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
                "seed": None,
                "bad": None,
                "stop": None,
            },
        )
        assert result == {
            "replies": ["42"],
            "meta": [
                {
                    "finish_reason": "stop",
                    "role": "assistant",
                    "usage": {
                        "total_tokens": 21,
                        "prompt_tokens": 19,
                        "completion_tokens": 2,
                    },
                },
            ],
        }

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the Nvidia API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_integration_with_nvcf_backend(self):
        generator = NvidiaGenerator(
            model="playground_nv_llama2_rlhf_70b",
            model_arguments={
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
                "seed": None,
                "bad": None,
                "stop": None,
            },
        )
        generator.warm_up()
        result = generator.run(prompt="What is the answer?")

        assert result["replies"]
        assert result["meta"]

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
                "top_p": 0.7,
                "max_tokens": 1024,
            },
        )
        generator.warm_up()
        result = generator.run(prompt="What is the answer?")

        assert result["replies"]
        assert result["meta"]
