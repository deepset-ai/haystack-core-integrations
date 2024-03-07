# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import patch

import pytest
from haystack.utils import Secret
from haystack_integrations.components.generators.nvidia import NvidiaGenerator
from haystack_integrations.components.generators.nvidia.models import NvidiaGeneratorModel


class TestNvidiaGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaGenerator(NvidiaGeneratorModel.NV_LLAMA2_RLHF_70B)

        assert generator._api_key == Secret.from_env_var("NVIDIA_API_KEY")
        assert generator._model == NvidiaGeneratorModel.NV_LLAMA2_RLHF_70B
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
        assert generator._model == NvidiaGeneratorModel.NEMOTRON_STEERLM_8B
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
        with pytest.raises(ValueError):
            NvidiaGenerator("playground_nemotron_steerlm_8b")

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaGenerator(NvidiaGeneratorModel.NEMOTRON_STEERLM_8B)
        data = generator.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.nvidia.generator.NvidiaGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "playground_nemotron_steerlm_8b",
                "model_arguments": {},
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaGenerator(
            model=NvidiaGeneratorModel.NEMOTRON_STEERLM_8B,
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

    @patch("haystack_integrations.components.generators.nvidia.generator.NvidiaCloudFunctionsClient")
    def test_run(self, mock_client):
        generator = NvidiaGenerator(
            model=NvidiaGeneratorModel.NEMOTRON_STEERLM_8B,
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
        mock_client.get_model_nvcf_id.return_value = "some_id"
        generator._client = mock_client
        generator.warm_up()
        mock_client.get_model_nvcf_id.assert_called_once_with("playground_nemotron_steerlm_8b")

        mock_client.query_function.return_value = {
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
                    "total_tokens": 21,
                    "prompt_tokens": 19,
                    "completion_tokens": 2,
                },
            ],
        }

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the Nvidia API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_integration(self):
        generator = NvidiaGenerator(
            model=NvidiaGeneratorModel.NV_LLAMA2_RLHF_70B,
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
