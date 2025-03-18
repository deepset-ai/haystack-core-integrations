# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from requests_mock import Mocker

from haystack_integrations.components.generators.nvidia.chat.chat_generator import NvidiaChatGenerator


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant."),
        ChatMessage.from_user("What is the answer to life, the universe, and everything?"),
    ]


@pytest.fixture
def mock_backend():
    with patch("haystack_integrations.components.generators.nvidia.chat.chat_generator.NimBackend") as mock:
        backend_instance = MagicMock()
        backend_instance.model = "meta/llama3-70b-instruct"
        backend_instance.models.return_value = [
            MagicMock(id="model1", base_model="model1"),
            MagicMock(id="model2", base_model="model2"),
        ]
        # Mock the generate_chat method to return a sample response
        backend_instance.generate_chat.return_value = ([{"content": "42", "model": "meta/llama3-70b-instruct", "finish_reason": "stop"}], {"model": "meta/llama3-70b-instruct"})
        mock.return_value = backend_instance
        yield mock


@pytest.fixture
def mock_local_chat_completion(requests_mock: Mocker) -> None:
    requests_mock.post(
        "http://localhost:8080/v1/chat/completions",
        json={
            "choices": [
                {
                    "message": {"content": "The answer is 42.", "role": "assistant"},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": 25,
                "total_tokens": 30,
                "completion_tokens": 5,
            },
            "model": "meta/llama3-70b-instruct",
        },
    )


class TestNvidiaChatGenerator:
    def test_init_default(self, monkeypatch):
        """Test default initialization"""
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaChatGenerator("meta/llama3-70b-instruct")

        assert generator._api_key == Secret.from_env_var("NVIDIA_API_KEY")
        assert generator._model == "meta/llama3-70b-instruct"
        assert generator._model_arguments == {}

    def test_init_with_parameters(self):
        """Test initialization with parameters"""
        generator = NvidiaChatGenerator(
            api_key=Secret.from_token("fake-api-key"),
            model="meta/llama3-70b-instruct",
            model_arguments={
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
            },
        )
        assert generator._api_key == Secret.from_token("fake-api-key")
        assert generator._model == "meta/llama3-70b-instruct"
        assert generator._model_arguments == {
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
        }

    def test_init_fail_wo_api_key(self, monkeypatch):
        """Test initialization fails without API key"""
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        generator = NvidiaChatGenerator("meta/llama3-70b-instruct")
        with pytest.raises(ValueError):
            generator.warm_up()

    def test_to_dict(self, monkeypatch):
        """Test serialization to dictionary"""
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaChatGenerator("meta/llama3-70b-instruct")
        data = generator.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.nvidia.chat.chat_generator.NvidiaChatGenerator",
            "init_parameters": {
                "api_url": "https://integrate.api.nvidia.com/v1",
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "meta/llama3-70b-instruct",
                "model_arguments": {},
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        """Test serialization with custom init parameters"""
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaChatGenerator(
            model="meta/llama3-70b-instruct",
            api_url="https://my.url.com",
            model_arguments={
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
            },
        )
        data = generator.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.nvidia.chat.chat_generator.NvidiaChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "api_url": "https://my.url.com/v1",
                "model": "meta/llama3-70b-instruct",
                "model_arguments": {
                    "temperature": 0.2,
                    "top_p": 0.7,
                    "max_tokens": 1024,
                },
            },
        }

    def test_from_dict(self, monkeypatch):
        """Test deserialization from dictionary"""
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.nvidia.chat.chat_generator.NvidiaChatGenerator",
            "init_parameters": {
                "api_url": "https://my.url.com/v1",
                "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
                "model": "meta/llama3-70b-instruct",
                "model_arguments": {
                    "temperature": 0.2,
                    "top_p": 0.7,
                    "max_tokens": 1024,
                },
            },
        }
        generator = NvidiaChatGenerator.from_dict(data)
        assert generator._api_key == Secret.from_env_var("NVIDIA_API_KEY")
        assert generator._model == "meta/llama3-70b-instruct"
        assert generator._model_arguments == {
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
        }
        assert generator.api_url == "https://my.url.com/v1"

    def test_warm_up_with_model(self, mock_backend, monkeypatch):
        """Test warm_up initializes backend with model"""
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaChatGenerator("meta/llama3-70b-instruct")
        generator.warm_up()
        
        mock_backend.assert_called_once()
        call_kwargs = mock_backend.call_args[1]
        assert call_kwargs["model"] == "meta/llama3-70b-instruct"
        assert call_kwargs["model_type"] == "chat"
        assert call_kwargs["client"] == "NvidiaChatGenerator"

    def test_warm_up_without_model_local(self, mock_backend, monkeypatch):
        """Test warm_up sets default model when none provided for local backend"""
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        
        # Override the mock to set model to None to trigger default_model call
        mock_backend.return_value.model = None
        
        generator = NvidiaChatGenerator(model=None, api_url="http://localhost:8080")
        generator.is_hosted = False  # Force local mode
        
        with pytest.warns(UserWarning, match="Default model is set as:"):
            generator.warm_up()
        
        assert generator._model == "model1"  # Should be set to first model in mocked models list

    def test_run(self, mock_backend, monkeypatch, chat_messages):
        """Test run method with regular non-streaming response"""
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaChatGenerator("meta/llama3-70b-instruct")
        generator.warm_up()
        
        result = generator.run(messages=chat_messages)
        
        mock_backend().generate_chat.assert_called_once()
        
        # Verify the output structure
        assert "replies" in result
        assert len(result["replies"]) == 1
        assert result["replies"][0].text == "42"
        assert result["replies"][0].meta["model"] == "meta/llama3-70b-instruct"
        assert result["replies"][0].meta["finish_reason"] == "stop"

    def test_convert_messages_to_nvidia_format(self, monkeypatch):
        """Test conversion of ChatMessages to NVIDIA format"""
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaChatGenerator("meta/llama3-70b-instruct")
        
        messages = [
            ChatMessage.from_system("You are a helpful assistant."),
            ChatMessage.from_user("What is the answer?"),
            ChatMessage.from_assistant("The answer is 42.")
        ]
        
        nvidia_messages = generator._convert_messages_to_nvidia_format(messages)
        
        assert len(nvidia_messages) == 3
        assert nvidia_messages[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert nvidia_messages[1] == {"role": "user", "content": "What is the answer?"}
        assert nvidia_messages[2] == {"role": "assistant", "content": "The answer is 42."}

    def test_convert_nvidia_response_to_chat_message(self, monkeypatch):
        """Test conversion of NVIDIA response to ChatMessage"""
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaChatGenerator("meta/llama3-70b-instruct")
        
        nvidia_response = {
            "content": "The answer is 42.",
            "model": "meta/llama3-70b-instruct",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 25, "completion_tokens": 5, "total_tokens": 30}
        }
        
        chat_message = generator._convert_nvidia_response_to_chat_message(nvidia_response)
        
        assert chat_message.text == "The answer is 42."
        assert chat_message.meta["model"] == "meta/llama3-70b-instruct"
        assert chat_message.meta["finish_reason"] == "stop"
        assert chat_message.meta["usage"] == {
            "prompt_tokens": 25, 
            "completion_tokens": 5, 
            "total_tokens": 30
        }

    def test_error_if_warm_up_not_called(self, monkeypatch, chat_messages):
        """Test error is raised if warm_up not called"""
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaChatGenerator("meta/llama3-70b-instruct")
        
        with pytest.raises(RuntimeError, match="The chat model has not been loaded"):
            generator.run(messages=chat_messages)

    def test_setting_timeout(self, monkeypatch, mock_backend):
        """Test timeout setting"""
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        generator = NvidiaChatGenerator(timeout=10.0)
        generator.warm_up()
        
        # Check that timeout is passed to backend
        assert mock_backend.call_args[1]["timeout"] == 10.0

    def test_setting_timeout_env(self, monkeypatch, mock_backend):
        """Test timeout from environment variable"""
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        monkeypatch.setenv("NVIDIA_TIMEOUT", "45")
        generator = NvidiaChatGenerator()
        generator.warm_up()
        
        # Check that timeout is passed to backend
        assert mock_backend.call_args[1]["timeout"] == 45.0

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    def test_run_integration_with_api_catalog(self):
        """Integration test with NVIDIA API Catalog"""
        generator = NvidiaChatGenerator(
            model="meta/llama3-70b-instruct",
            api_url="https://integrate.api.nvidia.com/v1",
            api_key=Secret.from_env_var("NVIDIA_API_KEY"),
            model_arguments={
                "temperature": 0.2,
                "max_tokens": 50,
            },
        )
        generator.warm_up()
        
        messages = [
            ChatMessage.from_system("You are a helpful assistant. Keep your answers brief."),
            ChatMessage.from_user("What is the answer to life, the universe, and everything?")
        ]
        
        result = generator.run(messages=messages)
        
        assert "replies" in result
        assert len(result["replies"]) == 1
        assert isinstance(result["replies"][0], ChatMessage)
        assert len(result["replies"][0].text) > 0
        assert result["replies"][0].meta["model"] is not None
        assert result["replies"][0].meta["finish_reason"] is not None
