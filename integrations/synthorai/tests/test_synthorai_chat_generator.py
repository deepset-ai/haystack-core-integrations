import os

import pytest
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.synthorai.chat.chat_generator import (
    SynthoraiChatGenerator,
)


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


class TestSynthoraiChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("SYNTHORAI_API_KEY", "test-api-key")
        component = SynthoraiChatGenerator()

        assert component.api_key == Secret.from_env_var("SYNTHORAI_API_KEY")
        assert component.model == "claude-opus-5"
        assert component.api_base_url == "https://synthorai.io/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fails_without_api_key(self, monkeypatch):
        monkeypatch.delenv("SYNTHORAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            # haystack-ai 2.x raises at init; haystack-ai >= 3.0 raises when the client
            # is created in warm_up
            component = SynthoraiChatGenerator()
            component.warm_up()

    def test_init_with_parameters(self, monkeypatch):
        monkeypatch.setenv("SYNTHORAI_API_KEY", "test-api-key")
        component = SynthoraiChatGenerator(
            model="deepseek-v4-pro",
            generation_kwargs={"max_tokens": 100, "temperature": 0.5},
            timeout=30,
            max_retries=2,
        )

        assert component.model == "deepseek-v4-pro"
        assert component.generation_kwargs == {"max_tokens": 100, "temperature": 0.5}
        assert component.timeout == 30
        assert component.max_retries == 2
        # the endpoint is not configurable: every model is served from one host
        assert component.api_base_url == "https://synthorai.io/v1"

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("SYNTHORAI_API_KEY", "test-api-key")
        component = SynthoraiChatGenerator()
        data = component.to_dict()

        assert data["type"] == (
            "haystack_integrations.components.generators.synthorai.chat.chat_generator.SynthoraiChatGenerator"
        )
        init_params = data["init_parameters"]
        assert init_params["api_key"] == {
            "env_vars": ["SYNTHORAI_API_KEY"],
            "strict": True,
            "type": "env_var",
        }
        assert init_params["model"] == "claude-opus-5"
        assert init_params["streaming_callback"] is None
        # api_base_url is deliberately not serialised - it is not an init parameter
        assert "api_base_url" not in init_params

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("SYNTHORAI_API_KEY", "test-api-key")
        data = {
            "type": (
                "haystack_integrations.components.generators.synthorai.chat.chat_generator.SynthoraiChatGenerator"
            ),
            "init_parameters": {
                "api_key": {"env_vars": ["SYNTHORAI_API_KEY"], "strict": True, "type": "env_var"},
                "model": "glm-5.2",
                "streaming_callback": None,
                "generation_kwargs": {"temperature": 0.2},
            },
        }
        component = SynthoraiChatGenerator.from_dict(data)

        assert component.model == "glm-5.2"
        assert component.generation_kwargs == {"temperature": 0.2}
        assert component.api_base_url == "https://synthorai.io/v1"

    @pytest.mark.skipif(
        not os.environ.get("SYNTHORAI_API_KEY", None),
        reason="Export an env var called SYNTHORAI_API_KEY containing the Synthorai API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self, chat_messages):
        component = SynthoraiChatGenerator()
        results = component.run(chat_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert "Paris" in message.text
        assert "claude-opus-5" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"
