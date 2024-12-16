from typing import List
from unittest.mock import Mock

import pytest
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from ollama._types import ChatResponse, ResponseError

from haystack_integrations.components.generators.ollama import OllamaChatGenerator


@pytest.fixture
def chat_messages() -> List[ChatMessage]:
    return [
        ChatMessage.from_user("Tell me about why Super Mario is the greatest superhero"),
        ChatMessage.from_assistant(
            "Super Mario has prevented Bowser from destroying the world", {"something": "something"}
        ),
    ]


class TestOllamaChatGenerator:
    def test_init_default(self):
        component = OllamaChatGenerator()
        assert component.model == "orca-mini"
        assert component.url == "http://localhost:11434"
        assert component.generation_kwargs == {}
        assert component.timeout == 120
        assert component.keep_alive is None

    def test_init(self):
        component = OllamaChatGenerator(
            model="llama2",
            url="http://my-custom-endpoint:11434",
            generation_kwargs={"temperature": 0.5},
            keep_alive="10m",
            timeout=5,
        )

        assert component.model == "llama2"
        assert component.url == "http://my-custom-endpoint:11434"
        assert component.generation_kwargs == {"temperature": 0.5}
        assert component.timeout == 5
        assert component.keep_alive == "10m"

    def test_to_dict(self):
        component = OllamaChatGenerator(
            model="llama2",
            streaming_callback=print_streaming_chunk,
            url="custom_url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            keep_alive="5m",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.ollama.chat.chat_generator.OllamaChatGenerator",
            "init_parameters": {
                "timeout": 120,
                "model": "llama2",
                "keep_alive": "5m",
                "url": "custom_url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.generators.ollama.chat.chat_generator.OllamaChatGenerator",
            "init_parameters": {
                "timeout": 120,
                "model": "llama2",
                "url": "custom_url",
                "keep_alive": "5m",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        component = OllamaChatGenerator.from_dict(data)
        assert component.model == "llama2"
        assert component.streaming_callback is print_streaming_chunk
        assert component.url == "custom_url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.keep_alive == "5m"

    def test_build_message_from_ollama_response(self):
        model = "some_model"

        ollama_response = ChatResponse(
            model=model,
            created_at="2023-12-12T14:13:43.416799Z",
            message={"role": "assistant", "content": "Hello! How are you today?"},
            done=True,
            total_duration=5191566416,
            load_duration=2154458,
            prompt_eval_count=26,
            prompt_eval_duration=383809000,
            eval_count=298,
            eval_duration=4799921000,
        )

        observed = OllamaChatGenerator(model=model)._build_message_from_ollama_response(ollama_response)

        assert observed.role == "assistant"
        assert observed.text == "Hello! How are you today?"

    @pytest.mark.integration
    def test_run(self):
        chat_generator = OllamaChatGenerator()

        user_questions_and_assistant_answers = [
            ("What's the capital of France?", "Paris"),
            ("What is the capital of Canada?", "Ottawa"),
            ("What is the capital of Ghana?", "Accra"),
        ]

        for question, answer in user_questions_and_assistant_answers:
            message = ChatMessage.from_user(question)

            response = chat_generator.run([message])

            assert isinstance(response, dict)
            assert isinstance(response["replies"], list)
            assert answer in response["replies"][0].text

    @pytest.mark.integration
    def test_run_with_chat_history(self):
        chat_generator = OllamaChatGenerator()

        chat_history = [
            ChatMessage.from_user("What is the largest city in the United Kingdom by population?"),
            ChatMessage.from_assistant("London is the largest city in the United Kingdom by population"),
            ChatMessage.from_user("And what is the second largest?"),
        ]

        response = chat_generator.run(chat_history)

        assert isinstance(response, dict)
        assert isinstance(response["replies"], list)
        assert "Manchester" in response["replies"][-1].text or "Glasgow" in response["replies"][-1].text

    @pytest.mark.integration
    def test_run_model_unavailable(self):
        component = OllamaChatGenerator(model="Alistair_and_Stefano_are_great")

        with pytest.raises(ResponseError):
            message = ChatMessage.from_user(
                "Based on your infinite wisdom, can you tell me why Alistair and Stefano are so great?"
            )
            component.run([message])

    @pytest.mark.integration
    def test_run_with_streaming(self):
        streaming_callback = Mock()
        chat_generator = OllamaChatGenerator(streaming_callback=streaming_callback)

        chat_history = [
            ChatMessage.from_user("What is the largest city in the United Kingdom by population?"),
            ChatMessage.from_assistant("London is the largest city in the United Kingdom by population"),
            ChatMessage.from_user("And what is the second largest?"),
        ]

        response = chat_generator.run(chat_history)

        streaming_callback.assert_called()

        assert isinstance(response, dict)
        assert isinstance(response["replies"], list)
        assert "Manchester" in response["replies"][-1].text or "Glasgow" in response["replies"][-1].text
