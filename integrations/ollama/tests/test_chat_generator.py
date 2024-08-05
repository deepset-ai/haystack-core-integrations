from typing import List
from unittest.mock import Mock

import pytest
from haystack.dataclasses import ChatMessage, ChatRole
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from requests import HTTPError, Response


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
        assert component.url == "http://localhost:11434/api/chat"
        assert component.generation_kwargs == {}
        assert component.template is None
        assert component.timeout == 120

    def test_init(self):
        component = OllamaChatGenerator(
            model="llama2",
            url="http://my-custom-endpoint:11434/api/chat",
            generation_kwargs={"temperature": 0.5},
            timeout=5,
        )

        assert component.model == "llama2"
        assert component.url == "http://my-custom-endpoint:11434/api/chat"
        assert component.generation_kwargs == {"temperature": 0.5}
        assert component.template is None
        assert component.timeout == 5

    def test_create_json_payload(self, chat_messages):
        observed = OllamaChatGenerator(model="some_model")._create_json_payload(
            chat_messages, False, {"temperature": 0.1}
        )
        expected = {
            "messages": [
                {"role": "user", "content": "Tell me about why Super Mario is the greatest superhero"},
                {"role": "assistant", "content": "Super Mario has prevented Bowser from destroying the world"},
            ],
            "model": "some_model",
            "stream": False,
            "template": None,
            "options": {"temperature": 0.1},
        }

        assert observed == expected

    def test_build_message_from_ollama_response(self):
        model = "some_model"

        mock_ollama_response = Mock(Response)
        mock_ollama_response.json.return_value = {
            "model": model,
            "created_at": "2023-12-12T14:13:43.416799Z",
            "message": {"role": "assistant", "content": "Hello! How are you today?"},
            "done": True,
            "total_duration": 5191566416,
            "load_duration": 2154458,
            "prompt_eval_count": 26,
            "prompt_eval_duration": 383809000,
            "eval_count": 298,
            "eval_duration": 4799921000,
        }

        observed = OllamaChatGenerator(model=model)._build_message_from_ollama_response(mock_ollama_response)

        assert observed.role == "assistant"
        assert observed.content == "Hello! How are you today?"

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
            assert answer in response["replies"][0].content

    @pytest.mark.integration
    def test_run_with_chat_history(self):
        chat_generator = OllamaChatGenerator()

        chat_history = [
            {"role": "user", "content": "What is the largest city in the United Kingdom by population?"},
            {"role": "assistant", "content": "London is the largest city in the United Kingdom by population"},
            {"role": "user", "content": "And what is the second largest?"},
        ]

        chat_messages = [
            ChatMessage(role=ChatRole(message["role"]), content=message["content"], name=None)
            for message in chat_history
        ]
        response = chat_generator.run(chat_messages)

        assert isinstance(response, dict)
        assert isinstance(response["replies"], list)
        assert "Manchester" in response["replies"][-1].content or "Glasgow" in response["replies"][-1].content

    @pytest.mark.integration
    def test_run_model_unavailable(self):
        component = OllamaChatGenerator(model="Alistair_and_Stefano_are_great")

        with pytest.raises(HTTPError):
            message = ChatMessage.from_user(
                "Based on your infinite wisdom, can you tell me why Alistair and Stefano are so great?"
            )
            component.run([message])

    @pytest.mark.integration
    def test_run_with_streaming(self):
        streaming_callback = Mock()
        chat_generator = OllamaChatGenerator(streaming_callback=streaming_callback)

        chat_history = [
            {"role": "user", "content": "What is the largest city in the United Kingdom by population?"},
            {"role": "assistant", "content": "London is the largest city in the United Kingdom by population"},
            {"role": "user", "content": "And what is the second largest?"},
        ]

        chat_messages = [
            ChatMessage(role=ChatRole(message["role"]), content=message["content"], name=None)
            for message in chat_history
        ]

        response = chat_generator.run(chat_messages)

        streaming_callback.assert_called()

        assert isinstance(response, dict)
        assert isinstance(response["replies"], list)
        assert "Manchester" in response["replies"][-1].content or "Glasgow" in response["replies"][-1].content
