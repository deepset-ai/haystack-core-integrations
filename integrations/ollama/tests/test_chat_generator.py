from unittest.mock import Mock

import pytest
from haystack.dataclasses import ChatMessage
from requests import Response

from ollama_haystack import OllamaChatGenerator


@pytest.fixture
def user_chat_message():
    msg = "Tell me about why Super Mario is the greatest superhero"
    return ChatMessage.from_user(msg)


@pytest.fixture
def assistant_chat_message():
    msg = "Super Mario has prevented Bowser from destroying the world"
    metadata = {"something": "something"}
    return ChatMessage.from_assistant(msg, metadata)


@pytest.fixture
def list_of_chat_messages(user_chat_message, assistant_chat_message):
    return [user_chat_message, assistant_chat_message]


class TestOllamaChatGenerator:
    def test_init_default(self):
        component = OllamaChatGenerator()
        assert component.model == "orca-mini"
        assert component.url == "http://localhost:11434/api/chat"
        assert component.generation_kwargs == {}
        assert component.template is None
        assert component.timeout == 30
        assert component.streaming_callback is None

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

    def test_user_message_to_dict(self, user_chat_message):
        observed = OllamaChatGenerator()._message_to_dict(user_chat_message)
        expected = {"role": "user", "content": "Tell me about why Super Mario is the greatest superhero"}

        assert observed == expected

    def test_assistant_message_to_dict(self, assistant_chat_message):
        observed = OllamaChatGenerator()._message_to_dict(assistant_chat_message)
        expected = {"role": "assistant", "content": "Super Mario has prevented Bowser from destroying the world"}

        assert observed == expected

    def test__chat_history_to_dict(self, list_of_chat_messages):
        observed = OllamaChatGenerator()._chat_history_to_dict(list_of_chat_messages)
        expected = [
            {"role": "user", "content": "Tell me about why Super Mario is the greatest superhero"},
            {"role": "assistant", "content": "Super Mario has prevented Bowser from destroying the world"},
        ]

        assert observed == expected

    def test__create_json_payload(self, list_of_chat_messages):
        observed = OllamaChatGenerator(model="some_model")._create_json_payload(
            list_of_chat_messages, {"temperature": 0.1}
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

    def test__build_message(self):
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

        observed = OllamaChatGenerator(model=model)._build_message(mock_ollama_response)

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
