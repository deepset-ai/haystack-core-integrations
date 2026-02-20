import os
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.dspy.chat.chat_generator import DSPyChatGenerator


@pytest.fixture
def mock_dspy_module():
    """
    Mock DSPy LM, configure, and module classes to avoid real API calls.
    """
    with patch("dspy.LM") as mock_lm_class, \
         patch("dspy.configure"), \
         patch("dspy.ChainOfThought") as mock_cot_class, \
         patch("dspy.Predict") as mock_predict_class, \
         patch("dspy.ReAct") as mock_react_class:
        mock_lm = MagicMock()
        mock_lm_class.return_value = mock_lm

        mock_module = MagicMock()
        mock_module.return_value = MagicMock(answer="Hello world!")
        mock_cot_class.return_value = mock_module
        mock_predict_class.return_value = mock_module
        mock_react_class.return_value = mock_module

        yield mock_module


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


class TestDSPyChatGenerator:
    def test_init_default(self, monkeypatch, mock_dspy_module):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = DSPyChatGenerator(signature="question -> answer")
        assert component.model == "openai/gpt-5-mini"
        assert component.signature == "question -> answer"
        assert component.module_type == "ChainOfThought"
        assert component.output_field == "answer"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.input_mapping is None

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            DSPyChatGenerator(signature="question -> answer")

    def test_init_with_parameters(self, mock_dspy_module):
        component = DSPyChatGenerator(
            signature="context, question -> answer",
            model="openai/gpt-4o",
            api_key=Secret.from_token("test-api-key"),
            module_type="Predict",
            output_field="response",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            input_mapping={"context": "context", "question": "question"},
        )
        assert component.model == "openai/gpt-4o"
        assert component.signature == "context, question -> answer"
        assert component.module_type == "Predict"
        assert component.output_field == "response"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.input_mapping == {"context": "context", "question": "question"}

    def test_init_with_invalid_module_type(self, mock_dspy_module):
        with pytest.raises(ValueError, match="Invalid module_type"):
            DSPyChatGenerator(
                signature="question -> answer",
                api_key=Secret.from_token("test-api-key"),
                module_type="InvalidModule",
            )

    def test_to_dict_default(self, monkeypatch, mock_dspy_module):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = DSPyChatGenerator(
            signature="question -> answer",
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.dspy.chat.chat_generator.DSPyChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "signature": "question -> answer",
                "model": "openai/gpt-5-mini",
                "module_type": "ChainOfThought",
                "output_field": "answer",
                "generation_kwargs": {},
                "input_mapping": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch, mock_dspy_module):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = DSPyChatGenerator(
            signature="context, question -> answer",
            model="openai/gpt-4o",
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            module_type="Predict",
            output_field="response",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            input_mapping={"context": "context", "question": "question"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.dspy.chat.chat_generator.DSPyChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "signature": "context, question -> answer",
                "model": "openai/gpt-4o",
                "module_type": "Predict",
                "output_field": "response",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "input_mapping": {"context": "context", "question": "question"},
            },
        }

    def test_from_dict(self, monkeypatch, mock_dspy_module):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.dspy.chat.chat_generator.DSPyChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "signature": "question -> answer",
                "model": "openai/gpt-4o",
                "module_type": "Predict",
                "output_field": "response",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "input_mapping": None,
            },
        }
        component = DSPyChatGenerator.from_dict(data)
        assert component.model == "openai/gpt-4o"
        assert component.signature == "question -> answer"
        assert component.module_type == "Predict"
        assert component.output_field == "response"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert component.input_mapping is None

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.generators.dspy.chat.chat_generator.DSPyChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "signature": "question -> answer",
                "model": "openai/gpt-4o",
                "module_type": "Predict",
                "output_field": "response",
                "generation_kwargs": {},
                "input_mapping": None,
            },
        }
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            DSPyChatGenerator.from_dict(data)

    def test_run(self, chat_messages, mock_dspy_module):
        component = DSPyChatGenerator(
            signature="question -> answer",
            api_key=Secret.from_token("test-api-key"),
        )
        response = component.run(chat_messages)

        # Verify the mock was called
        mock_dspy_module.assert_called_once()

        # Check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert all(isinstance(reply, ChatMessage) for reply in response["replies"])

    def test_run_with_params(self, chat_messages, mock_dspy_module):
        component = DSPyChatGenerator(
            signature="question -> answer",
            api_key=Secret.from_token("test-api-key"),
            generation_kwargs={"max_tokens": 10, "temperature": 0.5},
        )
        response = component.run(chat_messages, generation_kwargs={"temperature": 0.9})

        # Check that the component calls the DSPy module with the correct parameters
        _, kwargs = mock_dspy_module.call_args
        assert kwargs["config"] == {"temperature": 0.9}

        # Check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert all(isinstance(reply, ChatMessage) for reply in response["replies"])

    def test_run_with_multiple_messages(self, mock_dspy_module):
        component = DSPyChatGenerator(
            signature="question -> answer",
            api_key=Secret.from_token("test-api-key"),
        )
        messages = [
            ChatMessage.from_user("Hello"),
            ChatMessage.from_assistant("Hi there!"),
            ChatMessage.from_user("What is the capital of Germany?"),
        ]
        response = component.run(messages=messages)

        # Verify the last user message was used as input
        args, _ = mock_dspy_module.call_args
        # The first positional kwarg should be the question from the last user message
        call_kwargs = mock_dspy_module.call_args.kwargs
        assert call_kwargs.get("question") == "What is the capital of Germany?"

        assert "replies" in response
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], ChatMessage)

    def test_run_with_empty_messages(self, mock_dspy_module):
        component = DSPyChatGenerator(
            signature="question -> answer",
            api_key=Secret.from_token("test-api-key"),
        )
        with pytest.raises(ValueError, match="messages"):
            component.run(messages=[])

    def test_run_with_wrong_model(self, mock_dspy_module):
        mock_dspy_module.side_effect = Exception("Invalid model name")

        generator = DSPyChatGenerator(
            signature="question -> answer",
            api_key=Secret.from_token("test-api-key"),
            model="something-obviously-wrong",
        )

        with pytest.raises(Exception, match="Invalid model name"):
            generator.run(messages=[ChatMessage.from_user("Whatever")])

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = DSPyChatGenerator(signature="question -> answer")
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        metadata = results["meta"][0]
        assert metadata["model"] == "openai/gpt-5-mini"
        assert metadata["module_type"] == "ChainOfThought"

