import os
from unittest.mock import MagicMock, patch

import dspy
import pytest
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.dspy.chat.chat_generator import (
    VALID_MODULE_TYPES,
    DSPyChatGenerator,
    configure_dspy_lm,
    get_dspy_module_class,
)


@pytest.fixture
def mock_dspy_module():
    """
    Mock DSPy LM, configure, and module classes to avoid real API calls.
    """
    with (
        patch("haystack_integrations.components.generators.dspy.chat.chat_generator.dspy.LM") as mock_lm_class,
        patch("haystack_integrations.components.generators.dspy.chat.chat_generator.dspy.configure"),
        patch(
            "haystack_integrations.components.generators.dspy.chat.chat_generator.dspy.ChainOfThought"
        ) as mock_cot_class,
        patch(
            "haystack_integrations.components.generators.dspy.chat.chat_generator.dspy.Predict"
        ) as mock_predict_class,
        patch("haystack_integrations.components.generators.dspy.chat.chat_generator.dspy.ReAct") as mock_react_class,
    ):
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


@pytest.fixture
def sample_qa_signature():
    class QASignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    return QASignature


class TestValidModuleTypes:
    def test_contains_expected_types(self):
        assert VALID_MODULE_TYPES == {"Predict", "ChainOfThought", "ReAct"}


class TestGetDspyModuleClass:
    def test_predict(self):
        assert get_dspy_module_class("Predict") is dspy.Predict

    def test_chain_of_thought(self):
        assert get_dspy_module_class("ChainOfThought") is dspy.ChainOfThought

    def test_react(self):
        assert get_dspy_module_class("ReAct") is dspy.ReAct

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid module_type 'Unknown'"):
            get_dspy_module_class("Unknown")

    def test_invalid_type_lists_valid_options(self):
        with pytest.raises(ValueError, match="ChainOfThought"):
            get_dspy_module_class("BadType")


class TestConfigureDspyLm:
    @patch("dspy.configure")
    @patch("dspy.LM")
    def test_creates_lm_and_configures(self, mock_lm_class, mock_configure):
        mock_lm = MagicMock()
        mock_lm_class.return_value = mock_lm

        result = configure_dspy_lm(model="openai/gpt-5-mini", api_key="test-key")

        mock_lm_class.assert_called_once_with(model="openai/gpt-5-mini", api_key="test-key")
        mock_configure.assert_called_once_with(lm=mock_lm)
        assert result is mock_lm

    @patch("dspy.configure")
    @patch("dspy.LM")
    def test_passes_extra_kwargs(self, mock_lm_class, mock_configure):
        mock_lm = MagicMock()
        mock_lm_class.return_value = mock_lm

        configure_dspy_lm(model="openai/gpt-5-mini", api_key="test-key", temperature=0.7, max_tokens=100)

        mock_lm_class.assert_called_once_with(
            model="openai/gpt-5-mini", api_key="test-key", temperature=0.7, max_tokens=100
        )


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

    def test_init_with_signature_class(self, mock_dspy_module, sample_qa_signature):
        component = DSPyChatGenerator(
            signature=sample_qa_signature,
            api_key=Secret.from_token("test-api-key"),
        )
        assert component.signature is sample_qa_signature

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

        mock_dspy_module.assert_called_once()

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert all(isinstance(reply, ChatMessage) for reply in response["replies"])

    def test_run_with_generation_kwargs(self, chat_messages, mock_dspy_module):
        component = DSPyChatGenerator(
            signature="question -> answer",
            api_key=Secret.from_token("test-api-key"),
            generation_kwargs={"max_tokens": 10, "temperature": 0.5},
        )
        response = component.run(chat_messages, generation_kwargs={"temperature": 0.9})

        _, kwargs = mock_dspy_module.call_args
        assert kwargs["config"] == {"temperature": 0.9}

        assert isinstance(response, dict)
        assert "replies" in response
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

    def test_run_with_custom_output_field(self, mock_dspy_module):
        mock_dspy_module.return_value = MagicMock(summary="This is a summary")
        component = DSPyChatGenerator(
            signature="text -> summary",
            api_key=Secret.from_token("test-api-key"),
            output_field="summary",
        )
        messages = [ChatMessage.from_user("Summarize this text")]
        response = component.run(messages=messages)

        assert response["replies"][0].text == "This is a summary"

    def test_run_with_input_mapping(self, mock_dspy_module):
        component = DSPyChatGenerator(
            signature="context, question -> answer",
            api_key=Secret.from_token("test-api-key"),
            input_mapping={"context": "context", "question": "question"},
        )
        messages = [ChatMessage.from_user("What is ML?")]
        response = component.run(messages=messages, context="Machine learning is a subset of AI.")

        call_kwargs = mock_dspy_module.call_args.kwargs
        assert call_kwargs.get("context") == "Machine learning is a subset of AI."
        assert call_kwargs.get("question") == "What is ML?"

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
