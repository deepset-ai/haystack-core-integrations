import os
from unittest.mock import MagicMock, patch

import dspy
import pytest
from haystack.dataclasses import ChatMessage

from haystack_integrations.components.generators.dspy.chat.chat_generator import (
    VALID_MODULE_TYPES,
    DSPySignatureChatGenerator,
    _create_dspy_lm,
    _get_dspy_module_class,
)


@pytest.fixture
def mock_dspy_module():
    """
    Mock DSPy LM, configure, and module classes to avoid real API calls.
    """
    with (
        patch("haystack_integrations.components.generators.dspy.chat.chat_generator.dspy.LM") as mock_lm_class,
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
        assert _get_dspy_module_class("Predict") is dspy.Predict

    def test_chain_of_thought(self):
        assert _get_dspy_module_class("ChainOfThought") is dspy.ChainOfThought

    def test_react(self):
        assert _get_dspy_module_class("ReAct") is dspy.ReAct

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid module_type 'Unknown'"):
            _get_dspy_module_class("Unknown")

    def test_invalid_type_lists_valid_options(self):
        with pytest.raises(ValueError, match="ChainOfThought"):
            _get_dspy_module_class("BadType")


class TestCreateDspyLm:
    @patch("dspy.LM")
    def test_creates_lm(self, mock_lm_class):
        mock_lm = MagicMock()
        mock_lm_class.return_value = mock_lm

        result = _create_dspy_lm(model="openai/gpt-5-mini")

        mock_lm_class.assert_called_once_with(model="openai/gpt-5-mini")
        assert result is mock_lm

    @patch("dspy.LM")
    def test_passes_extra_kwargs(self, mock_lm_class):
        mock_lm = MagicMock()
        mock_lm_class.return_value = mock_lm

        _create_dspy_lm(model="openai/gpt-5-mini", temperature=0.7, max_tokens=100)

        mock_lm_class.assert_called_once_with(model="openai/gpt-5-mini", temperature=0.7, max_tokens=100)

    @patch("dspy.LM")
    def test_passes_api_base(self, mock_lm_class):
        mock_lm = MagicMock()
        mock_lm_class.return_value = mock_lm

        _create_dspy_lm(model="openai/local-model", api_base="http://localhost:8000")

        mock_lm_class.assert_called_once_with(model="openai/local-model", api_base="http://localhost:8000")

    @patch("dspy.LM")
    def test_omits_api_base_when_none(self, mock_lm_class):
        mock_lm = MagicMock()
        mock_lm_class.return_value = mock_lm

        _create_dspy_lm(model="openai/gpt-5-mini")

        mock_lm_class.assert_called_once_with(model="openai/gpt-5-mini")


class TestDSPySignatureChatGenerator:
    def test_init_default(self, mock_dspy_module):
        component = DSPySignatureChatGenerator(signature="question -> answer")
        assert component.model == "openai/gpt-5-mini"
        assert component.signature == "question -> answer"
        assert component.module_type == "ChainOfThought"
        assert component.output_field == "answer"
        assert not component.generation_kwargs
        assert component.input_mapping is None
        assert component.api_base is None
        assert not component.module_kwargs

    def test_init_with_parameters(self, mock_dspy_module):
        component = DSPySignatureChatGenerator(
            signature="context, question -> answer",
            model="openai/gpt-4o",
            api_base="http://localhost:8000",
            module_type="Predict",
            output_field="response",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            module_kwargs={"some_param": "value"},
            input_mapping={"context": "context", "question": "question"},
        )
        assert component.model == "openai/gpt-4o"
        assert component.signature == "context, question -> answer"
        assert component.module_type == "Predict"
        assert component.output_field == "response"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.module_kwargs == {"some_param": "value"}
        assert component.input_mapping == {"context": "context", "question": "question"}
        assert component.api_base == "http://localhost:8000"

    def test_init_with_invalid_module_type(self, mock_dspy_module):
        with pytest.raises(ValueError, match="Invalid module_type"):
            DSPySignatureChatGenerator(
                signature="question -> answer",
                module_type="InvalidModule",
            )

    def test_init_with_signature_class(self, mock_dspy_module, sample_qa_signature):
        component = DSPySignatureChatGenerator(
            signature=sample_qa_signature,
        )
        assert component.signature is sample_qa_signature

    def test_init_with_module_kwargs(self, mock_dspy_module):
        """Test that module_kwargs are passed to the DSPy module constructor."""
        tools = [MagicMock(), MagicMock()]
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
            module_type="ReAct",
            module_kwargs={"tools": tools},
        )
        assert component.module_kwargs == {"tools": tools}

    def test_init_with_api_base(self, mock_dspy_module):
        """Test initialization with api_base for local models."""
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
            api_base="http://localhost:11434/v1",
        )
        assert component.api_base == "http://localhost:11434/v1"

    def test_to_dict_default(self, mock_dspy_module):
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.dspy.chat.chat_generator.DSPySignatureChatGenerator",
            "init_parameters": {
                "signature": {"type": "str", "value": "question -> answer"},
                "model": "openai/gpt-5-mini",
                "api_base": None,
                "module_type": "ChainOfThought",
                "output_field": "answer",
                "generation_kwargs": {},
                "module_kwargs": {},
                "input_mapping": None,
            },
        }

    def test_to_dict_with_parameters(self, mock_dspy_module):
        component = DSPySignatureChatGenerator(
            signature="context, question -> answer",
            model="openai/gpt-4o",
            api_base="http://localhost:8000",
            module_type="Predict",
            output_field="response",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            input_mapping={"context": "context", "question": "question"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.dspy.chat.chat_generator.DSPySignatureChatGenerator",
            "init_parameters": {
                "signature": {"type": "str", "value": "context, question -> answer"},
                "model": "openai/gpt-4o",
                "api_base": "http://localhost:8000",
                "module_type": "Predict",
                "output_field": "response",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "module_kwargs": {},
                "input_mapping": {"context": "context", "question": "question"},
            },
        }

    def test_to_dict_with_signature_class(self, mock_dspy_module, sample_qa_signature):
        """Test that signature classes are serialized as fully qualified class paths."""
        component = DSPySignatureChatGenerator(
            signature=sample_qa_signature,
        )
        data = component.to_dict()
        signature_value = data["init_parameters"]["signature"]
        assert signature_value["type"] == "class"
        assert "QASignature" in signature_value["value"]
        assert "." in signature_value["value"]

    def test_from_dict(self, mock_dspy_module):
        data = {
            "type": "haystack_integrations.components.generators.dspy.chat.chat_generator.DSPySignatureChatGenerator",
            "init_parameters": {
                "signature": {"type": "str", "value": "question -> answer"},
                "model": "openai/gpt-4o",
                "api_base": None,
                "module_type": "Predict",
                "output_field": "response",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "module_kwargs": {},
                "input_mapping": None,
            },
        }
        component = DSPySignatureChatGenerator.from_dict(data)
        assert component.model == "openai/gpt-4o"
        assert component.signature == "question -> answer"
        assert component.module_type == "Predict"
        assert component.output_field == "response"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.input_mapping is None
        assert component.api_base is None

    def test_from_dict_with_api_base(self, mock_dspy_module):
        """Test deserialization with api_base."""
        data = {
            "type": "haystack_integrations.components.generators.dspy.chat.chat_generator.DSPySignatureChatGenerator",
            "init_parameters": {
                "signature": {"type": "str", "value": "question -> answer"},
                "model": "openai/local-model",
                "api_base": "http://localhost:8000",
                "module_type": "Predict",
                "output_field": "answer",
                "generation_kwargs": {},
                "module_kwargs": {},
                "input_mapping": None,
            },
        }
        component = DSPySignatureChatGenerator.from_dict(data)
        assert component.api_base == "http://localhost:8000"

    def test_from_dict_resolves_signature_class_path(self, mock_dspy_module):
        """Test that from_dict resolves a dotted signature class path."""
        data = {
            "type": "haystack_integrations.components.generators.dspy.chat.chat_generator.DSPySignatureChatGenerator",
            "init_parameters": {
                "signature": {"type": "class", "value": "dspy.Signature"},
                "model": "openai/gpt-5-mini",
                "module_type": "Predict",
                "output_field": "answer",
                "generation_kwargs": {},
                "module_kwargs": {},
                "input_mapping": None,
            },
        }
        component = DSPySignatureChatGenerator.from_dict(data)
        assert component.signature is dspy.Signature

    def test_from_dict_with_unknown_signature_type(self, mock_dspy_module):
        """Test that from_dict raises an error for unknown signature types."""
        data = {
            "type": "haystack_integrations.components.generators.dspy.chat.chat_generator.DSPySignatureChatGenerator",
            "init_parameters": {
                "signature": {"type": "unknown", "value": "question -> answer"},
                "model": "openai/gpt-5-mini",
                "module_type": "Predict",
                "output_field": "answer",
                "generation_kwargs": {},
                "module_kwargs": {},
                "input_mapping": None,
            },
        }
        with pytest.raises(ValueError, match="Unknown signature type 'unknown'"):
            DSPySignatureChatGenerator.from_dict(data)

    def test_run(self, chat_messages, mock_dspy_module):
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
        )
        response = component.run(chat_messages)

        mock_dspy_module.assert_called_once()

        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert all(isinstance(reply, ChatMessage) for reply in response["replies"])

    def test_run_always_passes_config(self, chat_messages, mock_dspy_module):
        """Test that config is always passed (even as empty dict) - simplified call."""
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
        )
        component.run(chat_messages)

        _, kwargs = mock_dspy_module.call_args
        assert kwargs["config"] == {}

    def test_run_with_generation_kwargs(self, chat_messages, mock_dspy_module):
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
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
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
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
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
        )
        with pytest.raises(ValueError, match="messages"):
            component.run(messages=[])

    def test_run_with_no_user_message(self, mock_dspy_module):
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
        )
        messages = [ChatMessage.from_assistant("I'm an assistant")]
        with pytest.raises(ValueError, match="No user message found"):
            component.run(messages=messages)

    def test_run_with_empty_user_message(self, mock_dspy_module):
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
        )
        messages = [ChatMessage.from_user("")]
        with pytest.raises(ValueError, match="no text content"):
            component.run(messages=messages)

    def test_run_with_wrong_output_field(self, mock_dspy_module):
        prediction = MagicMock(spec=["answer", "keys"])
        prediction.keys.return_value = ["answer"]
        mock_dspy_module.return_value = prediction
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
            output_field="nonexistent",
        )
        messages = [ChatMessage.from_user("Hello")]
        with pytest.raises(ValueError, match="Output field 'nonexistent' not found"):
            component.run(messages=messages)

    def test_run_with_custom_output_field(self, mock_dspy_module):
        mock_dspy_module.return_value = MagicMock(summary="This is a summary")
        component = DSPySignatureChatGenerator(
            signature="text -> summary",
            output_field="summary",
        )
        messages = [ChatMessage.from_user("Summarize this text")]
        response = component.run(messages=messages)

        assert response["replies"][0].text == "This is a summary"

    def test_run_with_input_mapping(self, mock_dspy_module):
        component = DSPySignatureChatGenerator(
            signature="context, question -> answer",
            input_mapping={"context": "context", "question": "question"},
        )
        messages = [ChatMessage.from_user("What is ML?")]
        component.run(messages=messages, context="Machine learning is a subset of AI.")

        call_kwargs = mock_dspy_module.call_args.kwargs
        assert call_kwargs.get("context") == "Machine learning is a subset of AI."
        assert call_kwargs.get("question") == "What is ML?"

    def test_run_with_wrong_model(self, mock_dspy_module):
        mock_dspy_module.side_effect = Exception("Invalid model name")

        generator = DSPySignatureChatGenerator(
            signature="question -> answer",
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
        component = DSPySignatureChatGenerator(signature="question -> answer")
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_predict_module(self):
        """Test using the Predict module type with a string signature."""
        chat_messages = [ChatMessage.from_user("What is 2 + 2?")]
        component = DSPySignatureChatGenerator(
            signature="question -> answer",
            module_type="Predict",
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        assert "4" in results["replies"][0].text

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_signature_class(self):
        """Test using a dspy.Signature class instead of a string signature."""

        class QASignature(dspy.Signature):
            """Answer questions accurately and concisely."""

            question = dspy.InputField(desc="The user's question")
            answer = dspy.OutputField(desc="A clear, concise answer")

        chat_messages = [ChatMessage.from_user("What language is spoken in Brazil?")]
        component = DSPySignatureChatGenerator(
            signature=QASignature,
            module_type="ChainOfThought",
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        assert "Portuguese" in results["replies"][0].text

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_multi_field_signature(self):
        """Test using a multi-input signature with input_mapping."""
        chat_messages = [ChatMessage.from_user("What is the main topic?")]
        component = DSPySignatureChatGenerator(
            signature="context, question -> answer",
            module_type="Predict",
            input_mapping={"context": "context", "question": "question"},
        )
        results = component.run(
            chat_messages,
            context="Python is a popular programming language created by Guido van Rossum.",
        )
        assert len(results["replies"]) == 1
        assert results["replies"][0].text
