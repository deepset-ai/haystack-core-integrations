import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.dspy.generator import DSPyGenerator


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
        mock_module.acall = AsyncMock(return_value=MagicMock(answer="Hello world!"))

        mock_cot_class.return_value = mock_module
        mock_predict_class.return_value = mock_module
        mock_react_class.return_value = mock_module

        yield mock_module


class TestDSPyGenerator:
    def test_init_default(self, monkeypatch, mock_dspy_module):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = DSPyGenerator(signature="question -> answer")
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
            DSPyGenerator(signature="question -> answer")

    def test_init_with_parameters(self, mock_dspy_module):
        component = DSPyGenerator(
            signature="context, question -> answer",
            model="openai/gpt-5-mini",
            api_key=Secret.from_token("test-api-key"),
            module_type="Predict",
            output_field="response",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            input_mapping={"context": "context", "question": "question"},
        )
        assert component.model == "openai/gpt-5-mini"
        assert component.signature == "context, question -> answer"
        assert component.module_type == "Predict"
        assert component.output_field == "response"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.input_mapping == {"context": "context", "question": "question"}

    def test_init_with_invalid_module_type(self, mock_dspy_module):
        with pytest.raises(ValueError, match="Invalid module_type"):
            DSPyGenerator(
                signature="question -> answer",
                api_key=Secret.from_token("test-api-key"),
                module_type="InvalidModule",
            )

    def test_to_dict_default(self, monkeypatch, mock_dspy_module):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = DSPyGenerator(
            signature="question -> answer",
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.dspy.generator.DSPyGenerator",
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
        component = DSPyGenerator(
            signature="context, question -> answer",
            model="openai/gpt-5-mini",
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            module_type="Predict",
            output_field="response",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            input_mapping={"context": "context", "question": "question"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.dspy.generator.DSPyGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "signature": "context, question -> answer",
                "model": "openai/gpt-5-mini",
                "module_type": "Predict",
                "output_field": "response",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "input_mapping": {"context": "context", "question": "question"},
            },
        }

    def test_from_dict(self, monkeypatch, mock_dspy_module):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.dspy.generator.DSPyGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "signature": "question -> answer",
                "model": "openai/gpt-5-mini",
                "module_type": "Predict",
                "output_field": "response",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "input_mapping": None,
            },
        }
        component = DSPyGenerator.from_dict(data)
        assert component.model == "openai/gpt-5-mini"
        assert component.signature == "question -> answer"
        assert component.module_type == "Predict"
        assert component.output_field == "response"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert component.input_mapping is None

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.generators.dspy.generator.DSPyGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "signature": "question -> answer",
                "model": "openai/gpt-5-mini",
                "module_type": "Predict",
                "output_field": "response",
                "generation_kwargs": {},
                "input_mapping": None,
            },
        }
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            DSPyGenerator.from_dict(data)

    def test_run(self, mock_dspy_module):
        component = DSPyGenerator(
            signature="question -> answer",
            api_key=Secret.from_token("test-api-key"),
        )
        response = component.run(prompt="What's Natural Language Processing?")

        # Verify the mock was called
        mock_dspy_module.assert_called_once()

        # Check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], str)

    def test_run_with_params(self, mock_dspy_module):
        component = DSPyGenerator(
            signature="question -> answer",
            api_key=Secret.from_token("test-api-key"),
            generation_kwargs={"max_tokens": 10, "temperature": 0.5},
        )
        response = component.run(
            prompt="What's Natural Language Processing?",
            generation_kwargs={"temperature": 0.9},
        )

        # Check that the component calls the DSPy module with the correct parameters
        _, kwargs = mock_dspy_module.call_args
        assert kwargs["config"] == {"temperature": 0.9}

        # Check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], str)

    def test_run_with_wrong_model(self, mock_dspy_module):
        mock_dspy_module.side_effect = Exception("Invalid model name")

        generator = DSPyGenerator(
            signature="question -> answer",
            api_key=Secret.from_token("test-api-key"),
            model="something-obviously-wrong",
        )

        with pytest.raises(Exception, match="Invalid model name"):
            generator.run(prompt="Whatever")

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        component = DSPyGenerator(signature="question -> answer")
        results = component.run(prompt="What's the capital of France?")
        assert len(results["replies"]) == 1
        assert len(results["meta"]) == 1
        response: str = results["replies"][0]
        assert "Paris" in response

        metadata = results["meta"][0]
        assert metadata["model"] == "openai/gpt-5-mini"
        assert metadata["module_type"] == "ChainOfThought"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.asyncio
    async def test_run_async(self, mock_dspy_module):
        component = DSPyGenerator(
            signature="question -> answer",
            api_key=Secret.from_token("test-api-key"),
        )
        response = await component.run_async(prompt="What's Natural Language Processing?")

        # Verify the async mock was called
        mock_dspy_module.acall.assert_called_once()

        # Check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], str)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.asyncio
    async def test_run_async_with_params(self, mock_dspy_module):
        component = DSPyGenerator(
            signature="question -> answer",
            api_key=Secret.from_token("test-api-key"),
        )
        response = await component.run_async(
            prompt="What's Natural Language Processing?",
            generation_kwargs={"temperature": 0.9},
        )

        # Check that acall was called with the correct parameters
        _, kwargs = mock_dspy_module.acall.call_args
        assert kwargs["config"] == {"temperature": 0.9}

        # Check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], str)
