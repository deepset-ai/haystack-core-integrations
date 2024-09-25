# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
from cohere.core import ApiError
from haystack.components.generators.utils import print_streaming_chunk
from haystack.utils import Secret
from haystack_integrations.components.generators.cohere import CohereGenerator

pytestmark = pytest.mark.generators
COHERE_API_URL = "https://api.cohere.com"


class TestCohereGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "foo")
        component = CohereGenerator()
        assert component.api_key == Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"])
        assert component.model == "command-r"
        assert component.streaming_callback is None
        assert component.api_base_url == COHERE_API_URL
        assert component.model_parameters == {}

    def test_init_with_parameters(self):
        callback = lambda x: x  # noqa: E731
        component = CohereGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="command-light",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=callback,
            api_base_url="test-base-url",
        )
        assert component.api_key == Secret.from_token("test-api-key")
        assert component.model == "command-light"
        assert component.streaming_callback == callback
        assert component.api_base_url == "test-base-url"
        assert component.model_parameters == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        component = CohereGenerator()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.cohere.generator.CohereGenerator",
            "init_parameters": {
                "model": "command-r",
                "api_key": {"env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True, "type": "env_var"},
                "streaming_callback": None,
                "api_base_url": COHERE_API_URL,
                "generation_kwargs": {},
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        monkeypatch.setenv("CO_API_KEY", "fake-api-key")
        component = CohereGenerator(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="command-light",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.cohere.generator.CohereGenerator",
            "init_parameters": {
                "model": "command-light",
                "api_base_url": "test-base-url",
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {},
            },
        }

    def test_to_dict_with_lambda_streaming_callback(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        component = CohereGenerator(
            model="command-r",
            max_tokens=10,
            some_test_param="test-params",
            streaming_callback=lambda x: x,
            api_base_url="test-base-url",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.cohere.generator.CohereGenerator",
            "init_parameters": {
                "model": "command-r",
                "streaming_callback": "tests.test_cohere_generator.<lambda>",
                "api_base_url": "test-base-url",
                "api_key": {"type": "env_var", "env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True},
                "generation_kwargs": {},
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "fake-api-key")
        monkeypatch.setenv("CO_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.cohere.generator.CohereGenerator",
            "init_parameters": {
                "model": "command-r",
                "max_tokens": 10,
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "some_test_param": "test-params",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            },
        }
        component: CohereGenerator = CohereGenerator.from_dict(data)
        assert component.api_key == Secret.from_env_var("ENV_VAR", strict=False)
        assert component.model == "command-r"
        assert component.streaming_callback == print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.model_parameters == {"max_tokens": 10, "some_test_param": "test-params"}

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_cohere_generator_run(self):
        component = CohereGenerator()
        results = component.run(prompt="What's the capital of France?")
        assert len(results["replies"]) == 1
        assert "Paris" in results["replies"][0]
        assert len(results["meta"]) == 1
        assert results["meta"][0]["finish_reason"] == "COMPLETE"

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_cohere_generator_run_wrong_model(self):
        component = CohereGenerator(model="something-obviously-wrong")
        with pytest.raises(ApiError):
            component.run(prompt="What's the capital of France?")

    @pytest.mark.skipif(
        not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
        reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
    )
    @pytest.mark.integration
    def test_cohere_generator_run_streaming(self):
        class Callback:
            def __init__(self):
                self.responses = ""

            def __call__(self, chunk):
                self.responses += chunk.content
                return chunk

        callback = Callback()
        component = CohereGenerator(streaming_callback=callback)
        results = component.run(prompt="What's the capital of France?")

        assert len(results["replies"]) == 1
        assert "Paris" in results["replies"][0]
        assert len(results["meta"]) == 1
        assert results["meta"][0]["finish_reason"] == "COMPLETE"
        assert callback.responses == results["replies"][0]
