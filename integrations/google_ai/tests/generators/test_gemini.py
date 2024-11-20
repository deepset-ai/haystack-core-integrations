import os
from unittest.mock import patch

import pytest
from google.generativeai import GenerationConfig, GenerativeModel
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from haystack.dataclasses import StreamingChunk

from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator


def test_init(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    generation_config = GenerationConfig(
        candidate_count=1,
        stop_sequences=["stop"],
        max_output_tokens=10,
        temperature=0.5,
        top_p=0.5,
        top_k=0.5,
    )
    safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}

    with patch("haystack_integrations.components.generators.google_ai.gemini.genai.configure") as mock_genai_configure:
        gemini = GoogleAIGeminiGenerator(
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
    mock_genai_configure.assert_called_once_with(api_key="test")
    assert gemini._model_name == "gemini-1.5-flash"
    assert gemini._generation_config == generation_config
    assert gemini._safety_settings == safety_settings
    assert isinstance(gemini._model, GenerativeModel)


def test_init_fails_with_tools():
    with pytest.raises(TypeError, match="GoogleAIGeminiGenerator does not support the `tools` parameter."):
        GoogleAIGeminiGenerator(tools=["tool1", "tool2"])


def test_to_dict(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    with patch("haystack_integrations.components.generators.google_ai.gemini.genai.configure"):
        gemini = GoogleAIGeminiGenerator()
    assert gemini.to_dict() == {
        "type": "haystack_integrations.components.generators.google_ai.gemini.GoogleAIGeminiGenerator",
        "init_parameters": {
            "model": "gemini-1.5-flash",
            "api_key": {"env_vars": ["GOOGLE_API_KEY"], "strict": True, "type": "env_var"},
            "generation_config": None,
            "safety_settings": None,
            "streaming_callback": None,
        },
    }


def test_to_dict_with_param(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    generation_config = GenerationConfig(
        candidate_count=1,
        stop_sequences=["stop"],
        max_output_tokens=10,
        temperature=0.5,
        top_p=0.5,
        top_k=2,
    )
    safety_settings = {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}

    with patch("haystack_integrations.components.generators.google_ai.gemini.genai.configure"):
        gemini = GoogleAIGeminiGenerator(
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
    assert gemini.to_dict() == {
        "type": "haystack_integrations.components.generators.google_ai.gemini.GoogleAIGeminiGenerator",
        "init_parameters": {
            "model": "gemini-1.5-flash",
            "api_key": {"env_vars": ["GOOGLE_API_KEY"], "strict": True, "type": "env_var"},
            "generation_config": {
                "temperature": 0.5,
                "top_p": 0.5,
                "top_k": 2,
                "candidate_count": 1,
                "max_output_tokens": 10,
                "stop_sequences": ["stop"],
            },
            "safety_settings": {10: 3},
            "streaming_callback": None,
        },
    }


def test_from_dict_with_param(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    with patch("haystack_integrations.components.generators.google_ai.gemini.genai.configure"):
        gemini = GoogleAIGeminiGenerator.from_dict(
            {
                "type": "haystack_integrations.components.generators.google_ai.gemini.GoogleAIGeminiGenerator",
                "init_parameters": {
                    "model": "gemini-1.5-flash",
                    "generation_config": {
                        "temperature": 0.5,
                        "top_p": 0.5,
                        "top_k": 0.5,
                        "candidate_count": 1,
                        "max_output_tokens": 10,
                        "stop_sequences": ["stop"],
                    },
                    "safety_settings": {10: 3},
                    "streaming_callback": None,
                },
            }
        )

    assert gemini._model_name == "gemini-1.5-flash"
    assert gemini._generation_config == GenerationConfig(
        candidate_count=1,
        stop_sequences=["stop"],
        max_output_tokens=10,
        temperature=0.5,
        top_p=0.5,
        top_k=0.5,
    )
    assert gemini._safety_settings == {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}
    assert isinstance(gemini._model, GenerativeModel)


def test_from_dict(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    with patch("haystack_integrations.components.generators.google_ai.gemini.genai.configure"):
        gemini = GoogleAIGeminiGenerator.from_dict(
            {
                "type": "haystack_integrations.components.generators.google_ai.gemini.GoogleAIGeminiGenerator",
                "init_parameters": {
                    "model": "gemini-1.5-flash",
                    "generation_config": {
                        "temperature": 0.5,
                        "top_p": 0.5,
                        "top_k": 0.5,
                        "candidate_count": 1,
                        "max_output_tokens": 10,
                        "stop_sequences": ["stop"],
                    },
                    "safety_settings": {10: 3},
                    "streaming_callback": None,
                },
            }
        )

    assert gemini._model_name == "gemini-1.5-flash"
    assert gemini._generation_config == GenerationConfig(
        candidate_count=1,
        stop_sequences=["stop"],
        max_output_tokens=10,
        temperature=0.5,
        top_p=0.5,
        top_k=0.5,
    )
    assert gemini._safety_settings == {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH}
    assert isinstance(gemini._model, GenerativeModel)


@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY", None), reason="GOOGLE_API_KEY env var not set")
def test_run():
    gemini = GoogleAIGeminiGenerator(model="gemini-pro")
    res = gemini.run("Tell me something cool")
    assert len(res["replies"]) > 0


@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY", None), reason="GOOGLE_API_KEY env var not set")
def test_run_with_streaming_callback():
    streaming_callback_called = False

    def streaming_callback(_chunk: StreamingChunk) -> None:
        nonlocal streaming_callback_called
        streaming_callback_called = True

    gemini = GoogleAIGeminiGenerator(model="gemini-pro", streaming_callback=streaming_callback)
    res = gemini.run("Tell me something cool")
    assert len(res["replies"]) > 0
    assert streaming_callback_called
