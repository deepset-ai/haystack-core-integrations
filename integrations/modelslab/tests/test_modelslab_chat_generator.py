# SPDX-FileCopyrightText: 2024-present ModelsLab
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest

from modelslab_haystack import ModelsLabChatGenerator
from modelslab_haystack.chat_generator import MODELSLAB_API_BASE


class TestModelsLabChatGeneratorInit:
    """Test constructor and defaults."""

    @patch.dict(os.environ, {"MODELSLAB_API_KEY": "test-key"})
    def test_default_model(self):
        gen = ModelsLabChatGenerator()
        assert gen.model == "llama-3.1-8b-uncensored"

    @patch.dict(os.environ, {"MODELSLAB_API_KEY": "test-key"})
    def test_default_api_base(self):
        gen = ModelsLabChatGenerator()
        assert gen.api_base_url == MODELSLAB_API_BASE

    def test_api_base_value(self):
        assert MODELSLAB_API_BASE == "https://modelslab.com/uncensored-chat/v1"

    @patch.dict(os.environ, {"MODELSLAB_API_KEY": "test-key"})
    def test_custom_model(self):
        gen = ModelsLabChatGenerator(model="llama-3.1-70b-uncensored")
        assert gen.model == "llama-3.1-70b-uncensored"

    @patch.dict(os.environ, {"MODELSLAB_API_KEY": "test-key"})
    def test_inherits_openai_chat_generator(self):
        from haystack.components.generators.chat import OpenAIChatGenerator
        gen = ModelsLabChatGenerator()
        assert isinstance(gen, OpenAIChatGenerator)

    def test_explicit_api_key(self):
        from haystack.utils import Secret
        gen = ModelsLabChatGenerator(
            api_key=Secret.from_token("explicit-key"),
            model="llama-3.1-8b-uncensored",
        )
        assert gen.model == "llama-3.1-8b-uncensored"

    @patch.dict(os.environ, {"MODELSLAB_API_KEY": "test-key"})
    def test_generation_kwargs_passed_through(self):
        gen = ModelsLabChatGenerator(
            generation_kwargs={"temperature": 0.5, "max_tokens": 1024}
        )
        assert gen.generation_kwargs["temperature"] == 0.5
        assert gen.generation_kwargs["max_tokens"] == 1024

    @patch.dict(os.environ, {"MODELSLAB_API_KEY": "test-key"})
    def test_streaming_callback_accepted(self):
        callback = MagicMock()
        gen = ModelsLabChatGenerator(streaming_callback=callback)
        assert gen.model == "llama-3.1-8b-uncensored"

    @patch.dict(os.environ, {"MODELSLAB_API_KEY": "test-key"})
    def test_import_from_package(self):
        from modelslab_haystack import ModelsLabChatGenerator as Gen
        from modelslab_haystack.chat_generator import ModelsLabChatGenerator as GenBase
        assert Gen is GenBase


class TestModelsLabChatGeneratorSerialization:
    """Test to_dict / from_dict round-trip."""

    @patch.dict(os.environ, {"MODELSLAB_API_KEY": "test-key"})
    def test_to_dict_contains_model(self):
        gen = ModelsLabChatGenerator(model="llama-3.1-70b-uncensored")
        d = gen.to_dict()
        assert d["init_parameters"]["model"] == "llama-3.1-70b-uncensored"

    @patch.dict(os.environ, {"MODELSLAB_API_KEY": "test-key"})
    def test_to_dict_contains_api_base(self):
        gen = ModelsLabChatGenerator()
        d = gen.to_dict()
        assert d["init_parameters"]["api_base_url"] == MODELSLAB_API_BASE

    @patch.dict(os.environ, {"MODELSLAB_API_KEY": "test-key"})
    def test_from_dict_round_trip(self):
        gen = ModelsLabChatGenerator(model="llama-3.1-8b-uncensored")
        d = gen.to_dict()
        restored = ModelsLabChatGenerator.from_dict(d)
        assert restored.model == "llama-3.1-8b-uncensored"
        assert restored.api_base_url == MODELSLAB_API_BASE
