# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack.utils import Secret

from haystack_integrations.components.converters.mistral import MistralOCRDocumentConverter


class TestMistralOCRDocumentConverter:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        converter = MistralOCRDocumentConverter()

        assert converter.api_key == Secret.from_env_var("MISTRAL_API_KEY")
        assert converter.model == "mistral-ocr-2505"
        assert converter.include_image_base64 is False
        assert converter.pages is None
        assert converter.image_limit is None
        assert converter.image_min_size is None

    def test_init_with_parameters(self):
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key"),
            model="mistral-ocr-custom",
            include_image_base64=True,
        )

        assert converter.api_key == Secret.from_token("test-api-key")
        assert converter.model == "mistral-ocr-custom"
        assert converter.include_image_base64 is True
        assert converter.pages is None
        assert converter.image_limit is None
        assert converter.image_min_size is None

    def test_init_with_all_optional_parameters(self):
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key"),
            model="mistral-ocr-custom",
            include_image_base64=True,
            pages=[0, 1, 2],
            image_limit=10,
            image_min_size=100,
        )

        assert converter.api_key == Secret.from_token("test-api-key")
        assert converter.model == "mistral-ocr-custom"
        assert converter.include_image_base64 is True
        assert converter.pages == [0, 1, 2]
        assert converter.image_limit == 10
        assert converter.image_min_size == 100

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        converter = MistralOCRDocumentConverter()
        converter_dict = converter.to_dict()

        assert converter_dict == {
            "type": "haystack_integrations.components.converters.mistral.ocr_document_converter.MistralOCRDocumentConverter",
            "init_parameters": {
                "api_key": {"env_vars": ["MISTRAL_API_KEY"], "strict": True, "type": "env_var"},
                "model": "mistral-ocr-2505",
                "include_image_base64": False,
                "pages": None,
                "image_limit": None,
                "image_min_size": None,
            },
        }

    def test_to_dict_with_custom_parameters(self):
        converter = MistralOCRDocumentConverter(
            api_key=Secret.from_token("test-api-key"),
            model="mistral-ocr-custom",
            include_image_base64=True,
            pages=[0, 1, 2],
            image_limit=10,
            image_min_size=100,
        )
        converter_dict = converter.to_dict()

        assert converter_dict == {
            "type": "haystack_integrations.components.converters.mistral.ocr_document_converter.MistralOCRDocumentConverter",
            "init_parameters": {
                "api_key": {"type": "env_var", "env_vars": ["MISTRAL_API_KEY"], "strict": True},
                "model": "mistral-ocr-custom",
                "include_image_base64": True,
                "pages": [0, 1, 2],
                "image_limit": 10,
                "image_min_size": 100,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        converter_dict = {
            "type": "haystack_integrations.components.converters.mistral.ocr_document_converter.MistralOCRDocumentConverter",
            "init_parameters": {
                "api_key": {"env_vars": ["MISTRAL_API_KEY"], "strict": True, "type": "env_var"},
                "model": "mistral-ocr-2505",
                "include_image_base64": False,
                "pages": None,
                "image_limit": None,
                "image_min_size": None,
            },
        }

        converter = MistralOCRDocumentConverter.from_dict(converter_dict)

        assert converter.model == "mistral-ocr-2505"
        assert converter.include_image_base64 is False
        assert converter.pages is None
        assert converter.image_limit is None
        assert converter.image_min_size is None

    def test_from_dict_with_custom_parameters(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        converter_dict = {
            "type": "haystack_integrations.components.converters.mistral.ocr_document_converter.MistralOCRDocumentConverter",
            "init_parameters": {
                "api_key": {"env_vars": ["MISTRAL_API_KEY"], "strict": True, "type": "env_var"},
                "model": "mistral-ocr-custom",
                "include_image_base64": True,
                "pages": [0, 1, 2],
                "image_limit": 10,
                "image_min_size": 100,
            },
        }

        converter = MistralOCRDocumentConverter.from_dict(converter_dict)

        assert converter.model == "mistral-ocr-custom"
        assert converter.include_image_base64 is True
        assert converter.pages == [0, 1, 2]
        assert converter.image_limit == 10
        assert converter.image_min_size == 100
