# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret

from haystack_integrations.components.translators.lara import LaraDocumentTranslator


class TestLaraDocumentTranslator:
    @pytest.fixture
    def mock_translation_response(self):
        mock_block = MagicMock()
        mock_block.text = "Translated text"
        mock_response = MagicMock()
        mock_response.translation = [mock_block]
        return mock_response

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("LARA_ACCESS_KEY_ID", "test-access-key-id")
        monkeypatch.setenv("LARA_ACCESS_KEY_SECRET", "test-access-key-secret")
        translator = LaraDocumentTranslator()
        assert translator.access_key_id.resolve_value() == "test-access-key-id"
        assert translator.access_key_secret.resolve_value() == "test-access-key-secret"
        assert translator.source_lang is None
        assert translator.target_lang is None
        assert translator.context is None
        assert translator.instructions is None
        assert translator.style == "faithful"
        assert translator.adapt_to is None
        assert translator.glossaries is None
        assert translator.reasoning is False
        assert translator._translator is None

    def test_init_with_parameters(self):
        translator = LaraDocumentTranslator(
            access_key_id=Secret.from_token("key-id"),
            access_key_secret=Secret.from_token("key-secret"),
            source_lang="en-US",
            target_lang="de-DE",
            context="Prior message.",
            instructions=["Be formal"],
            style="fluid",
            adapt_to=["tm-1"],
            glossaries=["glossary-1"],
            reasoning=True,
        )
        assert translator.access_key_id.resolve_value() == "key-id"
        assert translator.access_key_secret.resolve_value() == "key-secret"
        assert translator.source_lang == "en-US"
        assert translator.target_lang == "de-DE"
        assert translator.context == "Prior message."
        assert translator.instructions == ["Be formal"]
        assert translator.style == "fluid"
        assert translator.adapt_to == ["tm-1"]
        assert translator.glossaries == ["glossary-1"]
        assert translator.reasoning is True

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("LARA_ACCESS_KEY_ID", "from-env-id")
        monkeypatch.setenv("LARA_ACCESS_KEY_SECRET", "from-env-secret")
        data = {
            "type": "haystack_integrations.components.translators.lara.document_translator.LaraDocumentTranslator",
            "init_parameters": {
                "access_key_id": {"env_vars": ["LARA_ACCESS_KEY_ID"], "strict": False, "type": "env_var"},
                "access_key_secret": {"env_vars": ["LARA_ACCESS_KEY_SECRET"], "strict": False, "type": "env_var"},
                "source_lang": "en",
                "target_lang": "fr",
                "style": "creative",
            },
        }
        translator = component_from_dict(LaraDocumentTranslator, data, "document_translator")
        assert translator.source_lang == "en"
        assert translator.target_lang == "fr"
        assert translator.style == "creative"
        assert translator.access_key_id.resolve_value() == "from-env-id"
        assert translator.access_key_secret.resolve_value() == "from-env-secret"

    def test_to_dict(self):
        translator = LaraDocumentTranslator(
            source_lang="en-US",
            target_lang="de-DE",
        )
        data = component_to_dict(translator, "document_translator")
        assert data["type"] == (
            "haystack_integrations.components.translators.lara.document_translator.LaraDocumentTranslator"
        )
        init_params = data["init_parameters"]
        assert init_params["source_lang"] == "en-US"
        assert init_params["target_lang"] == "de-DE"
        assert "access_key_id" in init_params
        assert "access_key_secret" in init_params

    def test_run_single_document(self, mock_translation_response):
        translator = LaraDocumentTranslator(
            access_key_id=Secret.from_token("key-id"),
            access_key_secret=Secret.from_token("key-secret"),
            source_lang="en-US",
            target_lang="de-DE",
        )
        doc = Document(content="Hello, world!", meta={"source": "test"})

        with patch(
            "haystack_integrations.components.translators.lara.document_translator.Translator"
        ) as mock_translator_cls:
            mock_translator = MagicMock()
            mock_translator.translate.return_value = mock_translation_response
            mock_translator_cls.return_value = mock_translator

            result = translator.run(documents=[doc])

        assert len(result["documents"]) == 1
        out_doc = result["documents"][0]
        assert out_doc.content == "Translated text"
        assert out_doc.meta["original_document_id"] == doc.id
        assert out_doc.meta["source"] == "test"
        mock_translator.translate.assert_called_once()
        call_kwargs = mock_translator.translate.call_args.kwargs
        assert call_kwargs["source"] == "en-US"
        assert call_kwargs["target"] == "de-DE"

    def test_run_multiple_documents(self, mock_translation_response):
        translator = LaraDocumentTranslator(
            access_key_id=Secret.from_token("key-id"),
            access_key_secret=Secret.from_token("key-secret"),
            target_lang="de-DE",
        )
        doc1 = Document(content="First")
        doc2 = Document(content="Second")

        with patch(
            "haystack_integrations.components.translators.lara.document_translator.Translator"
        ) as mock_translator_cls:
            mock_translator = MagicMock()
            mock_translator.translate.return_value = mock_translation_response
            mock_translator_cls.return_value = mock_translator

            result = translator.run(documents=[doc1, doc2])

        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "Translated text"
        assert result["documents"][1].content == "Translated text"
        assert result["documents"][0].meta["original_document_id"] == doc1.id
        assert result["documents"][1].meta["original_document_id"] == doc2.id
        assert mock_translator.translate.call_count == 2

    def test_run_with_context_appends_non_translatable_block(self, mock_translation_response):
        translator = LaraDocumentTranslator(
            access_key_id=Secret.from_token("key-id"),
            access_key_secret=Secret.from_token("key-secret"),
            target_lang="de-DE",
            context="Surrounding sentence.",
        )
        doc = Document(content="Translate this.")

        with patch(
            "haystack_integrations.components.translators.lara.document_translator.Translator"
        ) as mock_translator_cls:
            mock_translator = MagicMock()
            mock_translator.translate.return_value = mock_translation_response
            mock_translator_cls.return_value = mock_translator

            translator.run(documents=[doc])

        call_args = mock_translator.translate.call_args
        text_blocks = call_args.kwargs["text"]
        assert len(text_blocks) == 2
        assert text_blocks[0].text == "Translate this."
        assert text_blocks[0].translatable is True
        assert text_blocks[1].text == "Surrounding sentence."
        assert text_blocks[1].translatable is False

    def test_run_overrides_from_init(self, mock_translation_response):
        translator = LaraDocumentTranslator(
            access_key_id=Secret.from_token("key-id"),
            access_key_secret=Secret.from_token("key-secret"),
            source_lang="en-US",
            target_lang="fr-FR",
            style="faithful",
        )
        doc = Document(content="Hi")

        with patch(
            "haystack_integrations.components.translators.lara.document_translator.Translator"
        ) as mock_translator_cls:
            mock_translator = MagicMock()
            mock_translator.translate.return_value = mock_translation_response
            mock_translator_cls.return_value = mock_translator

            translator.run(
                documents=[doc],
                source_lang="en-GB",
                target_lang="es-ES",
                style="creative",
            )

        call_kwargs = mock_translator.translate.call_args.kwargs
        assert call_kwargs["source"] == "en-GB"
        assert call_kwargs["target"] == "es-ES"
        assert call_kwargs["style"] == "creative"

    def test_run_no_documents(self):
        translator = LaraDocumentTranslator(
            access_key_id=Secret.from_token("key-id"),
            access_key_secret=Secret.from_token("key-secret"),
            target_lang="de-DE",
        )
        with patch(
            "haystack_integrations.components.translators.lara.document_translator.Translator"
        ) as mock_translator_cls:
            mock_translator = MagicMock()
            mock_translator_cls.return_value = mock_translator
            result = translator.run(documents=[])
        assert result["documents"] == []
        mock_translator.translate.assert_not_called()

    def test_run_with_no_content_document(self):
        translator = LaraDocumentTranslator(
            access_key_id=Secret.from_token("key-id"),
            access_key_secret=Secret.from_token("key-secret"),
            target_lang="de-DE",
        )
        doc = Document(content="")
        with patch(
            "haystack_integrations.components.translators.lara.document_translator.Translator"
        ) as mock_translator_cls:
            mock_translator = MagicMock()
            mock_translator_cls.return_value = mock_translator
            result = translator.run(documents=[doc])

        assert result["documents"][0].content == ""
        assert result["documents"][0].meta["original_document_id"] == doc.id
        mock_translator.translate.assert_not_called()

    def test_validate_params_scalar_expanded_to_list(self):
        translator = LaraDocumentTranslator()
        result = translator._validate_params(
            num_documents=3,
            source_lang="en-US",
            target_lang="de-DE",
            context="ctx",
            style="fluid",
            instructions="Be formal",
            adapt_to=["tm-1"],
            glossaries=["g1"],
            reasoning=False,
        )
        assert result["source_lang"] == ["en-US", "en-US", "en-US"]
        assert result["target_lang"] == ["de-DE", "de-DE", "de-DE"]
        assert result["context"] == ["ctx", "ctx", "ctx"]
        assert result["style"] == ["fluid", "fluid", "fluid"]
        assert result["reasoning"] == [False, False, False]
        assert len(result["instructions"]) == 3
        assert len(result["adapt_to"]) == 3
        assert len(result["glossaries"]) == 3

    def test_validate_params_list_per_document(self):
        translator = LaraDocumentTranslator()
        result = translator._validate_params(
            num_documents=2,
            source_lang=["en-US", "fr-FR"],
            target_lang=["de-DE", "es-ES"],
            context=None,
            style=["faithful", "creative"],
            instructions=[["Formal"], ["Casual"]],
            adapt_to=[["tm-1"], ["tm-2"]],
            glossaries=[["g1"], ["g2"]],
            reasoning=[True, False],
        )
        assert result["source_lang"] == ["en-US", "fr-FR"]
        assert result["target_lang"] == ["de-DE", "es-ES"]
        assert result["style"] == ["faithful", "creative"]
        assert result["instructions"] == [["Formal"], ["Casual"]]
        assert result["adapt_to"] == [["tm-1"], ["tm-2"]]
        assert result["glossaries"] == [["g1"], ["g2"]]
        assert result["reasoning"] == [True, False]

    def test_validate_params_source_lang_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="source language is a list, it must be the same length"):
            translator = LaraDocumentTranslator()
            translator._validate_params(
                num_documents=2,
                source_lang=["en-US", "fr-FR", "es-ES"],
                target_lang="de-DE",
                context=None,
                style=None,
                instructions=None,
                adapt_to=None,
                glossaries=None,
                reasoning=None,
            )

    def test_validate_params_target_lang_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="target language is a list, it must be the same length"):
            translator = LaraDocumentTranslator()
            translator._validate_params(
                num_documents=2,
                source_lang=None,
                target_lang=["de-DE"],
                context=None,
                style=None,
                instructions=None,
                adapt_to=None,
                glossaries=None,
                reasoning=None,
            )

    def test_validate_params_context_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="context is a list, it must be the same length"):
            translator = LaraDocumentTranslator()
            translator._validate_params(
                num_documents=2,
                source_lang=None,
                target_lang="de-DE",
                context=["c1", "c2", "c3"],
                style=None,
                instructions=None,
                adapt_to=None,
                glossaries=None,
                reasoning=None,
            )

    def test_validate_params_style_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="style is a list, it must be the same length"):
            translator = LaraDocumentTranslator()
            translator._validate_params(
                num_documents=2,
                source_lang=None,
                target_lang="de-DE",
                context=None,
                style=["faithful"],
                instructions=None,
                adapt_to=None,
                glossaries=None,
                reasoning=None,
            )

    def test_validate_params_instructions_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="instructions is a list, it must be the same length"):
            translator = LaraDocumentTranslator()
            translator._validate_params(
                num_documents=2,
                source_lang=None,
                target_lang="de-DE",
                context=None,
                style=None,
                instructions=[["a"], ["b"], ["c"]],
                adapt_to=None,
                glossaries=None,
                reasoning=None,
            )

    def test_validate_params_reasoning_list_wrong_length_raises(self):
        with pytest.raises(ValueError, match="reasoning is a list, it must be the same length"):
            translator = LaraDocumentTranslator()
            translator._validate_params(
                num_documents=2,
                source_lang=None,
                target_lang="de-DE",
                context=None,
                style=None,
                instructions=None,
                adapt_to=None,
                glossaries=None,
                reasoning=[True, False, True],
            )

    def test_validate_params_adapt_to_list_of_lists_wrong_length_raises(self):
        with pytest.raises(ValueError, match="adapt to is a list of lists, it must be the same length"):
            translator = LaraDocumentTranslator()
            translator._validate_params(
                num_documents=2,
                source_lang=None,
                target_lang="de-DE",
                context=None,
                style=None,
                instructions=None,
                adapt_to=[["tm-1"], ["tm-2"], ["tm-3"]],
                glossaries=None,
                reasoning=None,
            )

    def test_validate_params_glossaries_list_of_lists_wrong_length_raises(self):
        with pytest.raises(ValueError, match="glossaries is a list of lists, it must be the same length"):
            translator = LaraDocumentTranslator()
            translator._validate_params(
                num_documents=2,
                source_lang=None,
                target_lang="de-DE",
                context=None,
                style=None,
                instructions=None,
                adapt_to=None,
                glossaries=[["g1"], ["g2"], ["g3"]],
                reasoning=None,
            )

    def test_run_list_param_length_mismatch_raises(self):
        translator = LaraDocumentTranslator(
            access_key_id=Secret.from_token("key-id"),
            access_key_secret=Secret.from_token("key-secret"),
            target_lang="de-DE",
        )
        doc1 = Document(content="One")
        doc2 = Document(content="Two")

        with patch("haystack_integrations.components.translators.lara.document_translator.Translator"):
            with pytest.raises(ValueError, match="source language is a list, it must be the same length"):
                translator.run(
                    documents=[doc1, doc2],
                    source_lang=["en-US"],
                )

    def test_run_per_document_params(self):
        translator = LaraDocumentTranslator(
            access_key_id=Secret.from_token("key-id"),
            access_key_secret=Secret.from_token("key-secret"),
            source_lang="en-US",
            target_lang="de-DE",
        )
        doc1 = Document(content="One")
        doc2 = Document(content="Two")

        responses = [
            MagicMock(translation=[MagicMock(text="Eins")]),
            MagicMock(translation=[MagicMock(text="Zwei")]),
        ]

        with patch(
            "haystack_integrations.components.translators.lara.document_translator.Translator"
        ) as mock_translator_cls:
            mock_translator = MagicMock()
            mock_translator.translate.side_effect = responses
            mock_translator_cls.return_value = mock_translator

            translator.run(
                documents=[doc1, doc2],
                source_lang=["en-US", "fr-FR"],
                target_lang=["de-DE", "es-ES"],
            )

        assert mock_translator.translate.call_count == 2
        first_call = mock_translator.translate.call_args_list[0].kwargs
        second_call = mock_translator.translate.call_args_list[1].kwargs
        assert first_call["source"] == "en-US" and first_call["target"] == "de-DE"
        assert second_call["source"] == "fr-FR" and second_call["target"] == "es-ES"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("LARA_ACCESS_KEY_ID") or not os.environ.get("LARA_ACCESS_KEY_SECRET"),
        reason="LARA_ACCESS_KEY_ID and LARA_ACCESS_KEY_SECRET must be set to run integration tests",
    )
    def test_run_translates_document_via_lara_api(self):
        """Translates a document using the real Lara API (no mocks)."""
        translator = LaraDocumentTranslator(
            source_lang="en-US",
            target_lang="de-DE",
        )
        doc = Document(content="Hello, world!", meta={"source": "integration_test"})

        result = translator.run(documents=[doc])
        assert result["documents"][0].content
        assert result["documents"][0].content != doc.content
        assert "Hallo" in result["documents"][0].content or "Welt" in result["documents"][0].content
