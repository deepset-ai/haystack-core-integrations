# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import MagicMock, patch

import pytest
from haystack.core.serialization import component_from_dict, component_to_dict

from haystack_integrations.components.preprocessors.presidio import PresidioTextCleaner


class TestPresidioTextCleaner:
    def test_init_defaults(self):
        cleaner = PresidioTextCleaner()
        assert cleaner.language == "en"
        assert cleaner.entities is None
        assert cleaner.score_threshold == 0.35
        assert cleaner.models is None

    def test_to_dict(self):
        models = [{"lang_code": "es", "model_name": "es_core_news_lg"}]
        cleaner = PresidioTextCleaner(language="es", entities=["PHONE_NUMBER"], score_threshold=0.5, models=models)
        data = component_to_dict(cleaner, "PresidioTextCleaner")
        assert (
            data["type"]
            == "haystack_integrations.components.preprocessors.presidio.presidio_text_cleaner.PresidioTextCleaner"
        )
        assert data["init_parameters"]["language"] == "es"
        assert data["init_parameters"]["entities"] == ["PHONE_NUMBER"]
        assert data["init_parameters"]["models"] == models

    def test_from_dict(self):
        models = [{"lang_code": "es", "model_name": "es_core_news_lg"}]
        data = {
            "type": "haystack_integrations.components.preprocessors.presidio.presidio_text_cleaner.PresidioTextCleaner",
            "init_parameters": {"language": "es", "entities": None, "score_threshold": 0.4, "models": models},
        }
        cleaner = component_from_dict(PresidioTextCleaner, data, "PresidioTextCleaner")
        assert cleaner.score_threshold == 0.4
        assert cleaner.models == models

    def test_warm_up_auto_model(self):
        cleaner = PresidioTextCleaner(language="es")
        mock_nlp_engine = MagicMock()
        with (
            patch(
                "haystack_integrations.components.preprocessors.presidio.presidio_text_cleaner.NlpEngineProvider"
            ) as mock_provider_cls,
            patch(
                "haystack_integrations.components.preprocessors.presidio.presidio_text_cleaner.AnalyzerEngine"
            ) as mock_analyzer_cls,
            patch("haystack_integrations.components.preprocessors.presidio.presidio_text_cleaner.AnonymizerEngine"),
        ):
            mock_provider_cls.return_value.create_engine.return_value = mock_nlp_engine
            cleaner.warm_up()
            mock_provider_cls.assert_called_once_with(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "es", "model_name": "es_core_news_lg"}],
                }
            )
            mock_analyzer_cls.assert_called_once_with(nlp_engine=mock_nlp_engine, supported_languages=["es"])

    def test_warm_up_unknown_language_raises(self):
        cleaner = PresidioTextCleaner(language="xx")
        with pytest.raises(ValueError, match="No default spaCy model is available for language 'xx'"):
            cleaner.warm_up()

    def test_warm_up_with_models(self):
        models = [{"lang_code": "es", "model_name": "es_core_news_lg"}]
        cleaner = PresidioTextCleaner(language="es", models=models)
        mock_nlp_engine = MagicMock()
        with (
            patch(
                "haystack_integrations.components.preprocessors.presidio.presidio_text_cleaner.NlpEngineProvider"
            ) as mock_provider_cls,
            patch(
                "haystack_integrations.components.preprocessors.presidio.presidio_text_cleaner.AnalyzerEngine"
            ) as mock_analyzer_cls,
            patch("haystack_integrations.components.preprocessors.presidio.presidio_text_cleaner.AnonymizerEngine"),
        ):
            mock_provider_cls.return_value.create_engine.return_value = mock_nlp_engine
            cleaner.warm_up()
            mock_provider_cls.assert_called_once_with(nlp_configuration={"nlp_engine_name": "spacy", "models": models})
            mock_analyzer_cls.assert_called_once_with(nlp_engine=mock_nlp_engine, supported_languages=["es"])

    def _make_cleaner_with_mocks(self, **kwargs):
        """Return a cleaner with mocked engines so unit tests don't load real NLP models."""
        cleaner = PresidioTextCleaner(**kwargs)
        cleaner._analyzer = MagicMock()
        cleaner._anonymizer = MagicMock()
        cleaner._is_warmed_up = True
        return cleaner

    def test_run_anonymizes_pii(self):
        cleaner = self._make_cleaner_with_mocks()
        mock_result = MagicMock()
        mock_result.text = "Call me at <PHONE_NUMBER>"
        cleaner._anonymizer.anonymize.return_value = mock_result
        cleaner._analyzer.analyze.return_value = []

        result = cleaner.run(texts=["Call me at 212-555-1234"])

        assert result["texts"][0] == "Call me at <PHONE_NUMBER>"

    def test_run_multiple_texts(self):
        cleaner = self._make_cleaner_with_mocks()
        mock_result = MagicMock()
        mock_result.text = "cleaned"
        cleaner._anonymizer.anonymize.return_value = mock_result
        cleaner._analyzer.analyze.return_value = []

        result = cleaner.run(texts=["text 1", "text 2", "text 3"])

        assert len(result["texts"]) == 3

    def test_run_skips_on_error(self, caplog):
        cleaner = self._make_cleaner_with_mocks()
        cleaner._analyzer.analyze.side_effect = Exception("error")

        with caplog.at_level(logging.WARNING):
            result = cleaner.run(texts=["My name is John"])

        assert result["texts"][0] == "My name is John"
        assert "Could not anonymize" in caplog.text

    def test_run_empty_text(self):
        cleaner = self._make_cleaner_with_mocks()
        mock_result = MagicMock()
        mock_result.text = ""
        cleaner._anonymizer.anonymize.return_value = mock_result
        cleaner._analyzer.analyze.return_value = []

        result = cleaner.run(texts=[""])

        assert result["texts"][0] == ""

    @pytest.mark.integration
    def test_run_integration(self):
        cleaner = PresidioTextCleaner()
        cleaner.warm_up()
        result = cleaner.run(texts=["Hi, I am Alice and my phone is 212-555-5678"])

        assert len(result["texts"]) == 1
        assert "Alice" not in result["texts"][0]
        assert "212-555-5678" not in result["texts"][0]

    @pytest.mark.integration
    def test_run_integration_german(self):
        cleaner = PresidioTextCleaner(language="de")
        cleaner.warm_up()
        result = cleaner.run(texts=["Hallo, ich bin Thomas Schmidt und meine E-Mail ist thomas@example.com"])

        assert len(result["texts"]) == 1
        assert "Thomas Schmidt" not in result["texts"][0]
        assert "thomas@example.com" not in result["texts"][0]
