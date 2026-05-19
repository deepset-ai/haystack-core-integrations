# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import MagicMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict

from haystack_integrations.components.preprocessors.presidio import PresidioDocumentCleaner


class TestPresidioDocumentCleaner:
    def test_init_defaults(self):
        cleaner = PresidioDocumentCleaner()
        assert cleaner.language == "en"
        assert cleaner.entities is None
        assert cleaner.score_threshold == 0.35
        assert cleaner.models is None

    def test_init_custom_params(self):
        cleaner = PresidioDocumentCleaner(language="de", entities=["PERSON"], score_threshold=0.7)
        assert cleaner.language == "de"
        assert cleaner.entities == ["PERSON"]
        assert cleaner.score_threshold == 0.7

    def test_to_dict(self):
        models = [{"lang_code": "fr", "model_name": "fr_core_news_lg"}]
        cleaner = PresidioDocumentCleaner(language="fr", entities=["EMAIL_ADDRESS"], score_threshold=0.5, models=models)
        data = component_to_dict(cleaner, "PresidioDocumentCleaner")
        expected_type = (
            "haystack_integrations.components.preprocessors.presidio.presidio_document_cleaner.PresidioDocumentCleaner"
        )
        assert data["type"] == expected_type
        assert data["init_parameters"]["language"] == "fr"
        assert data["init_parameters"]["entities"] == ["EMAIL_ADDRESS"]
        assert data["init_parameters"]["score_threshold"] == 0.5
        assert data["init_parameters"]["models"] == models

    def test_from_dict(self):
        models = [{"lang_code": "de", "model_name": "de_core_news_lg"}]
        data = {
            "type": (
                "haystack_integrations.components.preprocessors.presidio"
                ".presidio_document_cleaner.PresidioDocumentCleaner"
            ),
            "init_parameters": {"language": "de", "entities": ["PERSON"], "score_threshold": 0.6, "models": models},
        }
        cleaner = component_from_dict(PresidioDocumentCleaner, data, "PresidioDocumentCleaner")
        assert cleaner.language == "de"
        assert cleaner.entities == ["PERSON"]
        assert cleaner.score_threshold == 0.6
        assert cleaner.models == models

    def test_warm_up_auto_model(self):
        cleaner = PresidioDocumentCleaner(language="en")
        mock_nlp_engine = MagicMock()
        with (
            patch(
                "haystack_integrations.components.preprocessors.presidio.presidio_document_cleaner.NlpEngineProvider"
            ) as mock_provider_cls,
            patch(
                "haystack_integrations.components.preprocessors.presidio.presidio_document_cleaner.AnalyzerEngine"
            ) as mock_analyzer_cls,
            patch("haystack_integrations.components.preprocessors.presidio.presidio_document_cleaner.AnonymizerEngine"),
        ):
            mock_provider_cls.return_value.create_engine.return_value = mock_nlp_engine
            cleaner.warm_up()
            mock_provider_cls.assert_called_once_with(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
                }
            )
            mock_analyzer_cls.assert_called_once_with(nlp_engine=mock_nlp_engine, supported_languages=["en"])

    def test_warm_up_unknown_language_raises(self):
        cleaner = PresidioDocumentCleaner(language="xx")
        with pytest.raises(ValueError, match="No default spaCy model is available for language 'xx'"):
            cleaner.warm_up()

    def test_warm_up_with_models(self):
        models = [{"lang_code": "fr", "model_name": "fr_core_news_lg"}]
        cleaner = PresidioDocumentCleaner(language="fr", models=models)
        mock_nlp_engine = MagicMock()
        with (
            patch(
                "haystack_integrations.components.preprocessors.presidio.presidio_document_cleaner.NlpEngineProvider"
            ) as mock_provider_cls,
            patch(
                "haystack_integrations.components.preprocessors.presidio.presidio_document_cleaner.AnalyzerEngine"
            ) as mock_analyzer_cls,
            patch("haystack_integrations.components.preprocessors.presidio.presidio_document_cleaner.AnonymizerEngine"),
        ):
            mock_provider_cls.return_value.create_engine.return_value = mock_nlp_engine
            cleaner.warm_up()
            mock_provider_cls.assert_called_once_with(nlp_configuration={"nlp_engine_name": "spacy", "models": models})
            mock_analyzer_cls.assert_called_once_with(nlp_engine=mock_nlp_engine, supported_languages=["fr"])

    def _make_cleaner_with_mocks(self, **kwargs):
        """Return a cleaner with mocked engines so unit tests don't load real NLP models."""
        cleaner = PresidioDocumentCleaner(**kwargs)
        cleaner._analyzer = MagicMock()
        cleaner._anonymizer = MagicMock()
        cleaner._is_warmed_up = True
        return cleaner

    def test_run_anonymizes_pii(self):
        cleaner = self._make_cleaner_with_mocks()
        mock_result = MagicMock()
        mock_result.text = "My name is <PERSON> and email is <EMAIL_ADDRESS>"
        cleaner._anonymizer.anonymize.return_value = mock_result
        cleaner._analyzer.analyze.return_value = []

        docs = [Document(content="My name is John and email is john@example.com")]
        result = cleaner.run(documents=docs)

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "My name is <PERSON> and email is <EMAIL_ADDRESS>"

    def test_run_preserves_metadata(self):
        cleaner = self._make_cleaner_with_mocks()
        mock_result = MagicMock()
        mock_result.text = "Hello <PERSON>"
        cleaner._anonymizer.anonymize.return_value = mock_result
        cleaner._analyzer.analyze.return_value = []

        docs = [Document(content="Hello John", meta={"source": "email", "page": 1})]
        result = cleaner.run(documents=docs)

        assert result["documents"][0].meta["source"] == "email"
        assert result["documents"][0].meta["page"] == 1

    def test_run_does_not_mutate_original(self):
        cleaner = self._make_cleaner_with_mocks()
        mock_result = MagicMock()
        mock_result.text = "Hello <PERSON>"
        cleaner._anonymizer.anonymize.return_value = mock_result
        cleaner._analyzer.analyze.return_value = []

        original = Document(content="Hello John")
        cleaner.run(documents=[original])

        assert original.content == "Hello John"

    def test_run_passes_through_none_content(self):
        cleaner = self._make_cleaner_with_mocks()
        doc = Document(content=None, meta={"source": "test"})
        result = cleaner.run(documents=[doc])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content is None
        assert result["documents"][0].meta["source"] == "test"

    def test_run_skips_on_error(self, caplog):
        cleaner = self._make_cleaner_with_mocks()
        cleaner._analyzer.analyze.side_effect = Exception("Analyzer error")

        doc = Document(content="Some text with PII")
        with caplog.at_level(logging.WARNING):
            result = cleaner.run(documents=[doc])

        assert len(result["documents"]) == 0
        assert "Could not anonymize" in caplog.text

    def test_run_multiple_documents(self):
        cleaner = self._make_cleaner_with_mocks()
        mock_result = MagicMock()
        mock_result.text = "cleaned"
        cleaner._anonymizer.anonymize.return_value = mock_result
        cleaner._analyzer.analyze.return_value = []

        docs = [Document(content=f"doc {i}") for i in range(3)]
        result = cleaner.run(documents=docs)

        assert len(result["documents"]) == 3

    def test_run_passes_language_and_entities_to_analyzer(self):
        cleaner = self._make_cleaner_with_mocks(language="de", entities=["PERSON"], score_threshold=0.8)
        mock_result = MagicMock()
        mock_result.text = "cleaned"
        cleaner._anonymizer.anonymize.return_value = mock_result
        cleaner._analyzer.analyze.return_value = []

        cleaner.run(documents=[Document(content="Hello John")])

        cleaner._analyzer.analyze.assert_called_once_with(
            text="Hello John", language="de", entities=["PERSON"], score_threshold=0.8
        )

    @pytest.mark.integration
    def test_run_integration(self):
        cleaner = PresidioDocumentCleaner()
        cleaner.warm_up()
        docs = [Document(content="My name is John Smith and my email is john@example.com")]
        result = cleaner.run(documents=docs)

        assert len(result["documents"]) == 1
        assert "John Smith" not in result["documents"][0].content
        assert "john@example.com" not in result["documents"][0].content

    @pytest.mark.integration
    def test_run_integration_german(self):
        cleaner = PresidioDocumentCleaner(language="de")
        cleaner.warm_up()
        docs = [Document(content="Mein Name ist Hans Müller und meine E-Mail ist hans@example.com")]
        result = cleaner.run(documents=docs)

        assert len(result["documents"]) == 1
        assert "Hans Müller" not in result["documents"][0].content
        assert "hans@example.com" not in result["documents"][0].content
