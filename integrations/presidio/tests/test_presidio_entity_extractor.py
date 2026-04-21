# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from unittest.mock import MagicMock

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict

from haystack_integrations.components.extractors.presidio import PresidioEntityExtractor


class TestPresidioEntityExtractor:
    def test_init_defaults(self):
        extractor = PresidioEntityExtractor()
        assert extractor.language == "en"
        assert extractor.entities is None
        assert extractor.score_threshold == 0.35

    def _make_extractor_with_mocks(self, **kwargs):
        """Return an extractor with a mocked analyzer so unit tests don't load real NLP models."""
        extractor = PresidioEntityExtractor(**kwargs)
        extractor._analyzer = MagicMock()
        extractor._is_warmed_up = True
        return extractor

    def test_to_dict(self):
        extractor = PresidioEntityExtractor(language="en", entities=["PERSON"], score_threshold=0.6)
        data = component_to_dict(extractor, "PresidioEntityExtractor")
        expected_type = (
            "haystack_integrations.components.extractors.presidio.presidio_entity_extractor.PresidioEntityExtractor"
        )
        assert data["type"] == expected_type
        assert data["init_parameters"]["entities"] == ["PERSON"]
        assert data["init_parameters"]["score_threshold"] == 0.6

    def test_from_dict(self):
        data = {
            "type": (
                "haystack_integrations.components.extractors.presidio.presidio_entity_extractor.PresidioEntityExtractor"
            ),
            "init_parameters": {"language": "en", "entities": ["EMAIL_ADDRESS"], "score_threshold": 0.5},
        }
        extractor = component_from_dict(PresidioEntityExtractor, data, "PresidioEntityExtractor")
        assert extractor.entities == ["EMAIL_ADDRESS"]

    def test_run_extracts_entities_into_metadata(self):
        extractor = self._make_extractor_with_mocks()
        mock_entity = MagicMock()
        mock_entity.entity_type = "PERSON"
        mock_entity.start = 11
        mock_entity.end = 15
        mock_entity.score = 0.85
        extractor._analyzer.analyze.return_value = [mock_entity]

        docs = [Document(content="My name is John")]
        result = extractor.run(documents=docs)

        entities = result["documents"][0].meta["entities"]
        assert len(entities) == 1
        assert entities[0]["entity_type"] == "PERSON"
        assert entities[0]["start"] == 11
        assert entities[0]["end"] == 15
        assert entities[0]["score"] == 0.85

    def test_run_does_not_mutate_original(self):
        extractor = self._make_extractor_with_mocks()
        extractor._analyzer.analyze.return_value = []

        original = Document(content="Hello John", meta={"source": "test"})
        extractor.run(documents=[original])

        assert "entities" not in original.meta

    def test_run_passes_through_none_content(self):
        extractor = self._make_extractor_with_mocks()
        doc = Document(content=None, meta={"source": "test"})
        result = extractor.run(documents=[doc])

        assert result["documents"][0].content is None
        assert "entities" not in result["documents"][0].meta

    def test_run_empty_entities(self):
        extractor = self._make_extractor_with_mocks()
        extractor._analyzer.analyze.return_value = []

        docs = [Document(content="No PII here")]
        result = extractor.run(documents=docs)

        assert result["documents"][0].meta["entities"] == []

    def test_run_skips_on_error(self, caplog):
        extractor = self._make_extractor_with_mocks()
        extractor._analyzer.analyze.side_effect = Exception("Analyzer error")

        doc = Document(content="Some text")
        with caplog.at_level(logging.WARNING):
            result = extractor.run(documents=[doc])

        assert result["documents"][0].content == "Some text"
        assert "entities" not in result["documents"][0].meta
        assert "Could not extract entities" in caplog.text

    def test_run_preserves_existing_metadata(self):
        extractor = self._make_extractor_with_mocks()
        extractor._analyzer.analyze.return_value = []

        docs = [Document(content="Hello", meta={"page": 3, "author": "Bob"})]
        result = extractor.run(documents=docs)

        assert result["documents"][0].meta["page"] == 3
        assert result["documents"][0].meta["author"] == "Bob"
        assert result["documents"][0].meta["entities"] == []

    @pytest.mark.integration
    def test_run_integration(self):
        extractor = PresidioEntityExtractor()
        extractor.warm_up()
        docs = [Document(content="Contact Alice at alice@example.com")]
        result = extractor.run(documents=docs)

        entities = result["documents"][0].meta["entities"]
        entity_types = [e["entity_type"] for e in entities]
        assert "EMAIL_ADDRESS" in entity_types
